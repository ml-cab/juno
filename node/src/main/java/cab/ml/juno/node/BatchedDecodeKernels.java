/*
 * Copyright 2026 Dmytro Soloviov (soulaway)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package cab.ml.juno.node;

import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;
import java.util.logging.Logger;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;

/**
 * The kernels of the decode batch device region ({@code batched_decode.ptx}): the
 * K-quant matmul over N activation rows that reads each weight once for all of them,
 * RoPE with a per-row position, and the KV append into each request's own mirror.
 *
 * <p>Every kernel computes, for every row, what one request's decode computes, bit
 * for bit ({@code batched_decode.cu}), so a request's reply does not depend on what
 * is batched with it. CUDA only: the packed K-quant weights are device matrices only
 * on that backend.
 */
final class BatchedDecodeKernels {

	private static final Logger log = Logger.getLogger(BatchedDecodeKernels.class.getName());
	private static final String RESOURCE = "/cab/ml/juno/node/batched_decode.ptx";

	/** Activation rows one matmul launch takes; a wider batch takes several launches. */
	static final int MAX_ROWS_PER_LAUNCH = 8;

	/** Threads per block of the matmul: 4 warps, one weight row per block. */
	private static final int GEMV_THREADS = 128;
	/** Threads per block of RoPE and the KV append. Matches {@code BD_THREADS}. */
	private static final int BLOCK_THREADS = 256;
	private static final int[] ROW_TILES = { 2, 4, 8 };

	private static final AtomicReference<BatchedDecodeKernels> INSTANCE = new AtomicReference<>();
	private static final ThreadLocal<KernelParams> GEMV_PARAMS = ThreadLocal.withInitial(() -> new KernelParams(7));
	private static final ThreadLocal<KernelParams> ROPE_PARAMS = ThreadLocal.withInitial(() -> new KernelParams(6));
	private static final ThreadLocal<KernelParams> ROPE_EXT_PARAMS = ThreadLocal.withInitial(() -> new KernelParams(9));
	private static final ThreadLocal<KernelParams> KV_PARAMS = ThreadLocal.withInitial(() -> new KernelParams(7));

	/** [type index][row tile index]; type index 0 = Q4_K, 1 = Q5_K, 2 = Q6_K. */
	private final MemorySegment[][] gemvRows;
	private final MemorySegment ropeRows;
	private final MemorySegment ropeSplitHalfRows;
	private final MemorySegment ropeExtRows;
	private final MemorySegment kvAppendRows;
	@SuppressWarnings("unused")
	private final Arena moduleArena; // keeps the module and function slots alive

	private BatchedDecodeKernels(MemorySegment[][] gemvRows, MemorySegment ropeRows, MemorySegment ropeSplitHalfRows,
			MemorySegment ropeExtRows, MemorySegment kvAppendRows, Arena moduleArena) {
		this.gemvRows = gemvRows;
		this.ropeRows = ropeRows;
		this.ropeSplitHalfRows = ropeSplitHalfRows;
		this.ropeExtRows = ropeExtRows;
		this.kvAppendRows = kvAppendRows;
		this.moduleArena = moduleArena;
	}

	/** Returns the loaded kernels, or {@code null} if the driver API or the PTX cannot load. */
	static BatchedDecodeKernels tryLoad() {
		BatchedDecodeKernels existing = INSTANCE.get();
		if (existing != null)
			return existing;
		synchronized (BatchedDecodeKernels.class) {
			existing = INSTANCE.get();
			if (existing != null)
				return existing;
			try {
				BatchedDecodeKernels k = loadNew();
				INSTANCE.set(k);
				log.info("Decode batch kernels loaded from " + RESOURCE);
				return k;
			} catch (Throwable t) {
				log.warning("Decode batch kernels unavailable: " + t.getMessage());
				return null;
			}
		}
	}

	private static BatchedDecodeKernels loadNew() throws IOException {
		if (!CudaDriverBindings.isAvailable())
			throw new IllegalStateException("CUDA driver API unavailable");
		CudaDriverBindings drv = CudaDriverBindings.instance();
		CudaBindings cuda = CudaBindings.instance();
		CudaBindings.check(CudaBindings.callInt(cuda.cudaSetDevice, 0), "cudaSetDevice");
		// Touch the primary context so cuCtxGetCurrent succeeds.
		MemorySegment probe = cuda.deviceMalloc(0, 4);
		cuda.deviceFree(probe);

		byte[] ptxBytes = readResource(RESOURCE);
		Arena arena = Arena.ofShared();
		MemorySegment ptx = arena.allocate(ptxBytes.length + 1L);
		MemorySegment.copy(MemorySegment.ofArray(ptxBytes), 0, ptx, 0, ptxBytes.length);
		ptx.set(JAVA_BYTE, ptxBytes.length, (byte) 0);

		MemorySegment moduleSlot = arena.allocate(ADDRESS);
		CudaDriverBindings.check(CudaDriverBindings.callInt(drv.cuModuleLoadData, moduleSlot, ptx),
				"cuModuleLoadData(batched_decode)");
		MemorySegment module = moduleSlot.get(ADDRESS, 0);
		String[] types = { "q4k", "q5k", "q6k" };
		MemorySegment[][] gemv = new MemorySegment[types.length][ROW_TILES.length];
		for (int t = 0; t < types.length; t++)
			for (int c = 0; c < ROW_TILES.length; c++)
				gemv[t][c] = function(drv, arena, module, types[t] + "_gemv_rows_" + ROW_TILES[c]);
		return new BatchedDecodeKernels(gemv, function(drv, arena, module, "rope_rows"),
				function(drv, arena, module, "rope_split_half_rows"),
				function(drv, arena, module, "rope_ext_split_half_rows"),
				function(drv, arena, module, "kv_append_rows"), arena);
	}

	private static MemorySegment function(CudaDriverBindings drv, Arena arena, MemorySegment module, String entry) {
		MemorySegment fnSlot = arena.allocate(ADDRESS);
		CudaDriverBindings.check(
				CudaDriverBindings.callInt(drv.cuModuleGetFunction, fnSlot, module, arena.allocateFrom(entry)),
				"cuModuleGetFunction(" + entry + ")");
		return fnSlot.get(ADDRESS, 0);
	}

	private static int typeIndex(int quantType) {
		return switch (quantType) {
			case QuantizationLayout.TYPE_Q4_K -> 0;
			case QuantizationLayout.TYPE_Q5_K -> 1;
			case QuantizationLayout.TYPE_Q6_K -> 2;
			default -> throw new IllegalArgumentException("No decode batch kernel for GGUF tensor type " + quantType);
		};
	}

	private static int tileIndex(int rows) {
		for (int i = 0; i < ROW_TILES.length; i++)
			if (rows <= ROW_TILES[i])
				return i;
		throw new IllegalArgumentException("no row tile covers " + rows + " rows");
	}

	/**
	 * {@code Y = A X} for {@code batch} activation rows already packed as Q8_1 in
	 * {@code dQ8} ({@link Q4KMmqKernel#q8Bytes} per row, contiguous): row {@code b} of
	 * the result is written at {@code dY + b * ldc} floats, columns
	 * {@code [0, A.rows())}, nothing else in {@code dY} is touched. A batch wider than
	 * {@link #MAX_ROWS_PER_LAUNCH} takes several launches.
	 */
	void gemvRows(DeviceQ4KMatrix A, MemorySegment dQ8, MemorySegment dY, int batch, int ldc, MemorySegment stream) {
		Objects.requireNonNull(A, "A");
		Objects.requireNonNull(dQ8, "dQ8");
		Objects.requireNonNull(dY, "dY");
		if (A.isClosed())
			throw new IllegalStateException("DeviceQ4KMatrix is closed");
		int rows = A.rows();
		int cols = A.cols();
		QuantizationLayout.require(A.quantType()).validateMatrix(rows, cols);
		if (batch <= 0)
			throw new IllegalArgumentException("batch must be positive: " + batch);
		if (ldc < rows)
			throw new IllegalArgumentException("ldc " + ldc + " < rows " + rows);
		int type = typeIndex(A.quantType());
		long q8RowBytes = Q4KMmqKernel.q8Bytes(cols);
		for (int done = 0; done < batch; done += MAX_ROWS_PER_LAUNCH) {
			int n = Math.min(batch - done, MAX_ROWS_PER_LAUNCH);
			GEMV_PARAMS.get().pointer(0, A.devicePointer()).pointer(1, dQ8.asSlice(done * q8RowBytes))
					.pointer(2, dY.asSlice((long) done * ldc * Float.BYTES)).i32(3, rows).i32(4, cols).i32(5, n)
					.i32(6, ldc).launch(gemvRows[type][tileIndex(n)], rows, GEMV_THREADS, stream,
							"cuLaunchKernel(kquant_gemv_rows)");
		}
	}

	/** Grid width for {@code items} items at {@link #BLOCK_THREADS} per block. */
	private static int grid(long items) {
		return Math.toIntExact((items + BLOCK_THREADS - 1) / BLOCK_THREADS);
	}

	/**
	 * Rotates each of {@code rows} rows of {@code nHeads} heads in place, row {@code r}
	 * at position {@code seqLens[r] - 1}, from the inverse-frequency table.
	 *
	 * @param splitHalf the split-half pairs of the Qwen2 family, otherwise adjacent pairs
	 */
	void ropeRows(MemorySegment x, MemorySegment invFreq, MemorySegment seqLens, int rows, int nHeads, int headDim,
			boolean splitHalf, MemorySegment stream) {
		long pairs = (long) rows * nHeads * (headDim >> 1);
		ROPE_PARAMS.get().pointer(0, x).pointer(1, invFreq).pointer(2, seqLens).i32(3, rows).i32(4, nHeads)
				.i32(5, headDim).launch(splitHalf ? ropeSplitHalfRows : ropeRows, grid(pairs), BLOCK_THREADS, stream,
						"cuLaunchKernel(rope_rows)");
	}

	/** As {@link #ropeRows}, with the Phi-3 family's extended rotation; {@code factors} may be {@code null}. */
	void ropeExtRows(MemorySegment x, MemorySegment factors, MemorySegment seqLens, int rows, int nHeads, int headDim,
			float thetaScale, float freqScale, float mscale, MemorySegment stream) {
		long pairs = (long) rows * nHeads * (headDim >> 1);
		ROPE_EXT_PARAMS.get().pointer(0, x).pointer(1, factors == null ? MemorySegment.NULL : factors)
				.pointer(2, seqLens).i32(3, rows).i32(4, nHeads).i32(5, headDim).f32(6, thetaScale).f32(7, freqScale)
				.f32(8, mscale).launch(ropeExtRows, grid(pairs), BLOCK_THREADS, stream, "cuLaunchKernel(rope_ext_rows)");
	}

	/**
	 * Casts each row's K and V to FP16 into that request's mirror at position
	 * {@code seqLens[r] - 1}. {@code kPtrs}, {@code vPtrs} and {@code seqLens} are the
	 * attention kernel's tables.
	 */
	void kvAppendRows(MemorySegment k, MemorySegment v, MemorySegment kPtrs, MemorySegment vPtrs, MemorySegment seqLens,
			int rows, int kvDim, MemorySegment stream) {
		KV_PARAMS.get().pointer(0, k).pointer(1, v).pointer(2, kPtrs).pointer(3, vPtrs).pointer(4, seqLens).i32(5, rows)
				.i32(6, kvDim).launch(kvAppendRows, grid(2L * rows * kvDim), BLOCK_THREADS, stream,
						"cuLaunchKernel(kv_append_rows)");
	}

	private static byte[] readResource(String path) throws IOException {
		try (InputStream in = BatchedDecodeKernels.class.getResourceAsStream(path)) {
			if (in == null)
				throw new IOException("missing classpath resource " + path);
			return in.readAllBytes();
		}
	}
}
