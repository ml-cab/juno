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
 * Tiled matrix multiply over still-packed Q4_K, Q5_K and Q6_K weights for batches
 * of activation rows: the prefill-width counterpart of the fused GEMV kernels in
 * {@link Q4KMmqKernel}. Loads the classpath PTX module {@code kquant_gemm.ptx}.
 *
 * <p>The activation batch is quantized to Q8_1 once (the GEMV path's
 * {@code quantize_q8_1}, over {@code batch * cols} contiguous values), then each
 * thread block multiplies a tile of 64 weight rows by a tile of activation rows,
 * one 256-element super-block at a time: packed weights are unpacked to int8 in
 * shared memory, integer-dotted with {@code dp4a}, and scaled into FP32
 * accumulators. No FP16 copy of the weights is ever made.
 *
 * <p>Three column-tile widths (16, 32 and 64 activation rows per block) are
 * compiled per type; {@link #columnTile} picks the narrowest that covers the
 * batch, up to 64, so a narrow continuous-schedule step does not pay for 64
 * columns of work.
 */
final class KQuantGemmKernel {

	private static final Logger log = Logger.getLogger(KQuantGemmKernel.class.getName());
	private static final String RESOURCE = "/cab/ml/juno/node/kquant_gemm.ptx";

	/** Threads per block: 8 warps. Matches {@code KG_THREADS} in {@code kquant_gemm.cu}. */
	static final int BLOCK_THREADS = 256;
	/** Weight rows per block. Matches {@code KG_TILE_ROWS}. */
	static final int TILE_ROWS = 64;
	private static final int[] COLUMN_TILES = { 16, 32, 64 };

	private static final AtomicReference<KQuantGemmKernel> INSTANCE = new AtomicReference<>();
	private static final ThreadLocal<KernelParams> PARAMS = ThreadLocal.withInitial(() -> new KernelParams(7));
	private static final ThreadLocal<KernelParams> QUANT_PARAMS = ThreadLocal.withInitial(() -> new KernelParams(3));
	/** Threads per block of {@code quantize_q8_1_half}: four warps, one 32-value block each. */
	private static final int QUANT_THREADS = 128;

	/** [type index][column tile index]; type index 0 = Q4_K, 1 = Q5_K, 2 = Q6_K. */
	private final MemorySegment[][] functions;
	private final MemorySegment quantizeHalf;
	@SuppressWarnings("unused")
	private final Arena moduleArena; // keeps the module and function slots alive

	private KQuantGemmKernel(MemorySegment[][] functions, MemorySegment quantizeHalf, Arena moduleArena) {
		this.functions = functions;
		this.quantizeHalf = quantizeHalf;
		this.moduleArena = moduleArena;
	}

	/** Device bytes of the Q8_1 packing of {@code batch} rows of {@code cols} activations. */
	static long q8Bytes(int batch, int cols) {
		if (batch <= 0)
			throw new IllegalArgumentException("batch must be positive: " + batch);
		return (long) batch * Q4KMmqKernel.q8Bytes(cols);
	}

	/** Activation rows per block for a batch of {@code batch} rows. */
	static int columnTile(int batch) {
		for (int tile : COLUMN_TILES)
			if (batch <= tile)
				return tile;
		return COLUMN_TILES[COLUMN_TILES.length - 1];
	}

	/** Returns the loaded kernels, or {@code null} if the driver API or the PTX cannot load. */
	static KQuantGemmKernel tryLoad() {
		KQuantGemmKernel existing = INSTANCE.get();
		if (existing != null)
			return existing;
		synchronized (KQuantGemmKernel.class) {
			existing = INSTANCE.get();
			if (existing != null)
				return existing;
			try {
				KQuantGemmKernel k = loadNew();
				INSTANCE.set(k);
				log.info("Tiled K-quant GEMM kernels loaded from " + RESOURCE + " (Q4_K, Q5_K, Q6_K)");
				return k;
			} catch (Throwable t) {
				log.warning("Tiled K-quant GEMM kernels unavailable: " + t.getMessage());
				return null;
			}
		}
	}

	private static KQuantGemmKernel loadNew() throws IOException {
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
				"cuModuleLoadData");
		MemorySegment module = moduleSlot.get(ADDRESS, 0);
		String[] types = { "q4k", "q5k", "q6k" };
		MemorySegment[][] functions = new MemorySegment[types.length][COLUMN_TILES.length];
		for (int t = 0; t < types.length; t++)
			for (int c = 0; c < COLUMN_TILES.length; c++)
				functions[t][c] = function(drv, arena, module, types[t] + "_gemm_" + COLUMN_TILES[c]);
		return new KQuantGemmKernel(functions, function(drv, arena, module, "quantize_q8_1_half"), arena);
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
			default -> throw new IllegalArgumentException("No tiled GEMM kernel for GGUF tensor type " + quantType);
		};
	}

	private static int tileIndex(int tile) {
		for (int i = 0; i < COLUMN_TILES.length; i++)
			if (COLUMN_TILES[i] == tile)
				return i;
		throw new IllegalArgumentException("no column tile " + tile);
	}

	/**
	 * {@code Y = A X} for {@code batch} FP32 activation rows {@code dX} (row-major,
	 * {@code A.cols()} floats each) on {@code stream}: quantizes them to Q8_1 into
	 * {@code dQ8} ({@link #q8Bytes} bytes), then runs the tiled kernel. Row {@code b}
	 * of the result is written at {@code dY + b * ldc} floats, columns
	 * {@code [0, A.rows())}; nothing else in {@code dY} is touched.
	 */
	void multiply(DeviceQ4KMatrix A, MemorySegment dX, MemorySegment dQ8, MemorySegment dY, int batch, int ldc,
			MemorySegment stream) {
		Objects.requireNonNull(A, "A");
		Q4KMmqKernel gemv = Q4KMmqKernel.tryLoad();
		if (gemv == null)
			throw new IllegalStateException("K-quant Q8_1 quantize kernel is not loaded");
		if (dQ8.byteSize() < q8Bytes(batch, A.cols()))
			throw new IllegalArgumentException("dQ8 bytes " + dQ8.byteSize() + " < " + q8Bytes(batch, A.cols()));
		gemv.quantizeX(dX, dQ8, Math.multiplyExact(batch, A.cols()), stream);
		launch(A, dQ8, dY, batch, ldc, stream);
	}

	/**
	 * As {@link #multiply}, for {@code batch} FP16 activation rows {@code dXh}: the
	 * prefill paths' GEMM input. The Q8_1 bytes are those {@link #multiply} makes
	 * from the same values widened to FP32.
	 */
	void multiplyHalf(DeviceQ4KMatrix A, MemorySegment dXh, MemorySegment dQ8, MemorySegment dY, int batch,
			int ldc, MemorySegment stream) {
		Objects.requireNonNull(A, "A");
		quantizeHalf(dXh, dQ8, batch, A.cols(), stream);
		launch(A, dQ8, dY, batch, ldc, stream);
	}

	/** Packs {@code batch} rows of {@code cols} FP16 values at {@code dXh} as Q8_1 into {@code dQ8}. */
	void quantizeHalf(MemorySegment dXh, MemorySegment dQ8, int batch, int cols, MemorySegment stream) {
		Objects.requireNonNull(dXh, "dXh");
		Objects.requireNonNull(dQ8, "dQ8");
		long needed = q8Bytes(batch, cols);
		if (dQ8.byteSize() < needed)
			throw new IllegalArgumentException("dQ8 bytes " + dQ8.byteSize() + " < " + needed);
		int n = Math.multiplyExact(batch, cols);
		int blocks = n / Q4KMmqKernel.Q8_1_BLOCK_ELEMS;
		int grid = (blocks + QUANT_THREADS / 32 - 1) / (QUANT_THREADS / 32);
		QUANT_PARAMS.get().pointer(0, dXh).pointer(1, dQ8).i32(2, n)
				.launch(quantizeHalf, grid, QUANT_THREADS, stream, "cuLaunchKernel(quantize_q8_1_half)");
	}

	/**
	 * Runs the tiled kernel on activations already packed as Q8_1 in {@code dQ8}
	 * ({@code batch} rows of {@code A.cols() / 32} blocks). Same output contract as
	 * {@link #multiply}.
	 */
	void launch(DeviceQ4KMatrix A, MemorySegment dQ8, MemorySegment dY, int batch, int ldc, MemorySegment stream) {
		Objects.requireNonNull(A, "A");
		Objects.requireNonNull(dQ8, "dQ8");
		Objects.requireNonNull(dY, "dY");
		if (A.isClosed())
			throw new IllegalStateException("DeviceQ4KMatrix is closed");
		int rows = A.rows(), cols = A.cols();
		QuantizationLayout.require(A.quantType()).validateMatrix(rows, cols);
		if (batch <= 0)
			throw new IllegalArgumentException("batch must be positive: " + batch);
		if (ldc < rows)
			throw new IllegalArgumentException("ldc " + ldc + " < rows " + rows);
		int tile = columnTile(batch);
		MemorySegment function = functions[typeIndex(A.quantType())][tileIndex(tile)];
		int gridX = (batch + tile - 1) / tile;
		int gridY = (rows + TILE_ROWS - 1) / TILE_ROWS;
		PARAMS.get().pointer(0, A.devicePointer()).pointer(1, dQ8).pointer(2, dY)
				.i32(3, rows).i32(4, cols).i32(5, batch).i32(6, ldc)
				.launch(function, gridX, gridY, BLOCK_THREADS, stream, "cuLaunchKernel(kquant_gemm)");
	}

	private static byte[] readResource(String path) throws IOException {
		try (InputStream in = KQuantGemmKernel.class.getResourceAsStream(path)) {
			if (in == null)
				throw new IOException("missing classpath resource " + path);
			return in.readAllBytes();
		}
	}
}
