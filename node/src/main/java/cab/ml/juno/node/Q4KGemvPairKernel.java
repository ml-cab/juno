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
 * Loads the classpath PTX module {@code q4k_gemv_pair.ptx} and launches {@code q4k_gemv_pair}: the FFN gate and up
 * projections of one decode row in one launch, two packed Q4_K matrices of the same shape against one Q8_1 input.
 *
 * <p>Each output row is the one {@code q4k_gemv} ({@link Q4KMmqKernel}) computes, bit for bit; a lane reads its slice
 * of the input once for both matrices. Rows of at most {@link #SPLIT_MAX_BLOCKS} super-blocks run two row indices a
 * block (the warps a single row leaves idle take the second), wider rows one. See {@code q4k_gemv_pair.cu}.
 *
 * <p>One module per process; the launch parameter block is reused per thread, so a launch allocates nothing.
 */
final class Q4KGemvPairKernel {

	private static final Logger log = Logger.getLogger(Q4KGemvPairKernel.class.getName());
	private static final String RESOURCE = "/cab/ml/juno/node/q4k_gemv_pair.ptx";
	private static final String ENTRY = "q4k_gemv_pair";
	private static final int BLOCK_THREADS = 128;
	/** Super-blocks a row may hold for a block to run two row indices; {@code PAIR_SPLIT_MAX_BLOCKS} in the kernel. */
	static final int SPLIT_MAX_BLOCKS = 8;

	private static final AtomicReference<Q4KGemvPairKernel> INSTANCE = new AtomicReference<>();
	private static final ThreadLocal<KernelParams> PARAMS = ThreadLocal.withInitial(() -> new KernelParams(7));

	private final MemorySegment fn;
	private final Arena moduleArena; // keeps the module and function slots alive

	private Q4KGemvPairKernel(MemorySegment fn, Arena moduleArena) {
		this.fn = fn;
		this.moduleArena = moduleArena;
	}

	/** Returns the loaded kernel, or {@code null} if the driver API or the PTX cannot load. */
	static Q4KGemvPairKernel tryLoad() {
		Q4KGemvPairKernel existing = INSTANCE.get();
		if (existing != null)
			return existing;
		synchronized (Q4KGemvPairKernel.class) {
			existing = INSTANCE.get();
			if (existing != null)
				return existing;
			try {
				Q4KGemvPairKernel k = loadNew();
				INSTANCE.set(k);
				log.fine("Paired Q4_K GEMV kernel loaded from " + RESOURCE);
				return k;
			} catch (Throwable t) {
				log.warning("Paired Q4_K GEMV kernel unavailable (gate and up run as two launches): " + t.getMessage());
				return null;
			}
		}
	}

	/** Whether {@code gate} and {@code up} can share one launch: both packed Q4_K, of the same shape. */
	static boolean pairs(DeviceQ4KMatrix gate, DeviceQ4KMatrix up) {
		return gate != null && up != null && gate.quantType() == QuantizationLayout.TYPE_Q4_K
				&& up.quantType() == QuantizationLayout.TYPE_Q4_K && gate.rows() == up.rows()
				&& gate.cols() == up.cols();
	}

	/** Blocks of a launch over {@code rows} rows of {@code cols} columns. */
	static int grid(int rows, int cols) {
		return cols / Q4KMmqKernel.QK_K <= SPLIT_MAX_BLOCKS ? (rows + 1) / 2 : rows;
	}

	/**
	 * Launches {@code yGate = gate * x} and {@code yUp = up * x} on {@code stream} (nullable: the default stream).
	 * {@code dQ8} holds the Q8_1 packing of {@code x}; both matrices are packed Q4_K, {@code rows x cols}.
	 */
	void launch(MemorySegment dGate, MemorySegment dUp, MemorySegment dQ8, MemorySegment dYGate, MemorySegment dYUp,
			int rows, int cols, MemorySegment stream) {
		Objects.requireNonNull(dGate, "dGate");
		Objects.requireNonNull(dUp, "dUp");
		Objects.requireNonNull(dQ8, "dQ8");
		Objects.requireNonNull(dYGate, "dYGate");
		Objects.requireNonNull(dYUp, "dYUp");
		QuantizationLayout.require(QuantizationLayout.TYPE_Q4_K).validateMatrix(rows, cols);
		PARAMS.get().pointer(0, dGate).pointer(1, dUp).pointer(2, dQ8).pointer(3, dYGate).pointer(4, dYUp)
				.i32(5, rows).i32(6, cols).launch(fn, grid(rows, cols), BLOCK_THREADS, stream, "cuLaunchKernel(" + ENTRY + ")");
	}

	private static Q4KGemvPairKernel loadNew() throws IOException {
		if (!CudaDriverBindings.isAvailable())
			throw new IllegalStateException("CUDA driver API unavailable");
		CudaDriverBindings drv = CudaDriverBindings.instance();
		CudaBindings cuda = CudaBindings.instance();
		CudaBindings.check(CudaBindings.callInt(cuda.cudaSetDevice, 0), "cudaSetDevice");
		MemorySegment probe = cuda.deviceMalloc(0, 4); // touches the primary context
		cuda.deviceFree(probe);

		byte[] ptxBytes;
		try (InputStream in = Q4KGemvPairKernel.class.getResourceAsStream(RESOURCE)) {
			if (in == null)
				throw new IOException("missing classpath resource " + RESOURCE);
			ptxBytes = in.readAllBytes();
		}
		Arena arena = Arena.ofShared();
		MemorySegment ptx = arena.allocate(ptxBytes.length + 1L);
		MemorySegment.copy(MemorySegment.ofArray(ptxBytes), 0, ptx, 0, ptxBytes.length);
		ptx.set(JAVA_BYTE, ptxBytes.length, (byte) 0);
		MemorySegment moduleSlot = arena.allocate(ADDRESS);
		CudaDriverBindings.check(CudaDriverBindings.callInt(drv.cuModuleLoadData, moduleSlot, ptx), "cuModuleLoadData");
		MemorySegment module = moduleSlot.get(ADDRESS, 0);
		MemorySegment fnSlot = arena.allocate(ADDRESS);
		CudaDriverBindings.check(
				CudaDriverBindings.callInt(drv.cuModuleGetFunction, fnSlot, module, arena.allocateFrom(ENTRY)),
				"cuModuleGetFunction(" + ENTRY + ")");
		return new Q4KGemvPairKernel(fnSlot.get(ADDRESS, 0), arena);
	}
}
