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
import static java.lang.foreign.ValueLayout.JAVA_INT;

/**
 * Loads the classpath PTX module {@code q4k_gemv.ptx} and launches the fused
 * K-quant GEMV kernels ({@code quantize_q8_1}, {@code q4k_gemv}, {@code q5k_gemv},
 * {@code q6k_gemv}).
 *
 * <p>One module per process; function handles are cached. Requires an active CUDA
 * primary context (any prior cudart allocation / {@code cudaSetDevice} is enough).
 *
 * <p>Launch geometry matches {@code KQ_WARPS} in {@code q4k_gemv.cu}: 128 threads
 * (4 warps) per block, one output row per block. Activations are quantized to
 * Q8_1 once, then integer-dotted against packed weights.
 */
final class Q4KMmqKernel {

	private static final Logger log = Logger.getLogger(Q4KMmqKernel.class.getName());
	private static final String RESOURCE = "/cab/ml/juno/node/q4k_gemv.ptx";
	private static final String ENTRY_Q8 = "quantize_q8_1";
	private static final String ENTRY_Q4K = "q4k_gemv";
	private static final String ENTRY_Q5K = "q5k_gemv";
	private static final String ENTRY_Q6K = "q6k_gemv";
	private static final String ENTRY_Q4K_DEQUANT = "q4k_dequant_to_fp16";
	private static final String ENTRY_Q5K_DEQUANT = "q5k_dequant_to_fp16";
	private static final String ENTRY_Q6K_DEQUANT = "q6k_dequant_to_fp16";
	private static final int WARPS = 4;
	private static final int BLOCK_THREADS = WARPS * 32;
	static final int ROWS_PER_BLOCK = 1;
	static final int Q8_1_BLOCK_ELEMS = 32;
	static final int Q8_1_BLOCK_BYTES = 36;
	/** Elements per K-quant super-block; also the dequant kernel's block thread count. */
	static final int QK_K = 256;

	private static final AtomicReference<Q4KMmqKernel> INSTANCE = new AtomicReference<>();

	private final MemorySegment module;      // CUmodule (opaque pointer value)
	private final MemorySegment fnQ8;
	private final MemorySegment fnQ4K;       // CUfunction per quant type
	private final MemorySegment fnQ5K;
	private final MemorySegment fnQ6K;
	private final MemorySegment fnQ4Dequant; // dequant-to-FP16 CUfunction per quant type
	private final MemorySegment fnQ5Dequant;
	private final MemorySegment fnQ6Dequant;
	private final Arena moduleArena;         // keeps module/function slots alive

	private Q4KMmqKernel(MemorySegment module, MemorySegment fnQ8, MemorySegment fnQ4K,
			MemorySegment fnQ5K, MemorySegment fnQ6K, MemorySegment fnQ4Dequant,
			MemorySegment fnQ5Dequant, MemorySegment fnQ6Dequant, Arena moduleArena) {
		this.module = module;
		this.fnQ8 = fnQ8;
		this.fnQ4K = fnQ4K;
		this.fnQ5K = fnQ5K;
		this.fnQ6K = fnQ6K;
		this.fnQ4Dequant = fnQ4Dequant;
		this.fnQ5Dequant = fnQ5Dequant;
		this.fnQ6Dequant = fnQ6Dequant;
		this.moduleArena = moduleArena;
	}

	/** Device bytes for a Q8_1 packing of {@code cols} FP32 activations. */
	static long q8Bytes(int cols) {
		if ((cols & (Q8_1_BLOCK_ELEMS - 1)) != 0)
			throw new IllegalArgumentException("cols=" + cols + " is not a multiple of " + Q8_1_BLOCK_ELEMS);
		return (long) (cols / Q8_1_BLOCK_ELEMS) * Q8_1_BLOCK_BYTES;
	}

	static boolean isAvailable() {
		return CudaDriverBindings.isAvailable() && CudaAvailability.isAvailable();
	}

	/**
	 * Returns a loaded kernel, or {@code null} if the driver API / PTX cannot load.
	 */
	static Q4KMmqKernel tryLoad() {
		Q4KMmqKernel existing = INSTANCE.get();
		if (existing != null)
			return existing;
		synchronized (Q4KMmqKernel.class) {
			existing = INSTANCE.get();
			if (existing != null)
				return existing;
			try {
				Q4KMmqKernel k = loadNew();
				INSTANCE.set(k);
				log.info("K-quant MMQ kernels loaded from " + RESOURCE + " (Q4_K, Q5_K, Q6_K)");
				return k;
			} catch (Throwable t) {
				log.warning("K-quant MMQ kernel unavailable: " + t.getMessage());
				return null;
			}
		}
	}

	private static Q4KMmqKernel loadNew() throws IOException {
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
		CudaDriverBindings.check(
				CudaDriverBindings.callInt(drv.cuModuleLoadData, moduleSlot, ptx),
				"cuModuleLoadData");
		MemorySegment module = moduleSlot.get(ADDRESS, 0);

		MemorySegment fnQ8 = getFunction(drv, arena, module, ENTRY_Q8);
		MemorySegment fnQ4K = getFunction(drv, arena, module, ENTRY_Q4K);
		MemorySegment fnQ5K = getFunction(drv, arena, module, ENTRY_Q5K);
		MemorySegment fnQ6K = getFunction(drv, arena, module, ENTRY_Q6K);
		MemorySegment fnQ4Dequant = getFunction(drv, arena, module, ENTRY_Q4K_DEQUANT);
		MemorySegment fnQ5Dequant = getFunction(drv, arena, module, ENTRY_Q5K_DEQUANT);
		MemorySegment fnQ6Dequant = getFunction(drv, arena, module, ENTRY_Q6K_DEQUANT);
		return new Q4KMmqKernel(module, fnQ8, fnQ4K, fnQ5K, fnQ6K,
				fnQ4Dequant, fnQ5Dequant, fnQ6Dequant, arena);
	}

	private static MemorySegment getFunction(CudaDriverBindings drv, Arena arena, MemorySegment module,
			String entry) {
		MemorySegment name = arena.allocateFrom(entry);
		MemorySegment fnSlot = arena.allocate(ADDRESS);
		CudaDriverBindings.check(
				CudaDriverBindings.callInt(drv.cuModuleGetFunction, fnSlot, module, name),
				"cuModuleGetFunction(" + entry + ")");
		return fnSlot.get(ADDRESS, 0);
	}

	private static byte[] readResource(String path) throws IOException {
		try (InputStream in = Q4KMmqKernel.class.getResourceAsStream(path)) {
			if (in == null)
				throw new IOException("missing classpath resource " + path);
			return in.readAllBytes();
		}
	}

	private MemorySegment functionFor(int quantType) {
		return switch (quantType) {
			case QuantizationLayout.TYPE_Q4_K -> fnQ4K;
			case QuantizationLayout.TYPE_Q5_K -> fnQ5K;
			case QuantizationLayout.TYPE_Q6_K -> fnQ6K;
			default -> throw new IllegalArgumentException("No fused GEMV kernel for GGML type " + quantType);
		};
	}

	private MemorySegment dequantFunctionFor(int quantType) {
		return switch (quantType) {
			case QuantizationLayout.TYPE_Q4_K -> fnQ4Dequant;
			case QuantizationLayout.TYPE_Q5_K -> fnQ5Dequant;
			case QuantizationLayout.TYPE_Q6_K -> fnQ6Dequant;
			default -> throw new IllegalArgumentException("No dequant kernel for GGML type " + quantType);
		};
	}

	/**
	 * Quantize FP32 {@code dX[cols]} into Q8_1 {@code dQ8}, then launch
	 * {@code y = A * x} on {@code stream} (nullable = default stream).
	 */
	void launch(DeviceQ4KMatrix A, MemorySegment dX, MemorySegment dQ8, MemorySegment dY, MemorySegment stream) {
		Objects.requireNonNull(A, "A");
		quantizeX(dX, dQ8, A.cols(), stream);
		launchPacked(A.devicePointer(), dQ8, dY, A.rows(), A.cols(), A.quantType(), stream);
	}

	/**
	 * Quantize FP32 {@code dX[cols]} to Q8_1 in {@code dQ8}. {@code cols} must be
	 * a multiple of {@link #Q8_1_BLOCK_ELEMS}.
	 */
	void quantizeX(MemorySegment dX, MemorySegment dQ8, int cols, MemorySegment stream) {
		Objects.requireNonNull(dX, "dX");
		Objects.requireNonNull(dQ8, "dQ8");
		long expected = q8Bytes(cols);
		if (dQ8.byteSize() < expected)
			throw new IllegalArgumentException("dQ8 bytes " + dQ8.byteSize() + " < " + expected);

		int nblocks = cols / Q8_1_BLOCK_ELEMS;
		int grid = (nblocks + WARPS - 1) / WARPS;
		CudaDriverBindings drv = CudaDriverBindings.instance();
		try (Arena arena = Arena.ofConfined()) {
			MemorySegment pX = arena.allocate(ADDRESS);
			MemorySegment pY = arena.allocate(ADDRESS);
			MemorySegment pN = arena.allocate(JAVA_INT);
			pX.set(ADDRESS, 0, dX);
			pY.set(ADDRESS, 0, dQ8);
			pN.set(JAVA_INT, 0, cols);
			MemorySegment params = arena.allocate(ADDRESS, 3);
			params.setAtIndex(ADDRESS, 0, pX);
			params.setAtIndex(ADDRESS, 1, pY);
			params.setAtIndex(ADDRESS, 2, pN);
			MemorySegment streamOrNull = stream == null ? MemorySegment.NULL : stream;
			CudaDriverBindings.check(
					CudaDriverBindings.callInt(drv.cuLaunchKernel,
							fnQ8,
							grid, 1, 1,
							BLOCK_THREADS, 1, 1,
							0,
							streamOrNull,
							params,
							MemorySegment.NULL),
					"cuLaunchKernel(quantize_q8_1)");
		}
	}

	/**
	 * Launch {@code y = A_kquant * x_q8} on {@code stream} (nullable = default
	 * stream). {@code dQ8} must already hold the Q8_1 packing of {@code x}.
	 */
	void launchPacked(MemorySegment dA, MemorySegment dQ8, MemorySegment dY, int rows, int cols, int quantType,
			MemorySegment stream) {
		Objects.requireNonNull(dA, "dA");
		Objects.requireNonNull(dQ8, "dQ8");
		Objects.requireNonNull(dY, "dY");
		QuantizationLayout.require(quantType).validateMatrix(rows, cols);
		MemorySegment function = functionFor(quantType);

		CudaDriverBindings drv = CudaDriverBindings.instance();
		int grid = (rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;

		try (Arena arena = Arena.ofConfined()) {
			MemorySegment pA = arena.allocate(ADDRESS);
			MemorySegment pX = arena.allocate(ADDRESS);
			MemorySegment pY = arena.allocate(ADDRESS);
			MemorySegment pRows = arena.allocate(JAVA_INT);
			MemorySegment pCols = arena.allocate(JAVA_INT);
			pA.set(ADDRESS, 0, dA);
			pX.set(ADDRESS, 0, dQ8);
			pY.set(ADDRESS, 0, dY);
			pRows.set(JAVA_INT, 0, rows);
			pCols.set(JAVA_INT, 0, cols);

			MemorySegment params = arena.allocate(ADDRESS, 5);
			params.setAtIndex(ADDRESS, 0, pA);
			params.setAtIndex(ADDRESS, 1, pX);
			params.setAtIndex(ADDRESS, 2, pY);
			params.setAtIndex(ADDRESS, 3, pRows);
			params.setAtIndex(ADDRESS, 4, pCols);

			MemorySegment streamOrNull = stream == null ? MemorySegment.NULL : stream;
			CudaDriverBindings.check(
					CudaDriverBindings.callInt(drv.cuLaunchKernel,
							function,
							grid, 1, 1,
							BLOCK_THREADS, 1, 1,
							0,
							streamOrNull,
							params,
							MemorySegment.NULL),
					"cuLaunchKernel(kquant_gemv type=" + quantType + ")");
		}
	}

	/**
	 * Launches the elementwise dequant-to-FP16 kernel for {@code A}: writes a
	 * row-major FP16 buffer at {@code dOutFp16} (sized {@code rows * cols * 2}
	 * bytes), independent of any activation vector. Feeds the batched-prefill
	 * tiled GEMM path ({@code CudaFp16GemmOps}) for {@code DeviceQ4KMatrix}.
	 */
	void launchDequant(DeviceQ4KMatrix A, MemorySegment dOutFp16, MemorySegment stream) {
		Objects.requireNonNull(A, "A");
		Objects.requireNonNull(dOutFp16, "dOutFp16");
		int rows = A.rows(), cols = A.cols();
		int quantType = A.quantType();
		QuantizationLayout.require(quantType).validateMatrix(rows, cols);
		MemorySegment function = dequantFunctionFor(quantType);

		CudaDriverBindings drv = CudaDriverBindings.instance();
		int nb = cols / QK_K;

		try (Arena arena = Arena.ofConfined()) {
			MemorySegment pA = arena.allocate(ADDRESS);
			MemorySegment pOut = arena.allocate(ADDRESS);
			MemorySegment pRows = arena.allocate(JAVA_INT);
			MemorySegment pCols = arena.allocate(JAVA_INT);
			pA.set(ADDRESS, 0, A.devicePointer());
			pOut.set(ADDRESS, 0, dOutFp16);
			pRows.set(JAVA_INT, 0, rows);
			pCols.set(JAVA_INT, 0, cols);

			MemorySegment params = arena.allocate(ADDRESS, 4);
			params.setAtIndex(ADDRESS, 0, pA);
			params.setAtIndex(ADDRESS, 1, pOut);
			params.setAtIndex(ADDRESS, 2, pRows);
			params.setAtIndex(ADDRESS, 3, pCols);

			MemorySegment streamOrNull = stream == null ? MemorySegment.NULL : stream;
			CudaDriverBindings.check(
					CudaDriverBindings.callInt(drv.cuLaunchKernel,
							function,
							rows, nb, 1,
							QK_K, 1, 1,
							0,
							streamOrNull,
							params,
							MemorySegment.NULL),
					"cuLaunchKernel(kquant_dequant_to_fp16 type=" + quantType + ")");
		}
	}
}
