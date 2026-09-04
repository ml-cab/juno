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
import java.nio.charset.StandardCharsets;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;
import java.util.logging.Logger;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;

/**
 * Loads the classpath PTX module {@code q4k_gemv.ptx} and launches fused Q4_K GEMV.
 *
 * <p>One module per process; function handle is cached. Requires an active CUDA
 * primary context (any prior cudart allocation / {@code cudaSetDevice} is enough).
 */
final class Q4KMmqKernel {

	private static final Logger log = Logger.getLogger(Q4KMmqKernel.class.getName());
	private static final String RESOURCE = "/cab/ml/juno/node/q4k_gemv.ptx";
	private static final String ENTRY = "q4k_gemv";
	private static final int BLOCK_THREADS = 128;

	private static final AtomicReference<Q4KMmqKernel> INSTANCE = new AtomicReference<>();

	private final MemorySegment module;   // CUmodule (opaque pointer value)
	private final MemorySegment function; // CUfunction
	private final Arena moduleArena;      // keeps module/function slots alive

	private Q4KMmqKernel(MemorySegment module, MemorySegment function, Arena moduleArena) {
		this.module = module;
		this.function = function;
		this.moduleArena = moduleArena;
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
				log.info("Q4K MMQ kernel loaded from " + RESOURCE);
				return k;
			} catch (Throwable t) {
				log.warning("Q4K MMQ kernel unavailable: " + t.getMessage());
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

		MemorySegment name = arena.allocateFrom(ENTRY);
		MemorySegment fnSlot = arena.allocate(ADDRESS);
		CudaDriverBindings.check(
				CudaDriverBindings.callInt(drv.cuModuleGetFunction, fnSlot, module, name),
				"cuModuleGetFunction(" + ENTRY + ")");
		MemorySegment function = fnSlot.get(ADDRESS, 0);
		return new Q4KMmqKernel(module, function, arena);
	}

	private static byte[] readResource(String path) throws IOException {
		try (InputStream in = Q4KMmqKernel.class.getResourceAsStream(path)) {
			if (in == null)
				throw new IOException("missing classpath resource " + path);
			return in.readAllBytes();
		}
	}

	/**
	 * Launch {@code y = A_q4k * x} on {@code stream} (nullable = default stream).
	 * Device pointers must already hold the packed matrix / vectors.
	 */
	void launch(MemorySegment dA, MemorySegment dX, MemorySegment dY, int rows, int cols,
			MemorySegment stream) {
		Objects.requireNonNull(dA, "dA");
		Objects.requireNonNull(dX, "dX");
		Objects.requireNonNull(dY, "dY");
		QuantizationLayout.Q4_K.validateMatrix(rows, cols);

		CudaDriverBindings drv = CudaDriverBindings.instance();
		int grid = (rows + BLOCK_THREADS - 1) / BLOCK_THREADS;

		try (Arena arena = Arena.ofConfined()) {
			MemorySegment pA = arena.allocate(ADDRESS);
			MemorySegment pX = arena.allocate(ADDRESS);
			MemorySegment pY = arena.allocate(ADDRESS);
			MemorySegment pRows = arena.allocate(JAVA_INT);
			MemorySegment pCols = arena.allocate(JAVA_INT);
			pA.set(ADDRESS, 0, dA);
			pX.set(ADDRESS, 0, dX);
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
					"cuLaunchKernel(q4k_gemv)");
		}
	}
}
