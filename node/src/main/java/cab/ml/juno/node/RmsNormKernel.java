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
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;

/**
 * Loads the classpath PTX module {@code rms_norm.ptx} and launches the
 * GPU-resident RMS-normalisation kernel ({@code rms_norm}).
 *
 * <p>One module per process; the function handle is cached. Requires an
 * active CUDA primary context — same loading convention as
 * {@link Q4KMmqKernel} / {@link GqaAttentionKernel}.
 *
 * <p>Launch geometry: one block of {@code RMSNORM_THREADS} (128) threads per
 * batch row, grid size {@code B}. See {@code rms_norm.cu} for the kernel
 * itself.
 */
final class RmsNormKernel {

	private static final Logger log = Logger.getLogger(RmsNormKernel.class.getName());
	private static final String RESOURCE = "/cab/ml/juno/node/rms_norm.ptx";
	private static final String ENTRY = "rms_norm";
	private static final int RMSNORM_THREADS = 128;

	private static final AtomicReference<RmsNormKernel> INSTANCE = new AtomicReference<>();

	private final MemorySegment module; // CUmodule (opaque pointer value)
	private final MemorySegment fn;     // CUfunction
	private final Arena moduleArena;    // keeps module/function slots alive

	private RmsNormKernel(MemorySegment module, MemorySegment fn, Arena moduleArena) {
		this.module = module;
		this.fn = fn;
		this.moduleArena = moduleArena;
	}

	static boolean isAvailable() {
		return CudaDriverBindings.isAvailable() && CudaAvailability.isAvailable();
	}

	/** Returns a loaded kernel, or {@code null} if the driver API / PTX cannot load. */
	static RmsNormKernel tryLoad() {
		RmsNormKernel existing = INSTANCE.get();
		if (existing != null)
			return existing;
		synchronized (RmsNormKernel.class) {
			existing = INSTANCE.get();
			if (existing != null)
				return existing;
			try {
				RmsNormKernel k = loadNew();
				INSTANCE.set(k);
				log.info("GPU-resident RMS-norm kernel loaded from " + RESOURCE);
				return k;
			} catch (Throwable t) {
				log.warning("GPU-resident RMS-norm kernel unavailable: " + t.getMessage());
				return null;
			}
		}
	}

	private static RmsNormKernel loadNew() throws IOException {
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
		MemorySegment fn = fnSlot.get(ADDRESS, 0);

		return new RmsNormKernel(module, fn, arena);
	}

	private static byte[] readResource(String path) throws IOException {
		try (InputStream in = RmsNormKernel.class.getResourceAsStream(path)) {
			if (in == null)
				throw new IOException("missing classpath resource " + path);
			return in.readAllBytes();
		}
	}

	/**
	 * Launches {@code rms_norm} over {@code batch} rows (grid = {@code batch}).
	 * All device pointer arguments must already be resident; {@code stream}
	 * may be {@code null} for the default stream.
	 */
	void launch(MemorySegment xBatch, MemorySegment weight, MemorySegment outBatch,
			int batch, int dim, float eps, MemorySegment stream) {
		Objects.requireNonNull(xBatch, "xBatch");
		Objects.requireNonNull(weight, "weight");
		Objects.requireNonNull(outBatch, "outBatch");

		CudaDriverBindings drv = CudaDriverBindings.instance();

		try (Arena arena = Arena.ofConfined()) {
			MemorySegment pX = arena.allocate(ADDRESS);
			MemorySegment pW = arena.allocate(ADDRESS);
			MemorySegment pOut = arena.allocate(ADDRESS);
			MemorySegment pDim = arena.allocate(JAVA_INT);
			MemorySegment pEps = arena.allocate(JAVA_FLOAT);

			pX.set(ADDRESS, 0, xBatch);
			pW.set(ADDRESS, 0, weight);
			pOut.set(ADDRESS, 0, outBatch);
			pDim.set(JAVA_INT, 0, dim);
			pEps.set(JAVA_FLOAT, 0, eps);

			MemorySegment params = arena.allocate(ADDRESS, 5);
			params.setAtIndex(ADDRESS, 0, pX);
			params.setAtIndex(ADDRESS, 1, pW);
			params.setAtIndex(ADDRESS, 2, pOut);
			params.setAtIndex(ADDRESS, 3, pDim);
			params.setAtIndex(ADDRESS, 4, pEps);

			MemorySegment streamOrNull = stream == null ? MemorySegment.NULL : stream;
			CudaDriverBindings.check(
					CudaDriverBindings.callInt(drv.cuLaunchKernel,
							fn,
							batch, 1, 1,
							RMSNORM_THREADS, 1, 1,
							0,
							streamOrNull,
							params,
							MemorySegment.NULL),
					"cuLaunchKernel(rms_norm)");
		}
	}
}
