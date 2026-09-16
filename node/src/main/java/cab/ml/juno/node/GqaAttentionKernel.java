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
 * Loads the classpath PTX module {@code gqa_attention.ptx} and launches the
 * GPU-resident grouped-query attention kernel ({@code gqa_attention}).
 *
 * <p>One module per process; the function handle is cached. Requires an
 * active CUDA primary context (any prior cudart allocation / {@code
 * cudaSetDevice} is enough) — same loading convention as {@link Q4KMmqKernel}.
 *
 * <p>Launch geometry: one block of {@code GQA_THREADS} (128) threads per
 * {@code (b, h)} pair, grid size {@code B * numHeads}. See {@code
 * gqa_attention.cu} for the kernel itself.
 */
final class GqaAttentionKernel {

	private static final Logger log = Logger.getLogger(GqaAttentionKernel.class.getName());
	private static final String RESOURCE = "/cab/ml/juno/node/gqa_attention.ptx";
	private static final String ENTRY = "gqa_attention";
	private static final int GQA_THREADS = 128;

	private static final AtomicReference<GqaAttentionKernel> INSTANCE = new AtomicReference<>();

	private final MemorySegment module; // CUmodule (opaque pointer value)
	private final MemorySegment fn;     // CUfunction
	private final Arena moduleArena;    // keeps module/function slots alive

	private GqaAttentionKernel(MemorySegment module, MemorySegment fn, Arena moduleArena) {
		this.module = module;
		this.fn = fn;
		this.moduleArena = moduleArena;
	}

	static boolean isAvailable() {
		return CudaDriverBindings.isAvailable() && CudaAvailability.isAvailable();
	}

	/** Returns a loaded kernel, or {@code null} if the driver API / PTX cannot load. */
	static GqaAttentionKernel tryLoad() {
		GqaAttentionKernel existing = INSTANCE.get();
		if (existing != null)
			return existing;
		synchronized (GqaAttentionKernel.class) {
			existing = INSTANCE.get();
			if (existing != null)
				return existing;
			try {
				GqaAttentionKernel k = loadNew();
				INSTANCE.set(k);
				log.info("GPU-resident attention kernel loaded from " + RESOURCE);
				return k;
			} catch (Throwable t) {
				log.warning("GPU-resident attention kernel unavailable: " + t.getMessage());
				return null;
			}
		}
	}

	private static GqaAttentionKernel loadNew() throws IOException {
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

		return new GqaAttentionKernel(module, fn, arena);
	}

	private static byte[] readResource(String path) throws IOException {
		try (InputStream in = GqaAttentionKernel.class.getResourceAsStream(path)) {
			if (in == null)
				throw new IOException("missing classpath resource " + path);
			return in.readAllBytes();
		}
	}

	/**
	 * Launches {@code gqa_attention} for {@code B} (batch-row, head) pairs
	 * (grid = {@code B * numHeads}). All device pointer arguments must already
	 * be resident; {@code stream} may be {@code null} for the default stream.
	 */
	void launch(MemorySegment qBatch, MemorySegment kPtrs, MemorySegment vPtrs, MemorySegment seqLens,
			MemorySegment scoresScratch, MemorySegment outBatch,
			int batch, int numHeads, int gqaRatio, int headDim, int kvDim, int rowStride,
			MemorySegment stream) {
		Objects.requireNonNull(qBatch, "qBatch");
		Objects.requireNonNull(kPtrs, "kPtrs");
		Objects.requireNonNull(vPtrs, "vPtrs");
		Objects.requireNonNull(seqLens, "seqLens");
		Objects.requireNonNull(scoresScratch, "scoresScratch");
		Objects.requireNonNull(outBatch, "outBatch");

		CudaDriverBindings drv = CudaDriverBindings.instance();
		int grid = batch * numHeads;

		try (Arena arena = Arena.ofConfined()) {
			MemorySegment pQ = arena.allocate(ADDRESS);
			MemorySegment pK = arena.allocate(ADDRESS);
			MemorySegment pV = arena.allocate(ADDRESS);
			MemorySegment pSeqLens = arena.allocate(ADDRESS);
			MemorySegment pScores = arena.allocate(ADDRESS);
			MemorySegment pOut = arena.allocate(ADDRESS);
			MemorySegment pNumHeads = arena.allocate(JAVA_INT);
			MemorySegment pGqaRatio = arena.allocate(JAVA_INT);
			MemorySegment pHeadDim = arena.allocate(JAVA_INT);
			MemorySegment pKvDim = arena.allocate(JAVA_INT);
			MemorySegment pRowStride = arena.allocate(JAVA_INT);

			pQ.set(ADDRESS, 0, qBatch);
			pK.set(ADDRESS, 0, kPtrs);
			pV.set(ADDRESS, 0, vPtrs);
			pSeqLens.set(ADDRESS, 0, seqLens);
			pScores.set(ADDRESS, 0, scoresScratch);
			pOut.set(ADDRESS, 0, outBatch);
			pNumHeads.set(JAVA_INT, 0, numHeads);
			pGqaRatio.set(JAVA_INT, 0, gqaRatio);
			pHeadDim.set(JAVA_INT, 0, headDim);
			pKvDim.set(JAVA_INT, 0, kvDim);
			pRowStride.set(JAVA_INT, 0, rowStride);

			MemorySegment params = arena.allocate(ADDRESS, 11);
			params.setAtIndex(ADDRESS, 0, pQ);
			params.setAtIndex(ADDRESS, 1, pK);
			params.setAtIndex(ADDRESS, 2, pV);
			params.setAtIndex(ADDRESS, 3, pSeqLens);
			params.setAtIndex(ADDRESS, 4, pScores);
			params.setAtIndex(ADDRESS, 5, pOut);
			params.setAtIndex(ADDRESS, 6, pNumHeads);
			params.setAtIndex(ADDRESS, 7, pGqaRatio);
			params.setAtIndex(ADDRESS, 8, pHeadDim);
			params.setAtIndex(ADDRESS, 9, pKvDim);
			params.setAtIndex(ADDRESS, 10, pRowStride);

			MemorySegment streamOrNull = stream == null ? MemorySegment.NULL : stream;
			CudaDriverBindings.check(
					CudaDriverBindings.callInt(drv.cuLaunchKernel,
							fn,
							grid, 1, 1,
							GQA_THREADS, 1, 1,
							0,
							streamOrNull,
							params,
							MemorySegment.NULL),
					"cuLaunchKernel(gqa_attention)");
		}
	}
}
