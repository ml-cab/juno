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
 * GPU-resident grouped-query attention kernel: tiled, with an online softmax, so
 * it keeps no score row and needs no device scratch that grows with the context.
 *
 * <p>One module per process; the function handles are cached. Requires an
 * active CUDA primary context (any prior cudart allocation / {@code
 * cudaSetDevice} is enough) - same loading convention as {@link Q4KMmqKernel}.
 *
 * <p>Launch geometry: blocks of {@code GQA_THREADS} (128) threads, grid
 * {@code ceil(B / rowsPerBlock) x numHeads}. A block takes {@code rowsPerBlock}
 * query rows of one head; see {@link #rowsPerBlock} and {@code gqa_attention.cu}.
 * Three entries bound the head width held in registers ({@code gqa_attention_d64},
 * {@code gqa_attention_d128}, {@code gqa_attention_d256}); {@link #supportsHeadDim} says which widths run.
 */
final class GqaAttentionKernel {

	private static final Logger log = Logger.getLogger(GqaAttentionKernel.class.getName());
	private static final String RESOURCE = "/cab/ml/juno/node/gqa_attention.ptx";
	private static final String ENTRY_D64 = "gqa_attention_d64";
	private static final String ENTRY_D128 = "gqa_attention_d128";
	private static final String ENTRY_D256 = "gqa_attention_d256";
	private static final int GQA_THREADS = 128;
	/** Query rows a block can take: the kernel's 32 slots of 4 lanes. */
	static final int MAX_ROWS_PER_BLOCK = 32;
	/** Widest head the kernel holds in registers. */
	static final int MAX_HEAD_DIM = 256;

	private static final AtomicReference<GqaAttentionKernel> INSTANCE = new AtomicReference<>();

	private final MemorySegment module; // CUmodule (opaque pointer value)
	private final MemorySegment fnD64;  // CUfunction, headDim <= 64
	private final MemorySegment fnD128; // CUfunction, headDim <= 128
	private final MemorySegment fnD256; // CUfunction, headDim <= 256
	private final Arena moduleArena;    // keeps module/function slots alive

	private GqaAttentionKernel(MemorySegment module, MemorySegment fnD64, MemorySegment fnD128, MemorySegment fnD256,
			Arena moduleArena) {
		this.module = module;
		this.fnD64 = fnD64;
		this.fnD128 = fnD128;
		this.fnD256 = fnD256;
		this.moduleArena = moduleArena;
	}

	/** Whether the kernel runs heads of {@code headDim} values: a multiple of 4, at most {@link #MAX_HEAD_DIM}. */
	static boolean supportsHeadDim(int headDim) {
		return headDim > 0 && headDim <= MAX_HEAD_DIM && (headDim & 3) == 0;
	}

	/**
	 * Query rows per block: a tile of consecutive rows reads each staged key once,
	 * which needs every row of it to attend over the same cache. So a window of
	 * {@code batch} rows over one cache takes the smallest power of two covering it,
	 * up to {@link #MAX_ROWS_PER_BLOCK}; rows over different caches take one row a
	 * block, which spreads that row's keys over all of the block's slots.
	 */
	static int rowsPerBlock(boolean oneCache, int batch) {
		if (!oneCache || batch <= 1)
			return 1;
		return Math.min(MAX_ROWS_PER_BLOCK, Integer.highestOneBit(batch - 1) << 1);
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

		return new GqaAttentionKernel(module, function(drv, arena, module, ENTRY_D64),
				function(drv, arena, module, ENTRY_D128),
				function(drv, arena, module, ENTRY_D256), arena);
	}

	private static MemorySegment function(CudaDriverBindings drv, Arena arena, MemorySegment module, String entry) {
		MemorySegment fnSlot = arena.allocate(ADDRESS);
		CudaDriverBindings.check(
				CudaDriverBindings.callInt(drv.cuModuleGetFunction, fnSlot, module, arena.allocateFrom(entry)),
				"cuModuleGetFunction(" + entry + ")");
		return fnSlot.get(ADDRESS, 0);
	}

	private static byte[] readResource(String path) throws IOException {
		try (InputStream in = GqaAttentionKernel.class.getResourceAsStream(path)) {
			if (in == null)
				throw new IOException("missing classpath resource " + path);
			return in.readAllBytes();
		}
	}

	/**
	 * Launches the kernel for {@code batch} query rows of {@code numHeads} heads.
	 * Row {@code b} attends over keys {@code [max(0, seqLens[b] - window), seqLens[b])}
	 * of the cache {@code kPtrs[b]}/{@code vPtrs[b]}; {@code window} 0 means no
	 * window. With {@code rowsPerBlock > 1} every row of a block reads the first
	 * row's cache, so pass {@link #rowsPerBlock}. All device pointer arguments must
	 * already be resident; {@code stream} may be {@code null} for the default stream.
	 */
	void launch(MemorySegment qBatch, MemorySegment kPtrs, MemorySegment vPtrs, MemorySegment seqLens,
			MemorySegment outBatch, int batch, int numHeads, int gqaRatio, int headDim, int kvDim,
			int rowsPerBlock, int window, MemorySegment stream) {
		Objects.requireNonNull(qBatch, "qBatch");
		Objects.requireNonNull(kPtrs, "kPtrs");
		Objects.requireNonNull(vPtrs, "vPtrs");
		Objects.requireNonNull(seqLens, "seqLens");
		Objects.requireNonNull(outBatch, "outBatch");
		if (!supportsHeadDim(headDim))
			throw new IllegalArgumentException("attention kernel runs head widths that are multiples of 4 up to " + MAX_HEAD_DIM
					+ " (got " + headDim + ")");
		if (rowsPerBlock < 1 || rowsPerBlock > MAX_ROWS_PER_BLOCK || Integer.bitCount(rowsPerBlock) != 1)
			throw new IllegalArgumentException("rowsPerBlock must be a power of two up to " + MAX_ROWS_PER_BLOCK
					+ " (got " + rowsPerBlock + ")");
		if (window < 0)
			throw new IllegalArgumentException("window must be >= 0 (got " + window + ")");

		CudaDriverBindings drv = CudaDriverBindings.instance();
		MemorySegment fn = headDim <= 64 ? fnD64 : headDim <= 128 ? fnD128 : fnD256;
		int gridX = (batch + rowsPerBlock - 1) / rowsPerBlock;

		try (Arena arena = Arena.ofConfined()) {
			MemorySegment params = arena.allocate(ADDRESS, 12);
			params.setAtIndex(ADDRESS, 0, pointerArg(arena, qBatch));
			params.setAtIndex(ADDRESS, 1, pointerArg(arena, kPtrs));
			params.setAtIndex(ADDRESS, 2, pointerArg(arena, vPtrs));
			params.setAtIndex(ADDRESS, 3, pointerArg(arena, seqLens));
			params.setAtIndex(ADDRESS, 4, pointerArg(arena, outBatch));
			params.setAtIndex(ADDRESS, 5, intArg(arena, batch));
			params.setAtIndex(ADDRESS, 6, intArg(arena, numHeads));
			params.setAtIndex(ADDRESS, 7, intArg(arena, gqaRatio));
			params.setAtIndex(ADDRESS, 8, intArg(arena, headDim));
			params.setAtIndex(ADDRESS, 9, intArg(arena, kvDim));
			params.setAtIndex(ADDRESS, 10, intArg(arena, rowsPerBlock));
			params.setAtIndex(ADDRESS, 11, intArg(arena, window));

			MemorySegment streamOrNull = stream == null ? MemorySegment.NULL : stream;
			CudaDriverBindings.check(
					CudaDriverBindings.callInt(drv.cuLaunchKernel,
							fn,
							gridX, numHeads, 1,
							GQA_THREADS, 1, 1,
							0,
							streamOrNull,
							params,
							MemorySegment.NULL),
					"cuLaunchKernel(gqa_attention)");
		}
	}

	private static MemorySegment pointerArg(Arena arena, MemorySegment value) {
		MemorySegment slot = arena.allocate(ADDRESS);
		slot.set(ADDRESS, 0, value);
		return slot;
	}

	private static MemorySegment intArg(Arena arena, int value) {
		MemorySegment slot = arena.allocate(JAVA_INT);
		slot.set(JAVA_INT, 0, value);
		return slot;
	}
}
