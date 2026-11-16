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

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;

import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;
import java.util.logging.Logger;

/**
 * Host-side launcher for {@code gqa_attention_kv.ptx}: {@link GqaAttentionKernel}'s attention over a device KV whose
 * keys and values are stored in any {@link DeviceKvFormat}, dequantized in registers inside the kernel. Same
 * geometry, arguments and order of accumulation as the FP16 kernel, plus the two formats; decode rows at head widths up
 * to 128 run a wider block instead ({@code gqa_decode_kv_d64/_d128}: 64 slots, the same per-slot loop and merge), since
 * a decode launch has one block per head and row and few heads leave a 128-thread block short of warps. A separate module, loaded
 * the first time a handler selects a quantized device KV ({@link DeviceKvFormats#select}), so an FP16 launch loads
 * and compiles exactly what it did. {@link GqaAttentionKernel#launch} routes here for quantized formats.
 */
final class GqaAttentionKvKernel {

	private static final Logger log = Logger.getLogger(GqaAttentionKvKernel.class.getName());
	private static final String RESOURCE = "/cab/ml/juno/node/gqa_attention_kv.ptx";
	private static final int GQA_THREADS = 128;
	/** Threads of a wide decode block: {@code GQD_THREADS}, 64 slots of 4 lanes. */
	private static final int DECODE_THREADS = 256;

	private static final AtomicReference<GqaAttentionKvKernel> INSTANCE = new AtomicReference<>();
	private static volatile boolean failed;

	private final MemorySegment fnD64;
	private final MemorySegment fnD128;
	private final MemorySegment fnD256;
	private final MemorySegment fnDecodeD64;
	private final MemorySegment fnDecodeD128;
	@SuppressWarnings("unused")
	private final Arena moduleArena; // keeps the module and function slots alive

	private GqaAttentionKvKernel(MemorySegment fnD64, MemorySegment fnD128, MemorySegment fnD256,
			MemorySegment fnDecodeD64, MemorySegment fnDecodeD128, Arena moduleArena) {
		this.fnD64 = fnD64;
		this.fnD128 = fnD128;
		this.fnD256 = fnD256;
		this.fnDecodeD64 = fnDecodeD64;
		this.fnDecodeD128 = fnDecodeD128;
		this.moduleArena = moduleArena;
	}

	/** Returns the loaded kernel, or {@code null} if the driver API or the PTX cannot load (tried once). */
	static GqaAttentionKvKernel tryLoad() {
		GqaAttentionKvKernel existing = INSTANCE.get();
		if (existing != null || failed)
			return existing;
		synchronized (GqaAttentionKvKernel.class) {
			existing = INSTANCE.get();
			if (existing != null || failed)
				return existing;
			try {
				GqaAttentionKvKernel k = loadNew();
				INSTANCE.set(k);
				log.info("GPU attention kernel for quantized device KV loaded from " + RESOURCE);
				return k;
			} catch (Throwable t) {
				failed = true;
				log.warning("GPU attention kernel for quantized device KV unavailable: " + t.getMessage());
				return null;
			}
		}
	}

	private static GqaAttentionKvKernel loadNew() throws IOException {
		if (!CudaDriverBindings.isAvailable())
			throw new IllegalStateException("CUDA driver API unavailable");
		CudaDriverBindings drv = CudaDriverBindings.instance();
		CudaBindings cuda = CudaBindings.instance();
		CudaBindings.check(CudaBindings.callInt(cuda.cudaSetDevice, 0), "cudaSetDevice");
		// Touch the primary context so cuCtxGetCurrent succeeds.
		MemorySegment probe = cuda.deviceMalloc(0, 4);
		cuda.deviceFree(probe);

		byte[] ptxBytes;
		try (InputStream in = GqaAttentionKvKernel.class.getResourceAsStream(RESOURCE)) {
			if (in == null)
				throw new IOException("missing classpath resource " + RESOURCE);
			ptxBytes = in.readAllBytes();
		}
		Arena arena = Arena.ofShared();
		MemorySegment ptx = arena.allocate(ptxBytes.length + 1L);
		MemorySegment.copy(MemorySegment.ofArray(ptxBytes), 0, ptx, 0, ptxBytes.length);
		ptx.set(JAVA_BYTE, ptxBytes.length, (byte) 0);
		MemorySegment moduleSlot = arena.allocate(ADDRESS);
		CudaDriverBindings.check(CudaDriverBindings.callInt(drv.cuModuleLoadData, moduleSlot, ptx),
				"cuModuleLoadData(gqa_attention_kv)");
		MemorySegment module = moduleSlot.get(ADDRESS, 0);
		return new GqaAttentionKvKernel(function(drv, arena, module, "gqa_attention_kv_d64"),
				function(drv, arena, module, "gqa_attention_kv_d128"),
				function(drv, arena, module, "gqa_attention_kv_d256"), function(drv, arena, module, "gqa_decode_kv_d64"),
				function(drv, arena, module, "gqa_decode_kv_d128"), arena);
	}

	private static MemorySegment function(CudaDriverBindings drv, Arena arena, MemorySegment module, String entry) {
		MemorySegment fnSlot = arena.allocate(ADDRESS);
		CudaDriverBindings.check(
				CudaDriverBindings.callInt(drv.cuModuleGetFunction, fnSlot, module, arena.allocateFrom(entry)),
				"cuModuleGetFunction(" + entry + ")");
		return fnSlot.get(ADDRESS, 0);
	}

	/**
	 * As {@link GqaAttentionKernel#launch}, over caches whose keys are stored as {@code formats.k()} and values as
	 * {@code formats.v()}. A quantized format needs {@code kvDim} to be a multiple of {@link DeviceKvFormat#BLOCK}.
	 */
	void launch(MemorySegment qBatch, MemorySegment kPtrs, MemorySegment vPtrs, MemorySegment seqLens,
			MemorySegment outBatch, int batch, int numHeads, int gqaRatio, int headDim, int kvDim, int rowsPerBlock,
			int window, DeviceKvFormats formats, MemorySegment stream) {
		Objects.requireNonNull(qBatch, "qBatch");
		Objects.requireNonNull(kPtrs, "kPtrs");
		Objects.requireNonNull(vPtrs, "vPtrs");
		Objects.requireNonNull(seqLens, "seqLens");
		Objects.requireNonNull(outBatch, "outBatch");
		if (!GqaAttentionKernel.supportsHeadDim(headDim))
			throw new IllegalArgumentException("attention kernel runs head widths that are multiples of 4 up to "
					+ GqaAttentionKernel.MAX_HEAD_DIM + " (got " + headDim + ")");
		if (rowsPerBlock < 1 || rowsPerBlock > GqaAttentionKernel.MAX_ROWS_PER_BLOCK
				|| Integer.bitCount(rowsPerBlock) != 1)
			throw new IllegalArgumentException("rowsPerBlock must be a power of two up to "
					+ GqaAttentionKernel.MAX_ROWS_PER_BLOCK + " (got " + rowsPerBlock + ")");
		if (window < 0)
			throw new IllegalArgumentException("window must be >= 0 (got " + window + ")");
		formats.k().rowBytes(kvDim); // refuses a kvDim the format cannot store
		formats.v().rowBytes(kvDim);
		// A head of whole blocks: the decode lanes read whole blocks of one head, and a prefill tile is staged eight
		// values at a time from one block. DeviceKvFormat selects quantized formats only for such heads.
		if (formats.quantized() && headDim % DeviceKvFormat.BLOCK != 0)
			throw new IllegalArgumentException("a quantized device KV needs a head width that is a multiple of "
					+ DeviceKvFormat.BLOCK + " (got " + headDim + ")");

		CudaDriverBindings drv = CudaDriverBindings.instance();
		int width = GqaAttentionKernel.registerWidth(headDim);
		MemorySegment streamOrNull = stream == null ? MemorySegment.NULL : stream;
		if (rowsPerBlock == 1 && width <= 128) {
			launchDecode(drv, width == 64 ? fnDecodeD64 : fnDecodeD128, qBatch, kPtrs, vPtrs, seqLens, outBatch, batch,
					numHeads, gqaRatio, headDim, kvDim, window, formats, streamOrNull);
			return;
		}
		MemorySegment fn = width == 64 ? fnD64 : width == 128 ? fnD128 : fnD256;
		int gridX = (batch + rowsPerBlock - 1) / rowsPerBlock;
		try (Arena arena = Arena.ofConfined()) {
			MemorySegment params = arena.allocate(ADDRESS, 14);
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
			params.setAtIndex(ADDRESS, 12, intArg(arena, formats.k().code));
			params.setAtIndex(ADDRESS, 13, intArg(arena, formats.v().code));
			CudaDriverBindings.check(CudaDriverBindings.callInt(drv.cuLaunchKernel, fn, gridX, numHeads, 1,
					GQA_THREADS, 1, 1, 0, streamOrNull, params, MemorySegment.NULL), "cuLaunchKernel(gqa_attention_kv)");
		}
	}

	/** Decode rows (one a block, each over its own cache) through the wide decode entry {@code fn}. */
	private static void launchDecode(CudaDriverBindings drv, MemorySegment fn, MemorySegment qBatch, MemorySegment kPtrs,
			MemorySegment vPtrs, MemorySegment seqLens, MemorySegment outBatch, int batch, int numHeads, int gqaRatio,
			int headDim, int kvDim, int window, DeviceKvFormats formats, MemorySegment stream) {
		try (Arena arena = Arena.ofConfined()) {
			MemorySegment params = arena.allocate(ADDRESS, 13);
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
			params.setAtIndex(ADDRESS, 10, intArg(arena, window));
			params.setAtIndex(ADDRESS, 11, intArg(arena, formats.k().code));
			params.setAtIndex(ADDRESS, 12, intArg(arena, formats.v().code));
			CudaDriverBindings.check(CudaDriverBindings.callInt(drv.cuLaunchKernel, fn, batch, numHeads, 1,
					DECODE_THREADS, 1, 1, 0, stream, params, MemorySegment.NULL), "cuLaunchKernel(gqa_decode_kv)");
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
