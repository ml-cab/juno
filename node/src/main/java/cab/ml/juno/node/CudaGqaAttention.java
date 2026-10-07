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

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;

/**
 * Handler-facing entry point for the GPU-resident attention path
 * ({@code --gpu-attention}).
 *
 * <p>{@link #attendBatched} dispatches to the real CUDA kernel
 * ({@code gqa_attention.cu} / {@link GqaAttentionKernel}) that computes
 * grouped-query attention (QK^T + softmax + weighted-V-sum — the same math as
 * {@link GqaMath}, extracted from {@code LlamaTransformerHandler}) in
 * parallel on-device, reading K/V straight from a {@link DeviceKvCache}
 * without a host round trip. Wired into all three of
 * {@code LlamaTransformerHandler}'s attention call sites (prefill window,
 * single-token decode, {@code --parallel} multi-decode) — each builds its own
 * {@code B}-sized batch of query positions and falls back to the scalar
 * {@link GqaMath} path when this returns {@code false} (kernel unavailable).
 */
final class CudaGqaAttention {

	private final GpuContext ctx;
	private final GpuBindings gpu;

	/**
	 * Device scratch, pooled by concurrent callers rather than kept per thread:
	 * this path runs outside the context's serialization lock, so callers at the
	 * same time need separate entries, and a request's thread ends with it.
	 */
	private final DeviceScratchPool<GqaScratch> scratch;

	private static final class GqaScratch {
		final DeviceScratchSlot q, out, kPtrs, vPtrs, seqLens;

		GqaScratch(GpuContext ctx) {
			q = DeviceScratchSlot.device(ctx);
			out = DeviceScratchSlot.device(ctx);
			kPtrs = DeviceScratchSlot.device(ctx);
			vPtrs = DeviceScratchSlot.device(ctx);
			seqLens = DeviceScratchSlot.device(ctx);
		}

		long deviceBytes() {
			return q.bytes() + out.bytes() + kPtrs.bytes() + vPtrs.bytes() + seqLens.bytes();
		}
	}

	private CudaGqaAttention(GpuContext ctx) {
		this.ctx = ctx;
		this.gpu = ctx.bindings();
		this.scratch = new DeviceScratchPool<>(() -> new GqaScratch(ctx), CudaGqaAttention::free);
	}

	private static void free(GqaScratch s) {
		s.q.free();
		s.out.free();
		s.kPtrs.free();
		s.vPtrs.free();
		s.seqLens.free();
	}

	/** Frees the pooled device scratch. A call after this still works and frees its own. */
	void close() {
		scratch.close();
	}

	/** Device bytes held by idle pooled scratch entries. */
	long scratchDeviceBytes() {
		return scratch.idleBytes(GqaScratch::deviceBytes);
	}

	/** Returns a usable instance for {@code ctx}, or {@code null} when the backend isn't CUDA. */
	static CudaGqaAttention tryCreate(GpuContext ctx) {
		if (ctx == null || !"cuda".equals(ctx.backendLabel()))
			return null;
		return new CudaGqaAttention(ctx);
	}

	/** Allocates one device KV mirror per layer for a new request. */
	DeviceKvCache[] newLayers(int layerCount, int kvDim) {
		return DeviceKvCache.newLayers(ctx, layerCount, kvDim);
	}

	/**
	 * Real GPU-parallel attention for {@code B = qBatch.length} (batch-row,
	 * head) pairs in one kernel launch. {@code kv[b]} may repeat the same
	 * {@link DeviceKvCache} across {@code b} (a growing prefill window all
	 * sharing one request/layer cache) or differ per {@code b} (independent
	 * {@code --parallel} decode streams) — the kernel takes one device pointer
	 * per {@code b} either way.
	 *
	 * @return {@code false} (writing nothing) if the kernel failed to load or does
	 *         not run heads of {@code headDim} values — caller must fall back to the
	 *         scalar {@link GqaMath} path
	 */
	boolean attendBatched(DeviceKvCache[] kv, float[][] qBatch, int[] seqLens, float[][] outBatch,
			int numHeads, int headDim, int gqaRatio, int kvDim) {
		return attendBatched(kv, qBatch, seqLens, outBatch, numHeads, headDim, gqaRatio, kvDim, 0);
	}

	/**
	 * As {@link #attendBatched(DeviceKvCache[], float[][], int[], float[][], int, int, int, int)},
	 * with row {@code b} attending over its last {@code window} keys only
	 * ({@code 0}: no window).
	 */
	boolean attendBatched(DeviceKvCache[] kv, float[][] qBatch, int[] seqLens, float[][] outBatch,
			int numHeads, int headDim, int gqaRatio, int kvDim, int window) {
		if (!GqaAttentionKernel.supportsHeadDim(headDim))
			return false;
		GqaAttentionKernel kernel = GqaAttentionKernel.tryLoad();
		if (kernel == null)
			return false;

		int batch = qBatch.length;
		int rowDim = numHeads * headDim;
		boolean oneCache = true;
		for (int b = 1; b < batch && oneCache; b++)
			oneCache = kv[b] == kv[0];

		long qBytes = (long) batch * rowDim * Float.BYTES;
		long outBytes = qBytes;
		long ptrBytes = (long) batch * ADDRESS.byteSize();
		long seqLensBytes = (long) batch * Integer.BYTES;

		GqaScratch s = scratch.acquire();
		try {
			return attendWith(s, kernel, kv, qBatch, seqLens, outBatch, batch, numHeads, headDim, gqaRatio, kvDim,
					rowDim, GqaAttentionKernel.rowsPerBlock(oneCache, batch), window, qBytes, outBytes, ptrBytes,
					seqLensBytes);
		} finally {
			scratch.release(s);
		}
	}

	private boolean attendWith(GqaScratch s, GqaAttentionKernel kernel, DeviceKvCache[] kv, float[][] qBatch,
			int[] seqLens, float[][] outBatch, int batch, int numHeads, int headDim, int gqaRatio, int kvDim,
			int rowDim, int rowsPerBlock, int window, long qBytes, long outBytes, long ptrBytes,
			long seqLensBytes) {
		MemorySegment dQ = s.q.ensure(qBytes);
		MemorySegment dOut = s.out.ensure(outBytes);
		MemorySegment dKPtrs = s.kPtrs.ensure(ptrBytes);
		MemorySegment dVPtrs = s.vPtrs.ensure(ptrBytes);
		MemorySegment dSeqLens = s.seqLens.ensure(seqLensBytes);

		try (Arena staging = Arena.ofConfined()) {
			MemorySegment hostQ = staging.allocate(qBytes);
			for (int b = 0; b < batch; b++)
				MemorySegment.copy(qBatch[b], 0, hostQ, JAVA_FLOAT, (long) b * rowDim * Float.BYTES, rowDim);
			DeviceStaging.copy(gpu, dQ, hostQ, qBytes, GpuBindings.H2D, batch, "memcpy(gqa qBatch H2D)");

			MemorySegment hostKPtrs = staging.allocate(ptrBytes);
			MemorySegment hostVPtrs = staging.allocate(ptrBytes);
			for (int b = 0; b < batch; b++) {
				hostKPtrs.setAtIndex(ADDRESS, b, kv[b].kPointer());
				hostVPtrs.setAtIndex(ADDRESS, b, kv[b].vPointer());
			}
			DeviceStaging.copy(gpu, dKPtrs, hostKPtrs, ptrBytes, GpuBindings.H2D, batch, "memcpy(gqa kPtrs H2D)");
			DeviceStaging.copy(gpu, dVPtrs, hostVPtrs, ptrBytes, GpuBindings.H2D, batch, "memcpy(gqa vPtrs H2D)");

			MemorySegment hostSeqLens = staging.allocate(seqLensBytes);
			for (int b = 0; b < batch; b++)
				hostSeqLens.setAtIndex(JAVA_INT, b, seqLens[b]);
			DeviceStaging.copy(gpu, dSeqLens, hostSeqLens, seqLensBytes, GpuBindings.H2D, batch, "memcpy(gqa seqLens H2D)");

			long t0 = DeviceComputeClock.start(gpu, batch);
			kernel.launch(dQ, dKPtrs, dVPtrs, dSeqLens, dOut,
					batch, numHeads, gqaRatio, headDim, kvDim, rowsPerBlock, window, null);
			DeviceComputeClock.done(gpu, DeviceComputeEvent.GQA_ATTENTION, batch, t0);

			MemorySegment hostOut = staging.allocate(outBytes);
			DeviceStaging.copy(gpu, hostOut, dOut, outBytes, GpuBindings.D2H, batch, "memcpy(gqa outBatch D2H)");
			for (int b = 0; b < batch; b++) {
				if (outBatch[b] == null || outBatch[b].length != rowDim)
					outBatch[b] = new float[rowDim];
				MemorySegment.copy(hostOut, JAVA_FLOAT, (long) b * rowDim * Float.BYTES, outBatch[b], 0, rowDim);
			}
		}
		return true;
	}
}
