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

	private static final ThreadLocal<GqaScratch> SCRATCH = ThreadLocal.withInitial(GqaScratch::new);

	private static final class GqaScratch {
		MemorySegment dQ, dOut, dScores, dKPtrs, dVPtrs, dSeqLens;
		long qBytes, outBytes, scoresBytes, ptrBytes, seqLensBytes;
	}

	private CudaGqaAttention(GpuContext ctx) {
		this.ctx = ctx;
		this.gpu = ctx.bindings();
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
	 * @return {@code false} (writing nothing) if the kernel failed to load —
	 *         caller must fall back to the scalar {@link GqaMath} path
	 */
	boolean attendBatched(DeviceKvCache[] kv, float[][] qBatch, int[] seqLens, float[][] outBatch,
			int numHeads, int headDim, int gqaRatio, int kvDim) {
		GqaAttentionKernel kernel = GqaAttentionKernel.tryLoad();
		if (kernel == null)
			return false;

		int batch = qBatch.length;
		int rowDim = numHeads * headDim;
		int maxSeqLen = 1;
		for (int s : seqLens)
			maxSeqLen = Math.max(maxSeqLen, s);

		long qBytes = (long) batch * rowDim * Float.BYTES;
		long outBytes = qBytes;
		long scoresBytes = (long) batch * numHeads * maxSeqLen * Float.BYTES;
		long ptrBytes = (long) batch * ADDRESS.byteSize();
		long seqLensBytes = (long) batch * Integer.BYTES;

		GqaScratch s = SCRATCH.get();
		int dev = ctx.deviceIndex();
		if (s.qBytes < qBytes) {
			gpu.deviceFree(s.dQ);
			s.dQ = gpu.deviceMalloc(dev, qBytes);
			s.qBytes = qBytes;
		}
		if (s.outBytes < outBytes) {
			gpu.deviceFree(s.dOut);
			s.dOut = gpu.deviceMalloc(dev, outBytes);
			s.outBytes = outBytes;
		}
		if (s.scoresBytes < scoresBytes) {
			gpu.deviceFree(s.dScores);
			s.dScores = gpu.deviceMalloc(dev, scoresBytes);
			s.scoresBytes = scoresBytes;
		}
		if (s.ptrBytes < ptrBytes) {
			gpu.deviceFree(s.dKPtrs);
			gpu.deviceFree(s.dVPtrs);
			s.dKPtrs = gpu.deviceMalloc(dev, ptrBytes);
			s.dVPtrs = gpu.deviceMalloc(dev, ptrBytes);
			s.ptrBytes = ptrBytes;
		}
		if (s.seqLensBytes < seqLensBytes) {
			gpu.deviceFree(s.dSeqLens);
			s.dSeqLens = gpu.deviceMalloc(dev, seqLensBytes);
			s.seqLensBytes = seqLensBytes;
		}

		try (Arena staging = Arena.ofConfined()) {
			MemorySegment hostQ = staging.allocate(qBytes);
			for (int b = 0; b < batch; b++)
				MemorySegment.copy(qBatch[b], 0, hostQ, JAVA_FLOAT, (long) b * rowDim * Float.BYTES, rowDim);
			GpuBindings.check(
					GpuBindings.callInt(gpu.gpuMemcpy(), s.dQ, hostQ, qBytes, GpuBindings.H2D),
					"memcpy(gqa qBatch H2D)");

			MemorySegment hostKPtrs = staging.allocate(ptrBytes);
			MemorySegment hostVPtrs = staging.allocate(ptrBytes);
			for (int b = 0; b < batch; b++) {
				hostKPtrs.setAtIndex(ADDRESS, b, kv[b].kPointer());
				hostVPtrs.setAtIndex(ADDRESS, b, kv[b].vPointer());
			}
			GpuBindings.check(
					GpuBindings.callInt(gpu.gpuMemcpy(), s.dKPtrs, hostKPtrs, ptrBytes, GpuBindings.H2D),
					"memcpy(gqa kPtrs H2D)");
			GpuBindings.check(
					GpuBindings.callInt(gpu.gpuMemcpy(), s.dVPtrs, hostVPtrs, ptrBytes, GpuBindings.H2D),
					"memcpy(gqa vPtrs H2D)");

			MemorySegment hostSeqLens = staging.allocate(seqLensBytes);
			for (int b = 0; b < batch; b++)
				hostSeqLens.setAtIndex(JAVA_INT, b, seqLens[b]);
			GpuBindings.check(
					GpuBindings.callInt(gpu.gpuMemcpy(), s.dSeqLens, hostSeqLens, seqLensBytes, GpuBindings.H2D),
					"memcpy(gqa seqLens H2D)");

			kernel.launch(s.dQ, s.dKPtrs, s.dVPtrs, s.dSeqLens, s.dScores, s.dOut,
					batch, numHeads, gqaRatio, headDim, kvDim, maxSeqLen, null);

			MemorySegment hostOut = staging.allocate(outBytes);
			GpuBindings.check(
					GpuBindings.callInt(gpu.gpuMemcpy(), hostOut, s.dOut, outBytes, GpuBindings.D2H),
					"memcpy(gqa outBatch D2H)");
			for (int b = 0; b < batch; b++) {
				if (outBatch[b] == null || outBatch[b].length != rowDim)
					outBatch[b] = new float[rowDim];
				MemorySegment.copy(hostOut, JAVA_FLOAT, (long) b * rowDim * Float.BYTES, outBatch[b], 0, rowDim);
			}
		}
		return true;
	}
}
