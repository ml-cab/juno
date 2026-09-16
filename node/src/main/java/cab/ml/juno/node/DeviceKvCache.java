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
import java.util.concurrent.atomic.AtomicLong;

import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * Device-resident mirror of one request/layer's K/V cache, for the GPU-resident
 * attention path ({@link CudaGqaAttention}).
 *
 * <p>Stored as real IEEE FP16 (binary16) — half the bytes of the host
 * {@code SessionKvTensor}'s {@code float32} storage (note: {@code kvcache}'s
 * {@code KvElementType.F16} is a misnomer for {@code float32}; this class uses
 * genuine half-precision, matching {@link DeviceHalfMatrix}'s convention).
 * Dot products and softmax are computed in FP32 by the attention path that
 * consumes this cache.
 *
 * <p><b>Numerical note:</b> like every other reduced-precision path in this
 * codebase (FP16-resident weights, {@code --mmq}'s Q4_K/Q8_1 GEMV), FP16 K/V
 * storage can occasionally flip a very close greedy-decoding decision after
 * enough decode steps compound the rounding — single-step logits stay within
 * a tight tolerance of the full-precision CPU path (see
 * {@code GqaAttentionKernelParityTest}), but multi-token greedy-sequence
 * identity across many steps is not guaranteed, consistent with this
 * project's other quantized/GPU-resident paths (none of which claim
 * multi-step token-sequence identity either — only per-step numerical
 * closeness).
 *
 * <p>This is a mirror, not a replacement: {@code LlamaTransformerHandler}
 * dual-writes — the host {@code SessionKvTensor} keeps recording every token
 * exactly as before (preserving CPU fallback, paged/continuous KV, quantized
 * KV, and cluster eviction/restore paths untouched); this class only receives
 * an additional copy of the same pre-quantization row when the GPU attention
 * path is active for a GPU-resident layer.
 *
 * <p>Growth is grow-and-preserve (not grow-and-discard like per-call scratch
 * buffers): KV history is irreplaceable, so growing allocates a larger device
 * buffer and copies the valid prefix device-to-device before freeing the old
 * one. Mirrors {@code DenseKvTensor}'s doubling policy and {@code MAX_SEQ_LEN}.
 *
 * <p>Vendor-neutral by construction (programs against {@link GpuBindings}, not
 * a CUDA-specific type) even though the attention kernel that reads it is
 * CUDA-only in v1 — matching {@link DeviceHalfMatrix}'s style.
 */
final class DeviceKvCache implements AutoCloseable {

	static final int INITIAL_SEQ_CAPACITY = 64;
	/** Mirrors {@code DenseKvTensor.MAX_SEQ_LEN}. */
	static final int MAX_SEQ_LEN = 32768;

	// Package-visible leak-freedom counter for DeviceKvCacheLifecycleTest — total bytes
	// currently allocated across all live DeviceKvCache instances (both K and V buffers).
	private static final AtomicLong ALLOCATED_BYTES = new AtomicLong();

	static long allocatedBytes() {
		return ALLOCATED_BYTES.get();
	}

	private final GpuContext ctx;
	private final GpuBindings gpu;
	private final int kvDim;
	private MemorySegment dK;
	private MemorySegment dV;
	private int capacityTokens;
	private boolean closed;

	DeviceKvCache(GpuContext ctx, int kvDim) {
		this(ctx, kvDim, INITIAL_SEQ_CAPACITY);
	}

	DeviceKvCache(GpuContext ctx, int kvDim, int initialTokens) {
		if (ctx == null)
			throw new IllegalArgumentException("ctx must not be null");
		if (kvDim < 1)
			throw new IllegalArgumentException("kvDim must be >= 1");
		if (initialTokens < 1)
			throw new IllegalArgumentException("initialTokens must be >= 1");
		this.ctx = ctx;
		this.gpu = ctx.bindings();
		this.kvDim = kvDim;
		this.capacityTokens = initialTokens;
		long bytes = bytesFor(initialTokens);
		this.dK = gpu.deviceMalloc(ctx.deviceIndex(), bytes);
		this.dV = gpu.deviceMalloc(ctx.deviceIndex(), bytes);
		ALLOCATED_BYTES.addAndGet(2 * bytes);
	}

	/** Allocate one device mirror per layer, same granularity as {@code SessionKvTensor[]}. */
	static DeviceKvCache[] newLayers(GpuContext ctx, int layerCount, int kvDim) {
		if (layerCount < 1)
			throw new IllegalArgumentException("layerCount must be >= 1");
		DeviceKvCache[] out = new DeviceKvCache[layerCount];
		for (int i = 0; i < layerCount; i++)
			out[i] = new DeviceKvCache(ctx, kvDim);
		return out;
	}

	private long bytesFor(int tokens) {
		return (long) tokens * kvDim * Short.BYTES;
	}

	int capacityTokens() {
		return capacityTokens;
	}

	int kvDim() {
		return kvDim;
	}

	/** Grow so position {@code pos} (0-based) fits, preserving all previously written rows. */
	void ensureCapacity(int pos) {
		if (closed)
			throw new IllegalStateException("DeviceKvCache already closed");
		if (pos < 0)
			throw new IllegalArgumentException("pos must be >= 0");
		if (pos >= MAX_SEQ_LEN)
			throw new IllegalStateException("KV cache position " + pos + " exceeds MAX_SEQ_LEN=" + MAX_SEQ_LEN);
		int need = pos + 1;
		if (need <= capacityTokens)
			return;
		int newCap = capacityTokens;
		while (newCap < need)
			newCap = Math.min(newCap * 2, MAX_SEQ_LEN);
		if (newCap < need)
			throw new IllegalStateException("KV cache position " + pos + " exceeds MAX_SEQ_LEN=" + MAX_SEQ_LEN);
		grow(newCap);
	}

	private void grow(int newCap) {
		long oldBytes = bytesFor(capacityTokens);
		long newBytes = bytesFor(newCap);
		MemorySegment newK = gpu.deviceMalloc(ctx.deviceIndex(), newBytes);
		MemorySegment newV = gpu.deviceMalloc(ctx.deviceIndex(), newBytes);
		GpuBindings.check(
				GpuBindings.callInt(gpu.gpuMemcpy(), newK, dK, oldBytes, GpuBindings.D2D),
				"memcpy(K D2D grow)");
		GpuBindings.check(
				GpuBindings.callInt(gpu.gpuMemcpy(), newV, dV, oldBytes, GpuBindings.D2D),
				"memcpy(V D2D grow)");
		gpu.deviceFree(dK);
		gpu.deviceFree(dV);
		ALLOCATED_BYTES.addAndGet(2 * (newBytes - oldBytes));
		dK = newK;
		dV = newV;
		capacityTokens = newCap;
	}

	/** Appends one K/V row at {@code pos}, growing first if needed. Packs float32 -> FP16 host-side. */
	void appendToken(int pos, float[] k, float[] v) {
		ensureCapacity(pos);
		if (k.length < kvDim || v.length < kvDim)
			throw new IllegalArgumentException("k/v must hold at least kvDim=" + kvDim + " floats");
		long rowBytes = (long) kvDim * Short.BYTES;
		long offset = (long) pos * rowBytes;
		try (Arena staging = Arena.ofConfined()) {
			MemorySegment stagingK = staging.allocate(rowBytes);
			MemorySegment stagingV = staging.allocate(rowBytes);
			for (int i = 0; i < kvDim; i++) {
				stagingK.setAtIndex(JAVA_SHORT, i, Float.floatToFloat16(k[i]));
				stagingV.setAtIndex(JAVA_SHORT, i, Float.floatToFloat16(v[i]));
			}
			GpuBindings.check(
					GpuBindings.callInt(gpu.gpuMemcpy(), dK.asSlice(offset, rowBytes), stagingK, rowBytes, GpuBindings.H2D),
					"memcpy(K row H2D)");
			GpuBindings.check(
					GpuBindings.callInt(gpu.gpuMemcpy(), dV.asSlice(offset, rowBytes), stagingV, rowBytes, GpuBindings.H2D),
					"memcpy(V row H2D)");
		}
	}

	/**
	 * Downloads positions {@code [0, seqLen)} back to a freshly-allocated host
	 * {@code float32} array (unpacked from FP16). Used by the Stage-1 CPU-oracle
	 * stub in {@link CudaGqaAttention} and by tests; the real kernel (Stage 2+)
	 * reads {@link #kPointer()}/{@link #vPointer()} directly on-device instead.
	 */
	float[] downloadK(int seqLen) {
		return download(dK, seqLen);
	}

	float[] downloadV(int seqLen) {
		return download(dV, seqLen);
	}

	private float[] download(MemorySegment d, int seqLen) {
		if (closed)
			throw new IllegalStateException("DeviceKvCache already closed");
		if (seqLen < 1 || seqLen > capacityTokens)
			throw new IllegalArgumentException("seqLen=" + seqLen + " out of range [1," + capacityTokens + "]");
		int n = seqLen * kvDim;
		long bytes = (long) n * Short.BYTES;
		float[] out = new float[n];
		try (Arena staging = Arena.ofConfined()) {
			MemorySegment stagingHost = staging.allocate(bytes);
			GpuBindings.check(
					GpuBindings.callInt(gpu.gpuMemcpy(), stagingHost, d, bytes, GpuBindings.D2H),
					"memcpy(D2H download)");
			for (int i = 0; i < n; i++)
				out[i] = Float.float16ToFloat(stagingHost.getAtIndex(JAVA_SHORT, i));
		}
		return out;
	}

	/** Device pointer for K, valid until {@link #close()}. For the real kernel path (Stage 2+). */
	MemorySegment kPointer() {
		if (closed) throw new IllegalStateException("DeviceKvCache already closed");
		return dK;
	}

	/** Device pointer for V, valid until {@link #close()}. For the real kernel path (Stage 2+). */
	MemorySegment vPointer() {
		if (closed) throw new IllegalStateException("DeviceKvCache already closed");
		return dV;
	}

	@Override
	public void close() {
		if (!closed) {
			closed = true;
			GpuBindings.callInt(gpu.gpuSetDevice(), ctx.deviceIndex());
			long bytes = bytesFor(capacityTokens);
			gpu.deviceFree(dK);
			gpu.deviceFree(dV);
			ALLOCATED_BYTES.addAndGet(-2 * bytes);
		}
	}
}
