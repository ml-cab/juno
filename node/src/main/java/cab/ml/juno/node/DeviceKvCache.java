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
	private int validTokens;
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

	/**
	 * Length of the contiguous run of positions actually written to this mirror,
	 * starting at 0. Device memory is never zeroed, so a position that was never
	 * appended holds whatever the allocator last left there — indistinguishable,
	 * on-device, from a real K/V row. This watermark is what makes that
	 * distinguishable host-side; {@link #readableThrough(int)} is the check
	 * callers should make before handing this mirror to an attention kernel.
	 */
	int validTokens() {
		return validTokens;
	}

	/**
	 * False once the mirror has been retired (closed) after running out of device
	 * memory. A retired mirror must not be appended to or read; the request
	 * continues from the host KV tensors, which hold the same history.
	 */
	boolean live() {
		return !closed;
	}

	/**
	 * Whether attention may read positions {@code [0, seqLen)} from this mirror.
	 * False once the mirror is closed, and false whenever the host KV tensors hold
	 * history this mirror never received — a request whose prefix was restored
	 * from the KV adapter, or one whose mirror was given up mid-flight and would
	 * otherwise be silently re-created empty.
	 */
	boolean readableThrough(int seqLen) {
		return !closed && seqLen <= validTokens;
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
		MemorySegment newV;
		try {
			newV = gpu.deviceMalloc(ctx.deviceIndex(), newBytes);
		} catch (RuntimeException e) {
			// Out of memory for the second tensor: give the first one back, or it leaks
			// for the life of the process. The mirror itself is unchanged.
			gpu.deviceFree(newK);
			throw e;
		}
		DeviceStaging.copy(gpu, newK, dK, oldBytes, GpuBindings.D2D, 0, "memcpy(K D2D grow)");
		DeviceStaging.copy(gpu, newV, dV, oldBytes, GpuBindings.D2D, 0, "memcpy(V D2D grow)");
		gpu.deviceFree(dK);
		gpu.deviceFree(dV);
		ALLOCATED_BYTES.addAndGet(2 * (newBytes - oldBytes));
		dK = newK;
		dV = newV;
		capacityTokens = newCap;
	}

	/** Appends one K/V row at {@code pos}, growing first if needed. Packs float32 -> FP16 host-side. */
	void appendToken(int pos, float[] k, float[] v) {
		appendToken(pos, k, v, 1);
	}

	/**
	 * As {@link #appendToken(int, float[], float[])}, from a forward call over
	 * {@code windowSize} rows; recorded on the row copies so a prefill window's KV
	 * writes are counted with the window rather than as single-token traffic.
	 */
	void appendToken(int pos, float[] k, float[] v, int windowSize) {
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
			DeviceStaging.copy(gpu, dK.asSlice(offset, rowBytes), stagingK, rowBytes, GpuBindings.H2D, windowSize,
					"memcpy(K row H2D)");
			DeviceStaging.copy(gpu, dV.asSlice(offset, rowBytes), stagingV, rowBytes, GpuBindings.H2D, windowSize,
					"memcpy(V row H2D)");
		}
		// Only extend the watermark when this row abuts the written prefix. A write
		// past the end leaves a hole of uninitialized device memory behind it, so
		// the prefix stops being readable there and the mirror stays unusable until
		// the gap is filled in order.
		if (pos == validTokens)
			validTokens = pos + 1;
	}

	/**
	 * Appends {@code count} K/V rows at positions {@code [startPos, startPos + count)}
	 * from host rows: packed to FP16 on the host and copied as one contiguous
	 * transfer per tensor, where {@link #appendToken} pays one per row. Grows first
	 * if needed. The watermark moves exactly as {@code count} calls to
	 * {@link #appendToken} would move it.
	 */
	void appendWindow(int startPos, float[][] k, float[][] v, int count) {
		if (count < 1 || k.length < count || v.length < count)
			throw new IllegalArgumentException("window of " + count + " rows from " + k.length + " K and "
					+ v.length + " V rows");
		ensureCapacity(startPos + count - 1);
		long rowBytes = (long) kvDim * Short.BYTES;
		long bytes = rowBytes * count;
		long offset = (long) startPos * rowBytes;
		try (Arena staging = Arena.ofConfined()) {
			MemorySegment stagingK = staging.allocate(bytes);
			MemorySegment stagingV = staging.allocate(bytes);
			for (int r = 0; r < count; r++) {
				if (k[r].length < kvDim || v[r].length < kvDim)
					throw new IllegalArgumentException("k/v rows must hold at least kvDim=" + kvDim + " floats");
				long base = (long) r * kvDim;
				for (int i = 0; i < kvDim; i++) {
					stagingK.setAtIndex(JAVA_SHORT, base + i, Float.floatToFloat16(k[r][i]));
					stagingV.setAtIndex(JAVA_SHORT, base + i, Float.floatToFloat16(v[r][i]));
				}
			}
			DeviceStaging.copy(gpu, dK.asSlice(offset, bytes), stagingK, bytes, GpuBindings.H2D, count,
					"memcpy(K window H2D)");
			DeviceStaging.copy(gpu, dV.asSlice(offset, bytes), stagingV, bytes, GpuBindings.H2D, count,
					"memcpy(V window H2D)");
		}
		markWritten(startPos, count);
	}

	/**
	 * Writes {@code count} K/V rows at {@code [startPos, startPos + count)} from
	 * device FP32 rows ({@code [count][kvDim]}, row-major), cast to FP16 straight
	 * into this mirror on {@code stream}: nothing crosses the host. Grows first if
	 * needed. Asynchronous, so the watermark does not move here; the caller marks
	 * the window with {@link #markWritten} once the stream has completed and the
	 * host KV tensors hold the same rows, which keeps the mirror a copy of the host
	 * KV rather than the only one.
	 */
	void writeWindowOnDevice(int startPos, int count, MemorySegment dK32, MemorySegment dV32,
			PrefillWindowKernels kernels, MemorySegment stream) {
		if (count < 1)
			throw new IllegalArgumentException("window of " + count + " rows");
		ensureCapacity(startPos + count - 1);
		long rowBytes = (long) kvDim * Short.BYTES;
		long bytes = rowBytes * count;
		long offset = (long) startPos * rowBytes;
		long elements = (long) count * kvDim;
		kernels.toHalf(dK32, dK.asSlice(offset, bytes), elements, stream);
		kernels.toHalf(dV32, dV.asSlice(offset, bytes), elements, stream);
	}

	/**
	 * Extends the watermark over a window of rows already written to positions
	 * {@code [startPos, startPos + count)}, by the rule {@link #appendToken} applies
	 * per row: only a window that starts within the written prefix extends it.
	 */
	void markWritten(int startPos, int count) {
		if (closed)
			throw new IllegalStateException("DeviceKvCache already closed");
		if (startPos <= validTokens && startPos + count > validTokens)
			validTokens = startPos + count;
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
			DeviceStaging.copy(gpu, stagingHost, d, bytes, GpuBindings.D2H, 0, "memcpy(D2H download)");
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
