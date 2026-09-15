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
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Row-major packed K-quant weight matrix resident on the GPU (Q4_K, Q5_K, Q6_K).
 *
 * <p>Stores raw GGUF super-block bytes without host dequant, consumed by the fused
 * dequant+GEMV kernels in {@link Q4KMmqKernel}. Q4_K (144 B) and Q5_K (176 B)
 * blocks are 16-byte multiples and are uploaded verbatim. Q6_K blocks are 210 B,
 * so each is written into a {@link #Q6K_SLOT_BYTES}-byte slot (14 zero pad bytes)
 * to keep the kernel's vector loads 16-byte aligned.
 *
 * <p>The class name predates Q5_K / Q6_K support and is kept to avoid churn in
 * the handler / MatVec surface; {@link #quantType()} tells the kernel which
 * decode to run.
 */
public final class DeviceQ4KMatrix implements AutoCloseable {

	/** Device slot size for one Q6_K super-block (210 raw bytes + 14 pad). */
	static final int Q6K_SLOT_BYTES = 224;

	private final GpuContext ctx;
	private final GpuBindings gpu;
	private final MemorySegment dA;
	private final int rows;
	private final int cols;
	private final int quantType;
	private final long byteLength;
	private final AtomicBoolean closed = new AtomicBoolean();

	private DeviceQ4KMatrix(GpuContext ctx, MemorySegment dA, int rows, int cols, int quantType,
			long byteLength) {
		this.ctx = ctx;
		this.gpu = ctx.bindings();
		this.dA = dA;
		this.rows = rows;
		this.cols = cols;
		this.quantType = quantType;
		this.byteLength = byteLength;
	}

	/** True when {@code typeId} has a fused device GEMV kernel. */
	public static boolean supportsType(int typeId) {
		return typeId == QuantizationLayout.TYPE_Q4_K
				|| typeId == QuantizationLayout.TYPE_Q5_K
				|| typeId == QuantizationLayout.TYPE_Q6_K;
	}

	/**
	 * Uploads packed Q4_K bytes for a row-major matrix with {@code cols} divisible
	 * by {@link QuantizationLayout#QK_K}.
	 */
	public static DeviceQ4KMatrix upload(GpuContext ctx, byte[] raw, int rows, int cols) {
		return upload(ctx, raw, rows, cols, QuantizationLayout.TYPE_Q4_K);
	}

	/**
	 * Uploads packed K-quant bytes ({@code typeId} in Q4_K / Q5_K / Q6_K) for a
	 * row-major matrix with {@code cols} divisible by {@link QuantizationLayout#QK_K}.
	 */
	public static DeviceQ4KMatrix upload(GpuContext ctx, byte[] raw, int rows, int cols, int typeId) {
		if (ctx == null)
			throw new IllegalArgumentException("ctx must not be null");
		if (!supportsType(typeId))
			throw new IllegalArgumentException("No fused GEMV kernel for GGML type " + typeId);
		QuantizationLayout layout = QuantizationLayout.require(typeId);
		layout.validateMatrix(rows, cols);
		long expected = layout.encodedBytes((long) rows * cols);
		if (raw == null || raw.length != expected)
			throw new IllegalArgumentException(
					layout.name() + " raw length " + (raw == null ? -1 : raw.length) + " != expected " + expected);

		byte[] deviceBytes = typeId == QuantizationLayout.TYPE_Q6_K ? padQ6KBlocks(raw) : raw;
		long deviceLength = deviceBytes.length;
		GpuBindings gpu = ctx.bindings();
		MemorySegment dA = gpu.deviceMalloc(ctx.deviceIndex(), deviceLength);
		try (Arena staging = Arena.ofConfined()) {
			MemorySegment host = staging.allocate(deviceLength);
			MemorySegment.copy(MemorySegment.ofArray(deviceBytes), 0, host, 0, deviceLength);
			GpuBindings.check(
					GpuBindings.callInt(gpu.gpuMemcpy(), dA, host, deviceLength, GpuBindings.H2D),
					"memcpy(" + layout.name() + " H2D)");
		}
		return new DeviceQ4KMatrix(ctx, dA, rows, cols, typeId, deviceLength);
	}

	/**
	 * Re-slots raw Q6_K super-blocks (210 B) into {@link #Q6K_SLOT_BYTES}-byte slots
	 * so every block starts 16-byte aligned; pad bytes are zero and never read.
	 */
	static byte[] padQ6KBlocks(byte[] raw) {
		int blockBytes = QuantizationLayout.Q6_K.blockBytes();
		if (raw.length % blockBytes != 0)
			throw new IllegalArgumentException("Q6_K raw length " + raw.length + " not a multiple of " + blockBytes);
		int blocks = raw.length / blockBytes;
		long paddedLength = (long) blocks * Q6K_SLOT_BYTES;
		if (paddedLength > Integer.MAX_VALUE)
			throw new IllegalArgumentException("Q6_K padded length overflows int for " + blocks + " blocks");
		byte[] padded = new byte[(int) paddedLength];
		for (int b = 0; b < blocks; b++)
			System.arraycopy(raw, b * blockBytes, padded, b * Q6K_SLOT_BYTES, blockBytes);
		return padded;
	}

	public int rows() {
		return rows;
	}

	public int cols() {
		return cols;
	}

	/** GGML type id of the packed blocks (Q4_K / Q5_K / Q6_K). */
	public int quantType() {
		return quantType;
	}

	/** Device bytes held (includes Q6_K slot padding). */
	public long byteLength() {
		return byteLength;
	}

	MemorySegment devicePointer() {
		if (closed.get())
			throw new IllegalStateException("DeviceQ4KMatrix already closed");
		return dA;
	}

	public boolean isClosed() {
		return closed.get();
	}

	@Override
	public void close() {
		if (closed.compareAndSet(false, true)) {
			GpuBindings.callInt(gpu.gpuSetDevice(), ctx.deviceIndex());
			gpu.deviceFree(dA);
		}
	}
}
