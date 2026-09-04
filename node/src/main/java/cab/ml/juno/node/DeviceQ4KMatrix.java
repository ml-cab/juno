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

/**
 * Row-major Q4_K packed weight matrix resident on the GPU.
 *
 * <p>Stores raw GGUF Q4_K bytes (144 B / 256 elements) without host dequant.
 * Used by the fused dequant+GEMV path ({@link Q4KMmqKernel}).
 */
public final class DeviceQ4KMatrix implements AutoCloseable {

	private final GpuContext ctx;
	private final GpuBindings gpu;
	private final MemorySegment dA;
	private final int rows;
	private final int cols;
	private final long byteLength;
	private volatile boolean closed;

	private DeviceQ4KMatrix(GpuContext ctx, MemorySegment dA, int rows, int cols, long byteLength) {
		this.ctx = ctx;
		this.gpu = ctx.bindings();
		this.dA = dA;
		this.rows = rows;
		this.cols = cols;
		this.byteLength = byteLength;
	}

	/**
	 * Uploads packed Q4_K bytes for a row-major matrix with {@code cols} divisible
	 * by {@link QuantizationLayout#QK_K}.
	 */
	public static DeviceQ4KMatrix upload(GpuContext ctx, byte[] raw, int rows, int cols) {
		if (ctx == null)
			throw new IllegalArgumentException("ctx must not be null");
		QuantizationLayout.Q4_K.validateMatrix(rows, cols);
		long expected = QuantizationLayout.Q4_K.encodedBytes((long) rows * cols);
		if (raw == null || raw.length != expected)
			throw new IllegalArgumentException(
					"Q4_K raw length " + (raw == null ? -1 : raw.length) + " != expected " + expected);

		GpuBindings gpu = ctx.bindings();
		MemorySegment dA = gpu.deviceMalloc(ctx.deviceIndex(), expected);
		try (Arena staging = Arena.ofConfined()) {
			MemorySegment host = staging.allocate(expected);
			MemorySegment.copy(MemorySegment.ofArray(raw), 0, host, 0, expected);
			GpuBindings.check(
					GpuBindings.callInt(gpu.gpuMemcpy(), dA, host, expected, GpuBindings.H2D),
					"memcpy(Q4_K H2D)");
		}
		return new DeviceQ4KMatrix(ctx, dA, rows, cols, expected);
	}

	public int rows() {
		return rows;
	}

	public int cols() {
		return cols;
	}

	public long byteLength() {
		return byteLength;
	}

	MemorySegment devicePointer() {
		if (closed)
			throw new IllegalStateException("DeviceQ4KMatrix already closed");
		return dA;
	}

	public boolean isClosed() {
		return closed;
	}

	@Override
	public void close() {
		if (!closed) {
			closed = true;
			GpuBindings.callInt(gpu.gpuSetDevice(), ctx.deviceIndex());
			gpu.deviceFree(dA);
		}
	}
}
