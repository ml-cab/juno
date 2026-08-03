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

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

/**
 * Reusable host packing and device scratch for microbatched frozen linears.
 *
 * <p>Columns are packed column-major for cuBLAS/rocBLAS GEMM:
 * {@code packed[c + b * dim] = columns[b][c]}. Device buffers grow lazily and
 * are freed by {@link #close()}.
 */
final class DeviceActivationBatch implements AutoCloseable {

	private final GpuContext ctx;
	private final GpuBindings gpu;
	private MemorySegment dIn;
	private MemorySegment dOut;
	private long dInBytes;
	private long dOutBytes;
	private boolean closed;

	DeviceActivationBatch(GpuContext ctx) {
		if (ctx == null)
			throw new IllegalArgumentException("ctx must not be null");
		this.ctx = ctx;
		this.gpu = ctx.bindings();
	}

	/**
	 * Packs {@code columns[0..batch)} each of length {@code dim} into a contiguous
	 * column-major {@code float[]} of length {@code dim * batch}.
	 */
	static float[] packColumns(float[][] columns, int batch, int dim) {
		if (batch < 0 || dim < 0)
			throw new IllegalArgumentException("batch and dim must be non-negative");
		if (columns.length < batch)
			throw new IllegalArgumentException("columns.length=" + columns.length + " < batch=" + batch);
		float[] packed = new float[dim * batch];
		for (int b = 0; b < batch; b++) {
			float[] col = columns[b];
			if (col == null || col.length != dim)
				throw new IllegalArgumentException(
						"columns[" + b + "] length must be " + dim + " (got " + (col == null ? -1 : col.length) + ")");
			System.arraycopy(col, 0, packed, b * dim, dim);
		}
		return packed;
	}

	/**
	 * Unpacks column-major {@code packed} of length {@code dim * batch} into
	 * {@code out[0..batch)} each of length {@code dim}.
	 */
	static void unpackColumns(float[] packed, float[][] out, int batch, int dim) {
		if (packed.length < (long) dim * batch)
			throw new IllegalArgumentException("packed too short");
		if (out.length < batch)
			throw new IllegalArgumentException("out.length < batch");
		for (int b = 0; b < batch; b++) {
			if (out[b] == null || out[b].length != dim)
				out[b] = new float[dim];
			System.arraycopy(packed, b * dim, out[b], 0, dim);
		}
	}

	MemorySegment ensureInput(long bytes) {
		ensureOpen();
		if (dIn == null || dInBytes < bytes) {
			if (dIn != null)
				gpu.deviceFree(dIn);
			dIn = gpu.deviceMalloc(ctx.deviceIndex(), bytes);
			dInBytes = bytes;
		}
		return dIn;
	}

	MemorySegment ensureOutput(long bytes) {
		ensureOpen();
		if (dOut == null || dOutBytes < bytes) {
			if (dOut != null)
				gpu.deviceFree(dOut);
			dOut = gpu.deviceMalloc(ctx.deviceIndex(), bytes);
			dOutBytes = bytes;
		}
		return dOut;
	}

	/** Synchronous H2D of a host float array into {@code dest}. */
	void copyH2D(MemorySegment dest, float[] host) {
		ensureOpen();
		long bytes = (long) host.length * Float.BYTES;
		try (Arena staging = Arena.ofConfined()) {
			MemorySegment nativeHost = staging.allocate(bytes);
			nativeHost.copyFrom(MemorySegment.ofArray(host));
			GpuBindings.check(
					GpuBindings.callInt(gpu.gpuMemcpy(), dest, nativeHost, bytes, GpuBindings.H2D),
					"memcpy(batch H2D)");
		}
	}

	/** Synchronous D2H into a new host float array. */
	float[] copyD2H(MemorySegment src, int floats) {
		ensureOpen();
		long bytes = (long) floats * Float.BYTES;
		float[] host = new float[floats];
		try (Arena staging = Arena.ofConfined()) {
			MemorySegment nativeHost = staging.allocate(bytes);
			GpuBindings.check(
					GpuBindings.callInt(gpu.gpuMemcpy(), nativeHost, src, bytes, GpuBindings.D2H),
					"memcpy(batch D2H)");
			MemorySegment.copy(nativeHost, JAVA_FLOAT, 0, host, 0, floats);
		}
		return host;
	}

	@Override
	public void close() {
		if (closed)
			return;
		closed = true;
		if (dIn != null) {
			gpu.deviceFree(dIn);
			dIn = null;
		}
		if (dOut != null) {
			gpu.deviceFree(dOut);
			dOut = null;
		}
	}

	private void ensureOpen() {
		if (closed)
			throw new IllegalStateException("DeviceActivationBatch is closed");
	}
}
