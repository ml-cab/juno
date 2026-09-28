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

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

/**
 * A batch of activation rows held in device memory across consecutive
 * operations: {@code capacityRows x dim} floats, row-major, of which the first
 * {@link #rows()} are valid. Allocated by, and ordered on, a
 * {@link ResidentChain}.
 *
 * <p><b>Materialization boundary.</b> Data crosses between host and device in
 * exactly two places. {@link #upload} copies host rows in (asynchronously, on
 * the chain's stream); {@link #materialize} copies the valid rows out and is the
 * only point the host waits for device work and sees its results. Operations in
 * between ({@link CudaRmsNorm#normalizeResident}, {@link CudaRope#applyResident})
 * read and write device memory only, and never write through to a host array.
 * A consumer that is not device-resident - the host-side matrix-vector path, the
 * sampler, a gRPC hand-off to the next node - takes a materialized copy.
 *
 * <p>Both copies go through one pinned host buffer, so the driver moves the
 * bytes directly instead of through its own bounce buffer, and neither
 * allocates. An upload issued while this activation's previous upload may still
 * be reading that buffer first waits for it, so two uploads in a row cannot
 * corrupt each other.
 */
final class ResidentActivation implements AutoCloseable {

	private final ResidentChain chain;
	private final int capacityRows;
	private final int dim;
	private final long bytes;
	private final MemorySegment device;
	private final MemorySegment staging;

	private int rows;

	/** The chain's sync epoch when this activation's last upload started, or -1 when none is outstanding. */
	private long uploadEpoch = -1;
	private boolean closed;

	private ResidentActivation(ResidentChain chain, int capacityRows, int dim, long bytes, MemorySegment device,
			MemorySegment staging) {
		this.chain = chain;
		this.capacityRows = capacityRows;
		this.dim = dim;
		this.bytes = bytes;
		this.device = device;
		this.staging = staging;
	}

	static ResidentActivation allocate(ResidentChain chain, int capacityRows, int dim) {
		if (capacityRows <= 0 || dim <= 0)
			throw new IllegalArgumentException("capacityRows and dim must be positive: " + capacityRows + ", " + dim);
		long bytes = (long) capacityRows * dim * Float.BYTES;
		GpuBindings gpu = chain.bindings();
		int deviceIndex = chain.context().deviceIndex();
		MemorySegment device = gpu.deviceMalloc(deviceIndex, bytes);
		MemorySegment staging;
		try {
			staging = gpu.hostMalloc(deviceIndex, bytes);
		} catch (RuntimeException e) {
			gpu.deviceFree(device);
			throw e;
		}
		return new ResidentActivation(chain, capacityRows, dim, bytes, device, staging);
	}

	/** Uploads every row of {@code x}. */
	void upload(float[][] x) {
		upload(x, x.length);
	}

	/**
	 * Copies the first {@code count} rows of {@code x} to the device, on the
	 * chain's stream, and makes them this activation's valid rows. Returns before
	 * the transfer completes; {@code x} may be modified as soon as it returns.
	 */
	void upload(float[][] x, int count) {
		requireOpen();
		if (count <= 0 || count > capacityRows)
			throw new IllegalArgumentException(
					"upload of " + count + " rows into an activation of capacity " + capacityRows);
		if (x.length < count)
			throw new IllegalArgumentException("upload of " + count + " rows from " + x.length);
		for (int b = 0; b < count; b++)
			if (x[b] == null || x[b].length != dim)
				throw new IllegalArgumentException("row " + b + " has length "
						+ (x[b] == null ? "null" : x[b].length) + ", this activation is " + dim + " wide");

		// The previous upload's transfer reads this same pinned buffer; it must land before it is overwritten.
		if (uploadEpoch == chain.syncEpoch())
			chain.sync();
		for (int b = 0; b < count; b++)
			MemorySegment.copy(x[b], 0, staging, JAVA_FLOAT, (long) b * dim * Float.BYTES, dim);
		copyAsync(device, staging, (long) count * dim * Float.BYTES, GpuBindings.H2D, "upload(resident activation)");
		rows = count;
		uploadEpoch = chain.syncEpoch();
	}

	/**
	 * Waits for every operation issued on the chain and copies this activation's
	 * valid rows into {@code out}, reusing {@code out[b]} when it is already
	 * {@code dim} long and allocating it otherwise.
	 */
	void materialize(float[][] out) {
		requireOpen();
		if (rows == 0)
			throw new IllegalStateException("nothing has been uploaded to or written into this activation");
		if (out.length < rows)
			throw new IllegalArgumentException("materialize of " + rows + " rows into " + out.length);
		copyAsync(staging, device, (long) rows * dim * Float.BYTES, GpuBindings.D2H, "materialize(resident activation)");
		chain.sync();
		for (int b = 0; b < rows; b++) {
			if (out[b] == null || out[b].length != dim)
				out[b] = new float[dim];
			MemorySegment.copy(staging, JAVA_FLOAT, (long) b * dim * Float.BYTES, out[b], 0, dim);
		}
	}

	/**
	 * Materializes several single-row activations of one chain with one wait: every
	 * download is queued, the chain is synchronized once, then each row is copied
	 * into {@code out[i]}, which must be exactly {@code acts[i].dim()} long.
	 * Allocation-free; the region's exit when it produces more than one result.
	 */
	static void materializeRows(ResidentActivation[] acts, float[][] out) {
		if (acts.length != out.length)
			throw new IllegalArgumentException(acts.length + " activations, " + out.length + " outputs");
		ResidentChain chain = acts[0].chain;
		for (int i = 0; i < acts.length; i++) {
			ResidentActivation a = acts[i];
			a.requireOpen();
			if (a.chain != chain)
				throw new IllegalArgumentException("activation " + i + " is on another chain");
			if (a.rows != 1)
				throw new IllegalStateException("activation " + i + " holds " + a.rows + " rows, expected 1");
			if (out[i] == null || out[i].length != a.dim)
				throw new IllegalArgumentException("output " + i + " must be " + a.dim + " long");
			a.copyAsync(a.staging, a.device, (long) a.dim * Float.BYTES, GpuBindings.D2H,
					"materializeRows(resident activation)");
		}
		chain.sync();
		for (int i = 0; i < acts.length; i++)
			MemorySegment.copy(acts[i].staging, JAVA_FLOAT, 0, out[i], 0, acts[i].dim);
	}

	/** Valid rows: the last upload, or the last operation that wrote into this activation. */
	int rows() {
		return rows;
	}

	int dim() {
		return dim;
	}

	int capacityRows() {
		return capacityRows;
	}

	/** Device bytes this activation holds while open. */
	long deviceBytes() {
		return closed ? 0 : bytes;
	}

	ResidentChain chain() {
		return chain;
	}

	boolean isClosed() {
		return closed;
	}

	/** Frees this activation's buffers once the work already issued on its chain completes. Idempotent. */
	@Override
	public void close() {
		if (closed)
			return;
		chain.detach(this);
	}

	// ── For the operations that run on resident activations ──────────────────

	/** The device buffer, valid until {@link #close}. Never dereferenced on the host. */
	MemorySegment devicePointer() {
		requireOpen();
		return device;
	}

	/** Records that a device operation wrote {@code count} valid rows into this activation. */
	void markWritten(int count) {
		if (count <= 0 || count > capacityRows)
			throw new IllegalArgumentException(count + " rows written into an activation of capacity " + capacityRows);
		rows = count;
	}

	void requireOpen() {
		if (closed)
			throw new IllegalStateException("resident activation is closed");
	}

	/** Releases the buffers. Called by the chain, after it has synchronized its stream. */
	void free() {
		if (closed)
			return;
		closed = true;
		rows = 0;
		GpuBindings gpu = chain.bindings();
		gpu.deviceFree(device);
		gpu.hostFree(staging);
	}

	private void copyAsync(MemorySegment dst, MemorySegment src, long n, int kind, String what) {
		int rc;
		try {
			rc = (int) chain.bindings().gpuMemcpyAsync().invokeExact(dst, src, n, kind, chain.stream());
		} catch (Throwable t) {
			throw new IllegalStateException(what + ": native call failed", t);
		}
		GpuBindings.check(rc, what);
	}
}
