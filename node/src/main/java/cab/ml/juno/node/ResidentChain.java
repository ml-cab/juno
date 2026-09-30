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
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static java.lang.foreign.ValueLayout.ADDRESS;

/**
 * A residency region: one device stream and the {@link ResidentActivation}
 * buffers that stay on the device across consecutive operations on it.
 *
 * <p>Every operation on an activation of this chain is issued on the chain's
 * stream, so operations run in the order they were issued without the host
 * waiting between them. The host waits only at the region's exit: a
 * {@link ResidentActivation#materialize} (the one point device results reach
 * host memory) or an explicit {@link #sync}. That is the whole mechanism - the
 * per-operation host round trip of the {@link MatVec} and {@link CudaRmsNorm}
 * round-trip paths is replaced by one upload where data enters the region and
 * one download where it leaves.
 *
 * <p>Ownership is by construction: {@link #close} synchronizes the stream,
 * frees every activation still open on it and destroys the stream, so a region
 * cannot leak device memory by forgetting a buffer. An activation may also be
 * closed on its own earlier.
 *
 * <p>Vendor-neutral: buffers and the stream go through {@link GpuBindings}. The
 * operations that run on a chain today ({@link CudaRmsNorm#normalizeResident},
 * {@link CudaRope#applyResident}) are CUDA kernels.
 *
 * <p>Not thread-safe. A chain and its activations belong to one thread at a
 * time, like the single decode stream that issues one operation after another.
 */
final class ResidentChain implements AutoCloseable {

	private final GpuContext ctx;
	private final GpuBindings gpu;
	private final MemorySegment stream;
	private final List<ResidentActivation> activations = new ArrayList<>();
	/** Raw device scratch owned by this chain (not activations), freed on {@link #close}. */
	private final List<MemorySegment> scratch = new ArrayList<>();
	/** Device-side timing of the copies issued on this chain, committed at each synchronization. */
	private final DeviceSpanTimer spans;

	/**
	 * Incremented at every synchronization. An activation records it when it
	 * starts an upload, so it can tell whether that transfer has landed before
	 * overwriting the pinned buffer the transfer reads from.
	 */
	private long syncEpoch;
	private boolean closed;

	private ResidentChain(GpuContext ctx, MemorySegment stream) {
		this.ctx = ctx;
		this.gpu = ctx.bindings();
		this.stream = stream;
		this.spans = new DeviceSpanTimer(gpu);
	}

	/** Opens a region on {@code ctx}'s device with a stream of its own. */
	static ResidentChain open(GpuContext ctx) {
		Objects.requireNonNull(ctx, "ctx");
		GpuBindings gpu = ctx.bindings();
		GpuBindings.check(GpuBindings.callInt(gpu.gpuSetDevice(), ctx.deviceIndex()), "setDevice(resident chain)");
		try (Arena tmp = Arena.ofConfined()) {
			MemorySegment slot = tmp.allocate(ADDRESS);
			GpuBindings.check(
					GpuBindings.callInt(gpu.gpuStreamCreateWithFlags(), slot, GpuBindings.STREAM_NON_BLOCKING),
					"streamCreate(resident chain)");
			return new ResidentChain(ctx, slot.get(ADDRESS, 0));
		}
	}

	/** Allocates a device buffer of {@code capacityRows x dim} floats on this chain. */
	ResidentActivation allocate(int capacityRows, int dim) {
		requireOpen();
		ResidentActivation a = ResidentActivation.allocate(this, capacityRows, dim);
		activations.add(a);
		return a;
	}

	/**
	 * Allocates {@code bytes} of raw device memory owned by this chain, for an
	 * operation's intermediate that is not an activation (a quantized copy of an
	 * input, say). Ordered by this chain's stream like everything else on it, so a
	 * kernel on another stream never overwrites it mid-use. Freed on {@link #close}.
	 */
	MemorySegment allocateScratch(long bytes) {
		requireOpen();
		if (bytes <= 0)
			throw new IllegalArgumentException("scratch bytes must be positive: " + bytes);
		MemorySegment s = gpu.deviceMalloc(ctx.deviceIndex(), bytes);
		scratch.add(s);
		return s.reinterpret(bytes);
	}

	/** Blocks until every operation issued on this chain has completed. */
	void sync() {
		requireOpen();
		synchronizeStream();
	}

	/** Device bytes held by the activations still open on this chain. */
	long deviceBytes() {
		long total = 0;
		for (ResidentActivation a : activations)
			total += a.deviceBytes();
		for (MemorySegment s : scratch)
			total += s.byteSize();
		return total;
	}

	boolean isClosed() {
		return closed;
	}

	@Override
	public void close() {
		if (closed)
			return;
		try {
			// Nothing issued on the stream may still be reading or writing a buffer when it is freed.
			synchronizeStream();
		} finally {
			for (ResidentActivation a : activations)
				a.free();
			activations.clear();
			for (MemorySegment s : scratch)
				gpu.deviceFree(s);
			scratch.clear();
			spans.close();
			GpuBindings.callInt(gpu.gpuStreamDestroy(), stream);
			closed = true;
		}
	}

	MemorySegment stream() {
		return stream;
	}

	GpuContext context() {
		return ctx;
	}

	GpuBindings bindings() {
		return gpu;
	}

	DeviceSpanTimer spans() {
		return spans;
	}

	long syncEpoch() {
		return syncEpoch;
	}

	/** Frees one activation ahead of the chain, after the work already issued on it completes. */
	void detach(ResidentActivation a) {
		try {
			synchronizeStream();
		} finally {
			activations.remove(a);
			a.free();
		}
	}

	private void synchronizeStream() {
		int rc;
		try {
			rc = (int) gpu.gpuStreamSynchronize().invokeExact(stream);
		} catch (Throwable t) {
			throw new IllegalStateException("streamSynchronize(resident chain): native call failed", t);
		}
		GpuBindings.check(rc, "streamSynchronize(resident chain)");
		syncEpoch++;
		spans.commit();
	}

	private void requireOpen() {
		if (closed)
			throw new IllegalStateException("resident chain is closed");
	}
}
