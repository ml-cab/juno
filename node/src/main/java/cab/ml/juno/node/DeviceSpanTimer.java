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
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;

/**
 * Device-side timing for asynchronous copies and dequantization kernels queued on
 * one stream, counted into {@link DeviceSpanTally}.
 *
 * <p>An asynchronous operation returns as soon as it is queued, so a host clock
 * around it measures the enqueue. The caller brackets the operation with
 * {@link #begin} and {@link #staging} (or {@link #dequant}), which record a stream
 * event on each side, and after the stream has synchronized calls {@link #commit},
 * which reads each span's elapsed time off the device and adds it to the tally.
 *
 * <p>Only work issued by a forward call over more than one row is timed. A
 * decode-width copy is counted with its bytes and no stream event is recorded:
 * decode issues several hundred copies per token, and the events would cost a
 * visible share of it. Without a recording nothing is recorded or counted.
 * Stream events are created lazily and reused; each mark within one set (between
 * two commits) uses its own event, because re-recording an event before the stream
 * reaches it would move the earlier mark.
 *
 * <p>Not thread-safe: an owner uses it where its stream is already serialized
 * (the matmul backends under the context's serialization lock, a resident chain
 * from its single caller). A timing failure never fails the operation it times;
 * the span is then counted untimed.
 */
final class DeviceSpanTimer implements AutoCloseable {

	private static final int KIND_STAGING = 0;
	private static final int KIND_DEQUANT = 1;

	private final GpuBindings gpu;
	private final Arena arena = Arena.ofShared();
	private final MemorySegment eventOut;
	private final MemorySegment elapsedOut;

	private MemorySegment[] events = new MemorySegment[0];
	private int marks;
	private boolean closed;

	private int pending;
	private int[] kind = new int[8];
	private int[] from = new int[8];
	private int[] to = new int[8];
	/** Staging: memcpy kind. Dequant: GGUF type. */
	private int[] code = new int[8];
	private int[] window = new int[8];
	private long[] bytes = new long[8];
	private String[] site = new String[8];

	DeviceSpanTimer(GpuBindings gpu) {
		this.gpu = gpu;
		this.eventOut = arena.allocate(ADDRESS);
		this.elapsedOut = arena.allocate(JAVA_FLOAT);
	}

	/**
	 * Opens a span before an operation issued by a forward call over {@code windowSize}
	 * rows. Returns the stream event's index, or {@code -1} when the operation is not
	 * timed (decode width, no recording, or the event could not be recorded).
	 */
	int begin(MemorySegment stream, int windowSize) {
		if (windowSize == 1 || closed)
			return -1;
		if (!DeviceSpanTally.stagingWanted() && !DeviceSpanTally.dequantWanted())
			return -1;
		return mark(stream);
	}

	/** Closes a copy's span, or counts it untimed if {@code beginMark} is {@code -1}. */
	void staging(int memcpyKind, long n, int windowSize, String copySite, int beginMark, MemorySegment stream) {
		int endMark = beginMark < 0 ? -1 : mark(stream);
		if (endMark < 0) {
			DeviceSpanTally.staging(copySite, memcpyKind, windowSize, n, -1L);
			return;
		}
		int i = nextPending();
		kind[i] = KIND_STAGING;
		code[i] = memcpyKind;
		bytes[i] = n;
		window[i] = windowSize;
		site[i] = copySite;
		from[i] = beginMark;
		to[i] = endMark;
	}

	/** Closes a device dequantization's span, or counts it untimed if {@code beginMark} is {@code -1}. */
	void dequant(int ggufType, int beginMark, MemorySegment stream) {
		int endMark = beginMark < 0 ? -1 : mark(stream);
		if (endMark < 0) {
			DeviceSpanTally.dequant(ggufType, DeviceStagingEvent.TIMING_DEVICE, -1L);
			return;
		}
		int i = nextPending();
		kind[i] = KIND_DEQUANT;
		code[i] = ggufType;
		site[i] = null;
		from[i] = beginMark;
		to[i] = endMark;
	}

	/**
	 * Adds every open span to the tally and starts a new set. Call only after the
	 * stream the marks were recorded on has synchronized; a span the device has not
	 * reached is counted untimed rather than waited for.
	 */
	void commit() {
		for (int i = 0; i < pending; i++) {
			long nanos = elapsedNanos(from[i], to[i]);
			if (kind[i] == KIND_STAGING)
				DeviceSpanTally.staging(site[i], code[i], window[i], bytes[i], nanos);
			else
				DeviceSpanTally.dequant(code[i], DeviceStagingEvent.TIMING_DEVICE, nanos);
		}
		reset();
	}

	/** Discards the open spans without counting them, e.g. when the operation failed. */
	void reset() {
		marks = 0;
		pending = 0;
	}

	/**
	 * Destroys the pooled stream events; the next timed span creates them again. For
	 * an owner that releases its device state but stays usable.
	 */
	void releaseEvents() {
		reset();
		for (MemorySegment e : events) {
			try {
				int rc = (int) gpu.gpuEventDestroy().invokeExact(e);
			} catch (Throwable t) {
				// best effort: an event that cannot be destroyed is left to the context teardown
			}
		}
		events = new MemorySegment[0];
	}

	@Override
	public void close() {
		if (closed)
			return;
		closed = true;
		releaseEvents();
		arena.close();
	}

	private int mark(MemorySegment stream) {
		if (marks == events.length && !grow())
			return -1;
		int rc;
		try {
			rc = (int) gpu.gpuEventRecord().invokeExact(events[marks], stream);
		} catch (Throwable t) {
			return -1;
		}
		return rc == 0 ? marks++ : -1;
	}

	private long elapsedNanos(int a, int b) {
		int rc;
		try {
			rc = (int) gpu.gpuEventElapsedTime().invokeExact(elapsedOut, events[a], events[b]);
		} catch (Throwable t) {
			return -1;
		}
		if (rc != 0)
			return -1;
		return Math.round(elapsedOut.get(JAVA_FLOAT, 0) * 1_000_000.0);
	}

	private int nextPending() {
		if (pending == kind.length) {
			int n = pending * 2;
			kind = Arrays.copyOf(kind, n);
			from = Arrays.copyOf(from, n);
			to = Arrays.copyOf(to, n);
			code = Arrays.copyOf(code, n);
			window = Arrays.copyOf(window, n);
			bytes = Arrays.copyOf(bytes, n);
			site = Arrays.copyOf(site, n);
		}
		return pending++;
	}

	/** Adds stream events to the pool; returns false if none could be added. */
	private boolean grow() {
		int n = Math.max(8, events.length * 2);
		MemorySegment[] grown = Arrays.copyOf(events, n);
		int created = events.length;
		for (; created < n; created++) {
			int rc;
			try {
				rc = (int) gpu.gpuEventCreate().invokeExact(eventOut);
			} catch (Throwable t) {
				rc = -1;
			}
			if (rc != 0)
				break;
			grown[created] = eventOut.get(ADDRESS, 0);
		}
		boolean added = created > events.length;
		events = Arrays.copyOf(grown, created);
		return added;
	}
}
