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

/**
 * Synchronous host-device copies and host-side dequantization, counted into
 * {@link DeviceSpanTally} when a recording asks for {@link DeviceStagingEvent} or
 * {@link WeightDequantEvent}.
 *
 * <p>With no recording the copy is exactly the plain {@code gpuMemcpy} it replaces.
 * With one, a copy outside the decode phase is timed on the host. A device-to-host
 * copy reads back a kernel's result, so the default stream is drained first and
 * the figure is the transfer rather than a wait for that kernel; the host was going
 * to wait for it anyway, so the drain moves the wait out of the figure, not into the
 * request. A host-to-device copy is timed as it stands.
 *
 * <p>Copies smaller than {@link #SAMPLE_BELOW_BYTES} are timed one in
 * {@link #SAMPLE_EVERY} and counted every time. A prefill window writes every KV
 * row of every layer as its own few-hundred-byte copy (22,528 of them for a
 * 512-token TinyLlama window), and two clock reads per copy cost about 5% of that
 * window on a host whose kernel rejected the CPU timestamp counter, where a clock
 * read is a system call. Bytes and copies stay exact; the duration of those sites
 * is estimated from the sampled mean ({@code timedCopies} says how many were timed).
 */
final class DeviceStaging {

	/** Copies below this size are timed one in {@link #SAMPLE_EVERY}. */
	static final long SAMPLE_BELOW_BYTES = 64 * 1024;
	static final int SAMPLE_EVERY = 16;

	/** Per-thread position in the sampling cycle; a thread issues its own copies in order. */
	private static final ThreadLocal<int[]> SAMPLE_TICK = ThreadLocal.withInitial(() -> new int[1]);

	private DeviceStaging() {
	}

	/**
	 * {@code gpuMemcpy(dst, src, bytes, kind)}, throwing with {@code site} on failure.
	 *
	 * @param windowSize rows of the forward call that issued the copy: more than 1 in prefill,
	 *                   1 in decode, 0 outside a forward call
	 */
	static void copy(GpuBindings gpu, MemorySegment dst, MemorySegment src, long bytes, int kind,
			int windowSize, String site) {
		if (!DeviceSpanTally.stagingWanted()) {
			GpuBindings.check(GpuBindings.callInt(gpu.gpuMemcpy(), dst, src, bytes, kind), site);
			return;
		}
		if (windowSize == 1 || (bytes < SAMPLE_BELOW_BYTES && ++SAMPLE_TICK.get()[0] % SAMPLE_EVERY != 0)) {
			GpuBindings.check(GpuBindings.callInt(gpu.gpuMemcpy(), dst, src, bytes, kind), site);
			DeviceSpanTally.staging(site, kind, windowSize, bytes, -1L);
			return;
		}
		if (kind == GpuBindings.D2H)
			GpuBindings.check(GpuBindings.callInt(gpu.gpuStreamSynchronize(), MemorySegment.NULL),
					"streamSynchronize(before " + site + ")");
		long t0 = System.nanoTime();
		GpuBindings.check(GpuBindings.callInt(gpu.gpuMemcpy(), dst, src, bytes, kind), site);
		DeviceSpanTally.staging(site, kind, windowSize, bytes, System.nanoTime() - t0);
	}

	/** Returns a start stamp for {@link #hostDequantDone}, or {@code -1} when no recording wants the event. */
	static long hostDequantStart() {
		return DeviceSpanTally.dequantWanted() ? System.nanoTime() : -1L;
	}

	/** Counts a host-side dequantization started at {@code start}; a no-op when {@code start < 0}. */
	static void hostDequantDone(long start, int ggufType) {
		if (start < 0)
			return;
		DeviceSpanTally.dequant(ggufType, DeviceStagingEvent.TIMING_HOST, System.nanoTime() - start);
	}
}
