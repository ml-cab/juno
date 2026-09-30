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

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;

import jdk.jfr.FlightRecorder;

/**
 * Running totals behind {@link DeviceStagingEvent} and {@link WeightDequantEvent}.
 *
 * <p>A prefill window issues tens of thousands of copies (every KV row of every
 * layer is its own copy) and a decoded token several hundred, so one JFR event per
 * copy costs a measurable share of the work it describes. Instead each copy adds to
 * lock-free counters keyed by its site and phase, and both events are periodic:
 * JFR calls {@link #emitStaging} and {@link #emitDequant} at the end of every chunk,
 * which commit one event per non-empty cell carrying the totals since the previous
 * emission. Starting a recording ends the chunk before it, so the totals a
 * recording holds are exactly the work done while it ran.
 *
 * <p>Nothing is counted while no recording asks for the events: every entry point
 * checks the event's enabled state first, which is a field read once compiled.
 */
final class DeviceSpanTally {

	static final int PREFILL = 0;
	static final int DECODE = 1;
	static final int OTHER = 2;
	private static final String[] PHASES = { "prefill", "decode", "other" };

	private static final ConcurrentHashMap<String, StagingCell> STAGING = new ConcurrentHashMap<>();
	private static final ConcurrentHashMap<String, DequantCell> DEQUANT = new ConcurrentHashMap<>();

	static {
		if (FlightRecorder.isAvailable()) {
			FlightRecorder.addPeriodicEvent(DeviceStagingEvent.class, DeviceSpanTally::emitStaging);
			FlightRecorder.addPeriodicEvent(WeightDequantEvent.class, DeviceSpanTally::emitDequant);
		}
	}

	private DeviceSpanTally() {
	}

	/** Whether a running recording wants {@code juno.DeviceStaging}. */
	static boolean stagingWanted() {
		return new DeviceStagingEvent().isEnabled();
	}

	/** Whether a running recording wants {@code juno.WeightDequant}. */
	static boolean dequantWanted() {
		return new WeightDequantEvent().isEnabled();
	}

	/** The phase of a copy issued by a forward call over {@code windowSize} rows (0: outside a forward call). */
	static int phase(int windowSize) {
		return windowSize > 1 ? PREFILL : windowSize == 1 ? DECODE : OTHER;
	}

	/**
	 * Adds one copy.
	 *
	 * @param nanos measured transfer time, or {@code -1} for a copy that was counted but not timed
	 */
	static void staging(String site, int memcpyKind, int windowSize, long bytes, long nanos) {
		if (!stagingWanted())
			return;
		StagingCell cell = STAGING.get(site);
		if (cell == null)
			cell = STAGING.computeIfAbsent(site, s -> new StagingCell(s, DeviceStagingEvent.direction(memcpyKind)));
		int p = phase(windowSize);
		cell.copies[p].increment();
		cell.bytes[p].add(bytes);
		if (nanos >= 0) {
			cell.timed[p].increment();
			cell.nanos[p].add(nanos);
		}
	}

	/**
	 * Adds one dequantization.
	 *
	 * @param nanos measured duration, or {@code -1} if it could not be timed
	 */
	static void dequant(int ggufType, String timing, long nanos) {
		if (!dequantWanted())
			return;
		String format = WeightDequantEvent.format(ggufType);
		String key = DeviceStagingEvent.TIMING_DEVICE.equals(timing) ? format : format + "/" + timing;
		DequantCell cell = DEQUANT.get(key);
		if (cell == null)
			cell = DEQUANT.computeIfAbsent(key, k -> new DequantCell(format, timing));
		cell.count.increment();
		if (nanos >= 0) {
			cell.timed.increment();
			cell.nanos.add(nanos);
		}
	}

	/** Periodic hook: commits one event per non-empty site and phase, and starts the next totals. */
	static void emitStaging() {
		for (StagingCell cell : STAGING.values()) {
			for (int p = 0; p < PHASES.length; p++) {
				long copies = cell.copies[p].sumThenReset();
				long bytes = cell.bytes[p].sumThenReset();
				long timed = cell.timed[p].sumThenReset();
				long nanos = cell.nanos[p].sumThenReset();
				if (copies == 0)
					continue;
				DeviceStagingEvent ev = new DeviceStagingEvent();
				ev.site = cell.site;
				ev.direction = cell.direction;
				ev.phase = PHASES[p];
				ev.copies = copies;
				ev.bytes = bytes;
				ev.timedCopies = timed;
				ev.transferNanos = nanos;
				ev.commit();
			}
		}
	}

	/** Periodic hook: commits one event per non-empty format and timing source. */
	static void emitDequant() {
		for (DequantCell cell : DEQUANT.values()) {
			long count = cell.count.sumThenReset();
			long timed = cell.timed.sumThenReset();
			long nanos = cell.nanos.sumThenReset();
			if (count == 0)
				continue;
			WeightDequantEvent ev = new WeightDequantEvent();
			ev.format = cell.format;
			ev.timing = cell.timing;
			ev.count = count;
			ev.timedCount = timed;
			ev.dequantNanos = nanos;
			ev.commit();
		}
	}

	private static final class StagingCell {
		final String site;
		final String direction;
		final LongAdder[] copies = adders();
		final LongAdder[] bytes = adders();
		final LongAdder[] timed = adders();
		final LongAdder[] nanos = adders();

		StagingCell(String site, String direction) {
			this.site = site;
			this.direction = direction;
		}

		private static LongAdder[] adders() {
			LongAdder[] a = new LongAdder[PHASES.length];
			for (int i = 0; i < a.length; i++)
				a[i] = new LongAdder();
			return a;
		}
	}

	private static final class DequantCell {
		final String format;
		final String timing;
		final LongAdder count = new LongAdder();
		final LongAdder timed = new LongAdder();
		final LongAdder nanos = new LongAdder();

		DequantCell(String format, String timing) {
			this.format = format;
			this.timing = timing;
		}
	}
}
