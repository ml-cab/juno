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

package cab.ml.juno.metrics;

import java.time.Duration;
import java.time.Instant;

/**
 * When each collection pause happened, so a pause can be attributed to the window
 * a throughput figure was actually derived from.
 *
 * <p>A recording covers more than the measurement. Model load, prefill and the
 * tail after the last token all sit inside it, and a pause landing in any of those
 * cannot have slowed the token-to-token rate. Judging a run by the largest pause
 * anywhere in its recording therefore rejects results that were never
 * contaminated: a real sweep produced three such rejections, each one a pause of
 * about 635 ms on a reading that agreed with its repetitions to within 1%.
 *
 * <p>A pause counts as inside the window when it overlaps it at all, not only when
 * it is contained by it. A collection already running when the first token was
 * produced took time from that measurement just as surely as one starting in the
 * middle of it.
 *
 * <p>Pause intervals are the one thing here that has to be remembered rather than
 * folded into a running total, because the window is not known until the whole
 * recording has been read. The list is bounded so a long recording cannot grow it
 * without limit, and says when it dropped anything.
 *
 * @author Yevhen Soldatov
 */
final class GcPauseSpans {

	/**
	 * Enough for any measurement window this project publishes a ratio from, and
	 * about 300 KB if it ever fills.
	 */
	private static final int DEFAULT_CAPACITY = 20_000;

	private static final double NANOS_PER_MS = 1_000_000.0;

	private final int capacity;
	private long[] startNanos;
	private long[] durationNanos;
	private int size;
	private boolean truncated;

	GcPauseSpans() {
		this(DEFAULT_CAPACITY);
	}

	GcPauseSpans(int capacity) {
		this.capacity = Math.max(1, capacity);
		int initial = Math.min(this.capacity, 256);
		this.startNanos = new long[initial];
		this.durationNanos = new long[initial];
	}

	/** Records one pause. Beyond capacity the pause is dropped and noted. */
	void add(Instant start, long nanos) {
		if (start == null)
			return;
		if (size == capacity) {
			truncated = true;
			return;
		}
		if (size == startNanos.length)
			grow();
		startNanos[size] = toNanos(start);
		durationNanos[size] = nanos;
		size++;
	}

	/** Whether any pause was dropped for want of room. */
	boolean truncated() {
		return truncated;
	}

	/**
	 * Count, maximum and total of the pauses overlapping {@code [from, to)}.
	 *
	 * <p>A null bound means the window is unknown — a recording with fewer than two
	 * tokens has no span to speak of — and every pause is reported. Reporting none
	 * would read as a clean run, which is the one answer that must not be given by
	 * default.
	 */
	Stats overlapping(Instant from, Instant to) {
		boolean wholeRecording = from == null || to == null;
		long fromNanos = wholeRecording ? 0 : toNanos(from);
		long toNanos = wholeRecording ? 0 : toNanos(to);

		int count = 0;
		long max = 0;
		long total = 0;
		for (int i = 0; i < size; i++) {
			if (!wholeRecording) {
				long pauseStart = startNanos[i];
				long pauseEnd = pauseStart + durationNanos[i];
				// Half-open on both sides: a pause ending exactly as the window opens,
				// or starting exactly as it closes, overlaps nothing.
				if (pauseEnd <= fromNanos || pauseStart >= toNanos)
					continue;
			}
			count++;
			total += durationNanos[i];
			if (durationNanos[i] > max)
				max = durationNanos[i];
		}
		return new Stats(count, max / NANOS_PER_MS, total / NANOS_PER_MS);
	}

	private void grow() {
		int next = Math.min(capacity, Math.max(startNanos.length * 2, 256));
		long[] s = new long[next];
		long[] d = new long[next];
		System.arraycopy(startNanos, 0, s, 0, size);
		System.arraycopy(durationNanos, 0, d, 0, size);
		startNanos = s;
		durationNanos = d;
	}

	private static long toNanos(Instant t) {
		return t.getEpochSecond() * 1_000_000_000L + t.getNano();
	}

	/** Pauses attributed to one window, in milliseconds. */
	record Stats(int count, double maxMs, double totalMs) {

		/** The share of {@code window} spent paused, or 0 when the window is empty. */
		double fractionOf(Duration window) {
			if (window == null || window.isZero() || window.isNegative())
				return 0.0;
			return totalMs / (window.toNanos() / NANOS_PER_MS);
		}
	}
}
