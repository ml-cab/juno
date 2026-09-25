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

import static org.assertj.core.api.Assertions.assertThat;

import java.time.Instant;

import org.junit.jupiter.api.Test;

/**
 * Which collection pauses actually fell inside the window a throughput figure was
 * derived from.
 *
 * <p>A recording covers more than the measurement: model load, prefill, and the
 * tail after the last token all sit inside it. A pause landing in any of those
 * cannot have slowed the token-to-token rate, so judging a run by the largest
 * pause anywhere in its recording rejects results that were never contaminated.
 * These cases pin down what "inside" means, including the two half-overlaps, which
 * are the ones a naive containment test gets wrong.
 */
class GcPauseSpansTest {

	private static final Instant T0 = Instant.parse("2026-09-25T12:00:00Z");

	private static Instant at(long millis) {
		return T0.plusMillis(millis);
	}

	private static long ms(long millis) {
		return millis * 1_000_000L;
	}

	@Test
	void countsOnlyThePausesOverlappingTheWindow() {
		GcPauseSpans spans = new GcPauseSpans();
		spans.add(at(0), ms(500)); // ends at 500, before the window
		spans.add(at(1200), ms(10)); // inside
		spans.add(at(1800), ms(20)); // inside
		spans.add(at(3000), ms(50)); // starts after the window

		GcPauseSpans.Stats in = spans.overlapping(at(1000), at(2000));

		assertThat(in.count()).isEqualTo(2);
		assertThat(in.maxMs()).isEqualTo(20.0);
		assertThat(in.totalMs()).isEqualTo(30.0);
	}

	@Test
	void aPauseStraddlingTheWindowStartCounts() {
		GcPauseSpans spans = new GcPauseSpans();
		spans.add(at(900), ms(300)); // 900 to 1200, crosses into the window

		GcPauseSpans.Stats in = spans.overlapping(at(1000), at(2000));

		assertThat(in.count()).as("a pause already running when the window opened stole time from it")
				.isEqualTo(1);
		assertThat(in.maxMs()).isEqualTo(300.0);
	}

	@Test
	void aPauseStraddlingTheWindowEndCounts() {
		GcPauseSpans spans = new GcPauseSpans();
		spans.add(at(1950), ms(200)); // 1950 to 2150, crosses out of the window

		assertThat(spans.overlapping(at(1000), at(2000)).count()).isEqualTo(1);
	}

	@Test
	void aPauseEndingExactlyAtTheWindowStartDoesNotCount() {
		GcPauseSpans spans = new GcPauseSpans();
		spans.add(at(500), ms(500)); // ends at exactly 1000

		assertThat(spans.overlapping(at(1000), at(2000)).count()).isZero();
	}

	@Test
	void aPauseBeginningExactlyAtTheWindowEndDoesNotCount() {
		GcPauseSpans spans = new GcPauseSpans();
		spans.add(at(2000), ms(100));

		assertThat(spans.overlapping(at(1000), at(2000)).count()).isZero();
	}

	@Test
	void theLongestPauseOutsideTheWindowIsIgnoredEvenWhenItDwarfsTheOnesInside() {
		// This is the case that motivated the class: a 635 ms pause during model
		// load alongside a 5 ms pause inside the measured span.
		GcPauseSpans spans = new GcPauseSpans();
		spans.add(at(0), ms(635));
		spans.add(at(1500), ms(5));

		GcPauseSpans.Stats in = spans.overlapping(at(1000), at(2000));

		assertThat(in.maxMs()).isEqualTo(5.0);
		assertThat(in.count()).isEqualTo(1);
	}

	@Test
	void noPausesYieldsZeroesRatherThanNothing() {
		GcPauseSpans.Stats in = new GcPauseSpans().overlapping(at(1000), at(2000));

		assertThat(in.count()).isZero();
		assertThat(in.maxMs()).isZero();
		assertThat(in.totalMs()).isZero();
	}

	@Test
	void anUnknownWindowReportsEveryPauseRatherThanNone() {
		GcPauseSpans spans = new GcPauseSpans();
		spans.add(at(0), ms(635));
		spans.add(at(1500), ms(5));

		// A recording with fewer than two tokens has no span to speak of. Reporting
		// nothing would read as a clean run; reporting everything is the safe answer.
		GcPauseSpans.Stats in = spans.overlapping(null, null);

		assertThat(in.count()).isEqualTo(2);
		assertThat(in.maxMs()).isEqualTo(635.0);
	}

	@Test
	void keepsABoundedNumberOfPausesAndSaysWhenItTruncated() {
		GcPauseSpans spans = new GcPauseSpans(3);
		for (int i = 0; i < 10; i++)
			spans.add(at(1000 + i), ms(1));

		assertThat(spans.truncated()).as("a long recording must not grow this without limit").isTrue();
		assertThat(spans.overlapping(at(1000), at(2000)).count()).isEqualTo(3);
	}
}
