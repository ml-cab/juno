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
package cab.ml.juno.kit;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class NoiseRuleTest {

	private static ProbeReport reading(double... values) {
		List<Double> v = java.util.Arrays.stream(values).boxed().toList();
		return new ProbeReport("r", ProbeKind.READING, null, List.of(),
				List.of(new ProbeReport.Reading("tg", "t/s", v, true, true)), Map.of());
	}

	@Test
	void spreadIsTheRangeOverTheMedian() {
		assertThat(NoiseRule.spread(List.of(100.0, 110.0, 90.0))).isCloseTo(0.20, within(1e-9));
		assertThat(NoiseRule.spread(List.of(42.0))).isZero();
		assertThat(NoiseRule.spread(List.of())).isZero();
	}

	@Test
	void aRowIsNoisyOnlyWhenItsRepetitionsDisagreeByMoreThanFifteenPercent() {
		assertThat(NoiseRule.noisy(List.of(100.0, 110.0, 90.0))).isTrue();
		assertThat(NoiseRule.noisy(List.of(100.0, 105.0, 95.0))).isFalse();
		assertThat(NoiseRule.noisy(List.of(100.0, 115.0, 100.0))).isFalse(); // exactly 15%: not more than
	}

	@Test
	void aNoisyReadingIsReRunTwiceAndThenMarked() {
		ProbeReport noisy = reading(100, 130, 90);
		assertThat(NoiseRule.verdict(noisy, 0.0, false, 1)).isEqualTo(NoiseRule.Verdict.RERUN);
		assertThat(NoiseRule.verdict(noisy, 0.0, false, 2)).isEqualTo(NoiseRule.Verdict.RERUN);
		assertThat(NoiseRule.verdict(noisy, 0.0, false, 3)).isEqualTo(NoiseRule.Verdict.NOISY);
	}

	@Test
	void aCleanReadingTakenUnderLoadIsReRunToo() {
		ProbeReport clean = reading(100, 101, 99);
		assertThat(NoiseRule.verdict(clean, 0.0, false, 1)).isEqualTo(NoiseRule.Verdict.OK);
		assertThat(NoiseRule.verdict(clean, 0.20, false, 1)).isEqualTo(NoiseRule.Verdict.RERUN);
		assertThat(NoiseRule.verdict(clean, 0.20, false, 3)).isEqualTo(NoiseRule.Verdict.NOISY);
	}

	@Test
	void aReadingTakenWhileAnotherProgramUsedTheGpuIsReRunToo() {
		ProbeReport clean = reading(100, 101, 99);
		assertThat(NoiseRule.verdict(clean, 0.0, true, 1)).isEqualTo(NoiseRule.Verdict.RERUN);
		assertThat(NoiseRule.verdict(clean, 0.0, true, 3)).isEqualTo(NoiseRule.Verdict.NOISY);
		assertThat(NoiseRule.reason(clean, 0.0, true)).contains("GPU");
	}

	@Test
	void anUnscoredReadingNeverTriggersAReRun() {
		ProbeReport gc = new ProbeReport("r", ProbeKind.READING, null, List.of(),
				List.of(new ProbeReport.Reading("gc_pause_ms", "ms", List.of(0.0, 4.0, 1.0), false, false)), Map.of());
		assertThat(NoiseRule.verdict(gc, 0.0, false, 1)).isEqualTo(NoiseRule.Verdict.OK);
	}

	@Test
	void checksAreNeverReRunForNoise() {
		ProbeReport check = new ProbeReport("c", ProbeKind.CHECK, true,
				List.of(new ProbeReport.Check("x", true, "")), List.of(), Map.of());
		assertThat(NoiseRule.verdict(check, 0.9, true, 1)).isEqualTo(NoiseRule.Verdict.OK);
	}
}
