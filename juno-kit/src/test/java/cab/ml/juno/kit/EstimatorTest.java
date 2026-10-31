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

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.junit.jupiter.api.Test;

class EstimatorTest {

	private static Probe probe(String id, int cost) {
		return new Probe(id, id, ProbeKind.READING, Set.of("quick"), Requirements.cpu(0), List.of(), cost, 1,
				"EnvProbe", Map.of(), 600, "1g");
	}

	@Test
	void scaleIsTheCalibrationReadingOverTheReferenceHostsAndIsClamped() {
		assertThat(Estimator.scale(150, 100)).isCloseTo(1.5, within(1e-9));
		assertThat(Estimator.scale(1, 100)).isEqualTo(Estimator.MIN_SCALE);
		assertThat(Estimator.scale(100_000, 100)).isEqualTo(Estimator.MAX_SCALE);
		assertThat(Estimator.scale(0, 100)).isEqualTo(1.0);
	}

	@Test
	void estimateIsTheScaledReferenceCostsPlusDownloadTimeAsARange() {
		Estimator.Range r = Estimator.estimate(List.of(probe("a", 100), probe("b", 200)), 1.5, 1_000_000_000L,
				100_000_000.0);
		assertThat(r.point()).isEqualTo(Duration.ofSeconds(460));
		assertThat(r.low().getSeconds()).isEqualTo(Math.round(460 * Estimator.LOW));
		assertThat(r.high().getSeconds()).isEqualTo(Math.round(460 * Estimator.HIGH));
	}

	@Test
	void noDownloadRateMeansNoDownloadTermRatherThanADivisionByZero() {
		Estimator.Range r = Estimator.estimate(List.of(probe("a", 100)), 1.0, 5_000_000L, 0);
		assertThat(r.point()).isEqualTo(Duration.ofSeconds(100));
	}

	@Test
	void remainingTimeFollowsTheObservedPaceOnceEnoughHasRun() {
		Estimator.Range r = Estimator.remaining(List.of(probe("done", 100)), List.of(Duration.ofSeconds(200)),
				List.of(probe("left", 300)), 1.0);
		assertThat(r.point()).isEqualTo(Duration.ofSeconds(600));
	}

	@Test
	void remainingTimeKeepsTheCalibratedScaleWhileTooLittleHasRun() {
		Estimator.Range r = Estimator.remaining(List.of(probe("done", 30)), List.of(Duration.ofSeconds(90)),
				List.of(probe("left", 300)), 1.0);
		assertThat(r.point()).isEqualTo(Duration.ofSeconds(300));
	}
}
