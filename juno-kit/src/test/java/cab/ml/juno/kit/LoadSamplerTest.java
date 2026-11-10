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

import java.util.Map;

import org.junit.jupiter.api.Test;

/**
 * Other programs' CPU in one interval between samples: the system's CPU minus what the probe's own processes used,
 * each process's CPU time read at both ends. A child that exits between samples has used CPU the system load
 * holds and no sample of the child does.
 */
class LoadSamplerTest {

	private static final int CPUS = 16;

	@Test
	void anIntervalInWhichOneOfTheProbesProcessesExitedIsLeftOut() {
		// pid 2 used 8 more CPU-seconds and exited before the second sample
		Map<Long, Double> before = Map.of(1L, 10.0, 2L, 5.0);
		Map<Long, Double> now = Map.of(1L, 10.1);
		assertThat(LoadSampler.otherCpuSeconds((8.0 + 0.1) / CPUS, CPUS, 1.0, before, now)).isEmpty();
	}

	@Test
	void aChildStartedDuringTheIntervalIsTheProbesOwnFromItsFirstSample() {
		Map<Long, Double> before = Map.of(1L, 10.0);
		Map<Long, Double> now = Map.of(1L, 10.1, 3L, 3.0);
		assertThat(LoadSampler.otherCpuSeconds(3.1 / CPUS, CPUS, 1.0, before, now)).hasValueCloseTo(0.0,
				within(1e-9));
	}

	@Test
	void cpuNoneOfTheProbesProcessesUsedIsOtherPrograms() {
		Map<Long, Double> before = Map.of(1L, 10.0, 2L, 4.0);
		Map<Long, Double> now = Map.of(1L, 11.0, 2L, 4.5);
		assertThat(LoadSampler.otherCpuSeconds(2.5 / CPUS, CPUS, 1.0, before, now)).hasValueCloseTo(1.0,
				within(1e-9));
	}
}
