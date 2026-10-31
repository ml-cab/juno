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

import java.util.Set;

import org.junit.jupiter.api.Test;

class GpuLoadTest {

	private static final String APPS = """
			58879, /usr/lib/jvm/java-25-openjdk-amd64/bin/java, 916
			58918, /usr/lib/jvm/java-25-openjdk-amd64/bin/java, 914
			60001, /usr/bin/blender, 2048
			""";

	@Test
	void ownProcessesMemoryAndOtherComputeProcessesAreSeparated() {
		GpuLoad.Sample s = GpuLoad.computeApps(APPS, Set.of(58879L, 58918L));
		assertThat(s.ownMiB()).isEqualTo(1830);
		assertThat(s.others()).containsExactly("60001 /usr/bin/blender");
	}

	@Test
	void noToolOrNoProcessesIsAnEmptySample() {
		assertThat(GpuLoad.computeApps("", Set.of(1L))).isEqualTo(new GpuLoad.Sample(0, java.util.List.of()));
		assertThat(GpuLoad.computeApps("No running processes found\n", Set.of(1L)).others()).isEmpty();
	}

	@Test
	void utilizationIsTheBusiestDeviceAndMinusOneWithoutAReading() {
		assertThat(GpuLoad.utilizationPercent("10\n")).isEqualTo(10);
		assertThat(GpuLoad.utilizationPercent("3\n47\n")).isEqualTo(47);
		assertThat(GpuLoad.utilizationPercent("")).isEqualTo(-1);
		assertThat(GpuLoad.busy(java.util.List.of(35, 12, 40))).isFalse(); // the quietest sample decides
		assertThat(GpuLoad.busy(java.util.List.of(35, 25, 40))).isTrue();
	}
}
