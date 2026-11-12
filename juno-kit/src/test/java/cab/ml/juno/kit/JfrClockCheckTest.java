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

import java.util.List;
import java.util.Map;
import java.util.Set;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

/**
 * JFR's clock against the operating system's, checked once per host. A laptop whose CPU counter ran at about
 * 1.99 GHz, against the 1.80 GHz in its brand string, read every JFR span 1.1066 times too long.
 */
class JfrClockCheckTest {

	@AfterEach
	void forgetTheDecision() {
		System.clearProperty(JvmLauncher.OS_CLOCK_PROPERTY);
	}

	@Test
	void aClockWithinOnePercentKeepsTheCpuCounter() {
		assertThat(JfrClockCheck.disagrees(1.0)).isFalse();
		assertThat(JfrClockCheck.disagrees(1.0099)).isFalse();
		assertThat(JfrClockCheck.disagrees(0.9901)).isFalse();
	}

	@Test
	void aCounterFastOrSlowByMoreThanOnePercentDisagrees() {
		assertThat(JfrClockCheck.disagrees(1.1066)).as("the laptop's counter").isTrue();
		assertThat(JfrClockCheck.disagrees(1.0101)).isTrue();
		assertThat(JfrClockCheck.disagrees(0.95)).as("a counter slower than its brand string").isTrue();
		assertThat(JfrClockCheck.disagrees(0.0)).isTrue();
	}

	@Test
	void aCheckThatDidNotRunLeavesTheClockSourceRule() {
		assertThat(JfrClockCheck.disagrees(Double.NaN)).isFalse();
		assertThat(JfrClockCheck.osClockNeeded("tsc", Double.NaN)).isFalse();
		assertThat(JfrClockCheck.osClockNeeded("hpet", Double.NaN)).isTrue();
		assertThat(JfrClockCheck.osClockNeeded("tsc", 1.0004)).isFalse();
		assertThat(JfrClockCheck.osClockNeeded("tsc", 1.1066)).isTrue();
		assertThat(JfrClockCheck.osClockNeeded("unknown", 0.95)).as("Windows and macOS report no clock source")
				.isTrue();
	}

	@Test
	void theCheckTimesASpanOnThisJvmsJfrClock() {
		double ratio = JfrClockCheck.measure();
		assertThat(ratio).isFinite().isBetween(0.5, 2.0);
	}

	@Test
	void theHostsDecisionReachesEveryJvmTheKitStartsAndTheirs() {
		JvmLauncher.useClockOf(host("tsc", Map.of("jfrClockRatio", 1.1066)));
		List<String> os = JvmLauncher.command("1g", false, List.of(), "Main", List.of());
		assertThat(os).containsSubsequence(JvmLauncher.OS_CLOCK_FLAGS)
				.contains("-D" + JvmLauncher.OS_CLOCK_PROPERTY + "=true");

		JvmLauncher.useClockOf(host("tsc", Map.of("jfrClockRatio", 1.0004)));
		List<String> cpu = JvmLauncher.command("1g", false, List.of(), "Main", List.of());
		assertThat(cpu).doesNotContain("-XX:-UseFastUnorderedTimeStamps")
				.contains("-D" + JvmLauncher.OS_CLOCK_PROPERTY + "=false");
	}

	@Test
	void aHostWithoutTheCheckKeepsTheClockSourceRule() {
		JvmLauncher.useClockOf(host("hpet", Map.of()));
		assertThat(JvmLauncher.command("1g", false, List.of(), "Main", List.of()))
				.containsSubsequence(JvmLauncher.OS_CLOCK_FLAGS);
		JvmLauncher.useClockOf(host("tsc", Map.of("error", "backend probe timed out")));
		assertThat(JvmLauncher.command("1g", false, List.of(), "Main", List.of()))
				.doesNotContain("-XX:-UseFastUnorderedTimeStamps");
	}

	@Test
	void theSummarySaysWhichClockTheProbesRead() {
		assertThat(JfrClockCheck.describe("tsc", 1.1066))
				.isEqualTo("JFR clock: the operating system's; the CPU counter read 1.1066x it");
		assertThat(JfrClockCheck.describe("tsc", 1.0004)).isEqualTo("JFR clock: the CPU counter, 1.0004x the operating system's");
		assertThat(JfrClockCheck.describe("hpet", Double.NaN))
				.isEqualTo("JFR clock: the operating system's; kernel clock source hpet");
		assertThat(JfrClockCheck.describe("tsc", Double.NaN)).isEqualTo("JFR clock: the CPU counter, not checked");
	}

	private static HostInfo host(String clockSource, Map<String, Object> backend) {
		return new HostInfo("Linux", "6.8", "amd64", "Test CPU", 4, 8, 8L << 30, 100L << 30, Set.of(), 256, 256, "2",
				false, false, List.of(), new HostInfo.Power(false, false, -1), "no", clockSource, backend);
	}
}
