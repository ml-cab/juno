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
package cab.ml.juno.player;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import cab.ml.juno.node.CpuThreads;

/**
 * {@code --threads}: parsed as a positive count (anything else is refused with
 * the value named), forwarded to every forked cluster node, and settable from
 * the embedding facade.
 */
class CpuThreadsFlagTest {

	private final int original = CpuThreads.threads();

	@AfterEach
	void restore() {
		System.clearProperty(CpuThreads.PROPERTY);
		CpuThreads.setThreads(original);
	}

	@Test
	void parsesAPositiveCountAndRefusesAnythingElse() {
		assertThat(ConsoleMain.parseThreads("6")).isEqualTo(6);
		assertThat(ConsoleMain.parseThreads(" 1 ")).isEqualTo(1);
		for (String bad : new String[] { "0", "-2", "six", "" })
			assertThatThrownBy(() -> ConsoleMain.parseThreads(bad)).isInstanceOf(IllegalArgumentException.class)
					.hasMessageContaining("--threads").hasMessageContaining("'" + bad + "'");
	}

	@Test
	void clusterNodesGetTheExplicitCountElseAnEvenShare() {
		System.clearProperty(CpuThreads.PROPERTY);
		org.junit.jupiter.api.Assumptions.assumeTrue(System.getenv(CpuThreads.ENV) == null,
				"JUNO_THREADS is set in this environment");
		assertThat(ClusterHarness.nodeThreads(4)).isEqualTo(4);
		System.setProperty(CpuThreads.PROPERTY, "3");
		assertThat(ClusterHarness.nodeThreads(4)).isEqualTo(3);
	}

	@Test
	void theFacadeSetsTheJvmThreadCount() {
		JunoPlayer.Builder b = JunoPlayer.builder(java.nio.file.Path.of("unused.gguf")).threads(2);
		assertThat(b).isNotNull();
		assertThatThrownBy(() -> JunoPlayer.builder(java.nio.file.Path.of("unused.gguf")).threads(0))
				.isInstanceOf(IllegalArgumentException.class);
	}
}
