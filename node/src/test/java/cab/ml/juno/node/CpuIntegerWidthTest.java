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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

/**
 * Selection of the CPU integer kernels' lane width ({@link CpuIntegerWidth}):
 * 128 bits by default on every host, since the 256-bit kernels are not yet
 * measured on a host with 256-bit integer instructions, and a forced width
 * through {@code -Djuno.cpu.integerWidth}.
 */
class CpuIntegerWidthTest {

	@Test
	void autoStaysAt128OnEveryHostUntilTheWideKernelsAreMeasured() {
		for (int preferred : new int[] { 64, 128, 256, 512 }) {
			assertThat(CpuIntegerWidth.select(null, preferred)).as("unset, preferred %d", preferred).isEqualTo(128);
			assertThat(CpuIntegerWidth.select("auto", preferred)).as("auto, preferred %d", preferred).isEqualTo(128);
			assertThat(CpuIntegerWidth.select(" AUTO ", preferred)).isEqualTo(128);
			assertThat(CpuIntegerWidth.select("", preferred)).isEqualTo(128);
		}
	}

	@Test
	void aForcedWidthOverridesTheHost() {
		assertThat(CpuIntegerWidth.select("128", 256)).isEqualTo(128);
		assertThat(CpuIntegerWidth.select("128", 512)).isEqualTo(128);
		assertThat(CpuIntegerWidth.select("256", 128)).isEqualTo(256);
		assertThat(CpuIntegerWidth.select(" 256 ", 256)).isEqualTo(256);
	}

	@Test
	void anUnsupportedWidthFailsClosed() {
		for (String bad : new String[] { "512", "64", "wide", "128bit" })
			assertThatThrownBy(() -> CpuIntegerWidth.select(bad, 256)).as(bad)
					.isInstanceOf(IllegalArgumentException.class)
					.hasMessageContaining(CpuIntegerWidth.PROPERTY)
					.hasMessageContaining(bad);
	}

	@Test
	void theSelectedWidthIsTheSelectionForThisHost() {
		assertThat(CpuIntegerWidth.BITS)
				.isEqualTo(CpuIntegerWidth.select(System.getProperty(CpuIntegerWidth.PROPERTY), CpuIntegerWidth.preferredBits()));
		assertThat(CpuIntegerWidth.WIDE).isEqualTo(CpuIntegerWidth.BITS == 256);
		if (System.getProperty(CpuIntegerWidth.PROPERTY) == null)
			assertThat(CpuIntegerWidth.BITS).as("the default is 128 bits on every host").isEqualTo(128);
	}

	@Test
	void summaryNamesTheWidthAndTheHostsPreference() {
		String s = CpuIntegerWidth.summary();
		assertThat(s).contains(CpuIntegerWidth.BITS + "-bit").contains("preferred " + CpuIntegerWidth.preferredBits() + "-bit");
		assertThat(VectorQuantKernels.diagnosticSummary()).contains(s);
	}
}
