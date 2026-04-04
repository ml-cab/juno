/*
 * Created by Yevhen Soldatov
 * Initial implementation: 2026
 *
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

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("ComputeBackendPolicy")
class ComputeBackendPolicyTest {

	@Test
	@DisplayName("parsePreference maps legacy and new tokens")
	void parsePreference_tokens() {
		assertThat(ComputeBackendPolicy.parsePreference(null)).isEqualTo(ComputeBackendPreference.AUTO);
		assertThat(ComputeBackendPolicy.parsePreference("")).isEqualTo(ComputeBackendPreference.AUTO);
		assertThat(ComputeBackendPolicy.parsePreference("  auto  ")).isEqualTo(ComputeBackendPreference.AUTO);
		assertThat(ComputeBackendPolicy.parsePreference("TRUE")).isEqualTo(ComputeBackendPreference.AUTO);
		assertThat(ComputeBackendPolicy.parsePreference("false")).isEqualTo(ComputeBackendPreference.DISABLE);
		assertThat(ComputeBackendPolicy.parsePreference("CPU")).isEqualTo(ComputeBackendPreference.DISABLE);
		assertThat(ComputeBackendPolicy.parsePreference("require")).isEqualTo(ComputeBackendPreference.REQUIRE);
		assertThat(ComputeBackendPolicy.parsePreference("REQUIRED")).isEqualTo(ComputeBackendPreference.REQUIRE);
	}

	@Test
	@DisplayName("preference property tokens round-trip labels")
	void toPropertyToken() {
		assertThat(ComputeBackendPreference.DISABLE.toPropertyToken()).isEqualTo("false");
		assertThat(ComputeBackendPreference.AUTO.toPropertyToken()).isEqualTo("auto");
		assertThat(ComputeBackendPreference.REQUIRE.toPropertyToken()).isEqualTo("require");
	}
}
