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

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("GpuAttentionOptions — parse and prefer")
class GpuAttentionOptionsTest {

	private String original;

	@AfterEach
	void restore() {
		if (original == null)
			System.clearProperty(GpuAttentionOptions.ENV_PROPERTY);
		else
			System.setProperty(GpuAttentionOptions.ENV_PROPERTY, original);
	}

	@Test
	@DisplayName("parse on|off|auto and aliases")
	void parse_values() {
		assertThat(GpuAttentionOptions.parse("off").mode()).isEqualTo(GpuAttentionOptions.Mode.OFF);
		assertThat(GpuAttentionOptions.parse("ON").mode()).isEqualTo(GpuAttentionOptions.Mode.ON);
		assertThat(GpuAttentionOptions.parse("auto").mode()).isEqualTo(GpuAttentionOptions.Mode.AUTO);
		assertThat(GpuAttentionOptions.parse("1").mode()).isEqualTo(GpuAttentionOptions.Mode.ON);
		assertThat(GpuAttentionOptions.parse("0").mode()).isEqualTo(GpuAttentionOptions.Mode.OFF);
		assertThat(GpuAttentionOptions.parse(null).mode()).isEqualTo(GpuAttentionOptions.Mode.OFF);
		assertThat(GpuAttentionOptions.parse("").mode()).isEqualTo(GpuAttentionOptions.Mode.OFF);
	}

	@Test
	@DisplayName("invalid spec fails closed")
	void parse_invalid() {
		assertThatThrownBy(() -> GpuAttentionOptions.parse("maybe"))
				.isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("on|off|auto");
	}

	@Test
	@DisplayName("preferGpuAttention respects mode")
	void prefer_gpu_attention() {
		assertThat(GpuAttentionOptions.off().preferGpuAttention()).isFalse();
		assertThat(GpuAttentionOptions.on().preferGpuAttention()).isTrue();
		assertThat(GpuAttentionOptions.auto().preferGpuAttention()).isEqualTo(CudaAvailability.isAvailable());
	}

	@Test
	@DisplayName("fromEnv reads JUNO_GPU_ATTENTION")
	void from_env() {
		original = System.getProperty(GpuAttentionOptions.ENV_PROPERTY);
		System.setProperty(GpuAttentionOptions.ENV_PROPERTY, "on");
		assertThat(GpuAttentionOptions.fromEnv().mode()).isEqualTo(GpuAttentionOptions.Mode.ON);
	}
}
