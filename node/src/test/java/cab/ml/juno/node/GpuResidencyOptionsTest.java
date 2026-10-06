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
import org.junit.jupiter.api.Test;

/**
 * {@code --gpu-residency} / {@code JUNO_GPU_RESIDENCY}: off unless asked for,
 * because the path has not yet been measured end to end.
 */
class GpuResidencyOptionsTest {

	@AfterEach
	void clear() {
		System.clearProperty(GpuResidencyOptions.ENV_PROPERTY);
		System.clearProperty(GpuAttentionOptions.ENV_PROPERTY);
	}

	@Test
	void parses_every_spelling_and_rejects_others() {
		assertThat(GpuResidencyOptions.parse("on").mode()).isEqualTo(GpuResidencyOptions.Mode.ON);
		assertThat(GpuResidencyOptions.parse(" TRUE ").mode()).isEqualTo(GpuResidencyOptions.Mode.ON);
		assertThat(GpuResidencyOptions.parse("1").mode()).isEqualTo(GpuResidencyOptions.Mode.ON);
		assertThat(GpuResidencyOptions.parse("off").mode()).isEqualTo(GpuResidencyOptions.Mode.OFF);
		assertThat(GpuResidencyOptions.parse("no").mode()).isEqualTo(GpuResidencyOptions.Mode.OFF);
		assertThat(GpuResidencyOptions.parse("auto").mode()).isEqualTo(GpuResidencyOptions.Mode.AUTO);
		assertThat(GpuResidencyOptions.parse("").mode()).isEqualTo(GpuResidencyOptions.Mode.OFF);
		assertThatThrownBy(() -> GpuResidencyOptions.parse("sometimes"))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("--gpu-residency");
	}

	@Test
	void defaults_to_off_when_nothing_is_set() {
		// The environment variable could be set on a developer machine; the
		// property wins over it, so clear only what this test controls.
		if (System.getenv(GpuResidencyOptions.ENV_PROPERTY) == null) {
			assertThat(GpuResidencyOptions.fromEnv().mode()).isEqualTo(GpuResidencyOptions.Mode.OFF);
			assertThat(GpuResidencyOptions.fromEnv().requested()).isFalse();
		}
	}

	@Test
	void the_system_property_selects_the_mode() {
		System.setProperty(GpuResidencyOptions.ENV_PROPERTY, "on");
		assertThat(GpuResidencyOptions.fromEnv().mode()).isEqualTo(GpuResidencyOptions.Mode.ON);
		assertThat(GpuResidencyOptions.fromEnv().requested()).isTrue();
		System.setProperty(GpuResidencyOptions.ENV_PROPERTY, "auto");
		assertThat(GpuResidencyOptions.fromEnv().requested()).isEqualTo(CudaAvailability.isAvailable());
	}

	@Test
	void an_unsupported_surface_is_announced_once_not_per_instance() {
		String surface = "test-surface-" + System.nanoTime();
		assertThat(GpuResidencyOptions.announceOnce(surface)).isTrue();
		assertThat(GpuResidencyOptions.announceOnce(surface)).isFalse();
	}

	@Test
	void names_every_architecture_the_region_cannot_run_and_none_it_can() {
		for (String arch : new String[] { "llama", "mistral", "tinyllama" })
			assertThat(GpuResidencyOptions.unsupportedArchitectureReason(arch)).as(arch).isNull();
		assertThat(GpuResidencyOptions.unsupportedArchitectureReason("qwen2")).contains("split-half");
		assertThat(GpuResidencyOptions.unsupportedArchitectureReason("qwen2.5")).contains("split-half");
		for (String arch : new String[] { "phi2", "phi3", "qwen3", "qwen3moe" })
			assertThat(GpuResidencyOptions.unsupportedArchitectureReason(arch)).as(arch).isNotNull();
	}

	@Test
	void the_console_notice_covers_lora_and_cpu_and_stays_quiet_when_not_requested() {
		System.setProperty(GpuResidencyOptions.ENV_PROPERTY, "on");
		assertThat(GpuResidencyOptions.consoleNotice("llama", false, false)).isNull();
		assertThat(GpuResidencyOptions.consoleNotice("llama", true, false)).contains("LoRA");
		assertThat(GpuResidencyOptions.consoleNotice("llama", false, true)).contains("CPU");
		assertThat(GpuResidencyOptions.consoleNotice("phi3", false, false)).contains("phi3");
		System.setProperty(GpuResidencyOptions.ENV_PROPERTY, "off");
		assertThat(GpuResidencyOptions.consoleNotice("phi3", true, true)).isNull();
	}

	@Test
	void the_console_notice_says_attention_stays_outside_the_region_with_gpu_attention_off() {
		System.setProperty(GpuResidencyOptions.ENV_PROPERTY, "on");
		System.setProperty(GpuAttentionOptions.ENV_PROPERTY, "off");
		assertThat(GpuResidencyOptions.consoleNotice("llama", false, false)).contains("--gpu-attention off")
				.contains("attention");
		System.setProperty(GpuAttentionOptions.ENV_PROPERTY, "on");
		assertThat(GpuResidencyOptions.consoleNotice("llama", false, false)).isNull();
		System.setProperty(GpuResidencyOptions.ENV_PROPERTY, "off");
		System.setProperty(GpuAttentionOptions.ENV_PROPERTY, "off");
		assertThat(GpuResidencyOptions.consoleNotice("llama", false, false)).as("residency not requested").isNull();
	}
}
