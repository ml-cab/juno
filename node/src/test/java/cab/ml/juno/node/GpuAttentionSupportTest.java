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

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Which handlers can run the GPU attention kernel, and what a console launch
 * says when {@code --gpu-attention} cannot take effect. The notice is the
 * point: a launch that asks for the kernel and does not get it must say so
 * rather than resolve to the scalar path silently.
 */
@DisplayName("GpuAttentionSupport - per-architecture coverage and startup notice")
class GpuAttentionSupportTest {

	private String saved;

	@BeforeEach
	void save() {
		saved = System.getProperty(GpuAttentionOptions.ENV_PROPERTY);
	}

	@AfterEach
	void restore() {
		if (saved == null)
			System.clearProperty(GpuAttentionOptions.ENV_PROPERTY);
		else
			System.setProperty(GpuAttentionOptions.ENV_PROPERTY, saved);
	}

	@Test
	@DisplayName("the kernel is wired into the Llama-family, Phi-3 and Qwen3 handlers, not Phi-2 or Qwen3-MoE")
	void kernel_coverage_per_architecture() {
		for (String arch : new String[] { "llama", "mistral", "tinyllama", "qwen2", "phi3", "qwen3" })
			assertThat(GpuAttentionSupport.handlerRunsKernel(arch)).as(arch).isTrue();
		for (String arch : new String[] { "phi2", "qwen3moe" })
			assertThat(GpuAttentionSupport.handlerRunsKernel(arch)).as(arch).isFalse();
		assertThat(GpuAttentionSupport.handlerRunsKernel("PHI3")).as("case-insensitive").isTrue();
	}

	@Test
	@DisplayName("Phi-2 and Qwen3-MoE on a GPU run say they compute on the CPU, whatever --gpu-attention is")
	void cpu_only_handlers_announce_on_a_gpu_run() {
		for (String mode : new String[] { "on", "auto", "off" }) {
			System.setProperty(GpuAttentionOptions.ENV_PROPERTY, mode);
			for (String arch : new String[] { "phi2", "qwen3moe" }) {
				String notice = GpuAttentionSupport.consoleNotice(arch, false, false, true);
				assertThat(notice).as(arch + " with --gpu-attention " + mode).isNotNull().contains(arch)
						.contains("CPU").contains("--gpu-layers").contains("--gpu-attention");
			}
		}
	}

	@Test
	@DisplayName("a CPU run of a CPU-only handler is exactly what was asked for, so it says nothing")
	void cpu_only_handler_on_cpu_run_is_silent() {
		System.setProperty(GpuAttentionOptions.ENV_PROPERTY, "auto");
		assertThat(GpuAttentionSupport.consoleNotice("phi2", false, true, false)).isNull();
		assertThat(GpuAttentionSupport.consoleNotice("qwen3moe", false, true, false)).isNull();
	}

	@Test
	@DisplayName("a GPU run on a backend without the kernel (ROCm) names the backend and the CPU path, under on and auto")
	void non_cuda_backend_announces() {
		for (String mode : new String[] { "on", "auto" }) {
			System.setProperty(GpuAttentionOptions.ENV_PROPERTY, mode);
			String notice = GpuAttentionSupport.consoleNotice("llama", false, false, false);
			assertThat(notice).as("--gpu-attention " + mode).isNotNull().contains("CUDA").contains("CPU");
		}
		System.setProperty(GpuAttentionOptions.ENV_PROPERTY, "off");
		assertThat(GpuAttentionSupport.consoleNotice("llama", false, false, false)).as("off asks for nothing").isNull();
	}

	@Test
	@DisplayName("a covered handler on CUDA says nothing: the kernel runs")
	void covered_handler_on_cuda_is_silent() {
		System.setProperty(GpuAttentionOptions.ENV_PROPERTY, "auto");
		for (String arch : new String[] { "llama", "qwen2", "phi3", "qwen3" })
			assertThat(GpuAttentionSupport.consoleNotice(arch, false, false, true)).as(arch).isNull();
	}

	@Test
	@DisplayName("an explicit on with the CPU backend says it has no effect; auto on the CPU backend stays silent")
	void explicit_on_with_cpu_backend() {
		System.setProperty(GpuAttentionOptions.ENV_PROPERTY, "on");
		assertThat(GpuAttentionSupport.consoleNotice("llama", false, true, false)).isNotNull().contains("CPU backend");
		System.setProperty(GpuAttentionOptions.ENV_PROPERTY, "auto");
		assertThat(GpuAttentionSupport.consoleNotice("llama", false, true, false)).isNull();
	}

	@Test
	@DisplayName("LoRA training and --lora-play are left to their own notice")
	void lora_is_left_to_its_own_notice() {
		System.setProperty(GpuAttentionOptions.ENV_PROPERTY, "on");
		assertThat(GpuAttentionSupport.consoleNotice("phi3", true, false, true)).isNull();
		assertThat(GpuAttentionSupport.consoleNotice("llama", true, false, false)).isNull();
	}
}
