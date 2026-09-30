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
 * LoRA training and {@code --lora-play} keep attention on the CPU whatever
 * {@code --gpu-attention} says, on every LoRA architecture. The notice that says so
 * is raised by {@link LoraTrainingHandlerFactory}, which every LoRA handler passes
 * through, so no architecture drops the flag silently. (Before, only the
 * Llama-family handler raised it; the Phi-3, Qwen2 and Qwen3 handlers did not.)
 */
@DisplayName("LoRA - --gpu-attention exemption is announced for every LoRA architecture")
class LoraGpuAttentionNoticeTest {

	private String saved;

	@BeforeEach
	void save() {
		saved = System.getProperty(GpuAttentionOptions.ENV_PROPERTY);
		LoraTrainNotices.clear();
	}

	@AfterEach
	void restore() {
		if (saved == null)
			System.clearProperty(GpuAttentionOptions.ENV_PROPERTY);
		else
			System.setProperty(GpuAttentionOptions.ENV_PROPERTY, saved);
		LoraTrainNotices.clear();
	}

	@Test
	@DisplayName("a GPU LoRA load with the kernel requested raises the notice, on every load")
	void gpu_load_with_kernel_requested_notices() {
		System.setProperty(GpuAttentionOptions.ENV_PROPERTY, "on");
		LoraTrainingHandlerFactory.noteGpuAttentionIgnored(true);
		assertThat(LoraTrainNotices.drain()).contains(LoraTrainNotices.GPU_ATTENTION_IGNORED.strip());
		LoraTrainingHandlerFactory.noteGpuAttentionIgnored(true);
		assertThat(LoraTrainNotices.drain()).as("a second load in the same process still tells its user")
				.contains(LoraTrainNotices.GPU_ATTENTION_IGNORED.strip());
	}

	@Test
	@DisplayName("off, or the CPU backend, asks for nothing the handler ignores")
	void off_or_cpu_is_silent() {
		System.setProperty(GpuAttentionOptions.ENV_PROPERTY, "off");
		LoraTrainingHandlerFactory.noteGpuAttentionIgnored(true);
		assertThat(LoraTrainNotices.drain()).isEmpty();
		System.setProperty(GpuAttentionOptions.ENV_PROPERTY, "on");
		LoraTrainingHandlerFactory.noteGpuAttentionIgnored(false);
		assertThat(LoraTrainNotices.drain()).isEmpty();
	}
}
