package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("ForwardPassHandlerLoader LoRA architecture gate")
class ForwardPassHandlerLoaderLoraArchTest {

	@Test
	@DisplayName("rejects fused/MoE architectures")
	void rejects_incompatible() {
		assertThatThrownBy(() -> ForwardPassHandlerLoader.requireLoraCompatibleArchitecture("phi3"))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("phi3");
		assertThatThrownBy(() -> ForwardPassHandlerLoader.requireLoraCompatibleArchitecture("qwen3"))
				.isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> ForwardPassHandlerLoader.requireLoraCompatibleArchitecture("qwen3moe"))
				.isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	@DisplayName("allows dense LLaMA-family architectures")
	void allows_llama_family() {
		ForwardPassHandlerLoader.requireLoraCompatibleArchitecture("llama");
		ForwardPassHandlerLoader.requireLoraCompatibleArchitecture("mistral");
		ForwardPassHandlerLoader.requireLoraCompatibleArchitecture("gemma");
		assertThat(true).isTrue();
	}
}
