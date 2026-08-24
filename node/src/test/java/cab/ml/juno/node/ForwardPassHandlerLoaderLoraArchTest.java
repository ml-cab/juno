package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("ForwardPassHandlerLoader LoRA architecture gate")
class ForwardPassHandlerLoaderLoraArchTest {

	@Test
	@DisplayName("rejects MoE, gemma, and unknown")
	void rejects_incompatible() {
		assertThatThrownBy(() -> ForwardPassHandlerLoader.requireLoraCompatibleArchitecture("qwen3moe"))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("qwen3moe");
		assertThatThrownBy(() -> ForwardPassHandlerLoader.requireLoraCompatibleArchitecture("gemma"))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("gemma");
		assertThatThrownBy(() -> ForwardPassHandlerLoader.requireLoraCompatibleArchitecture("qwen35"))
				.isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	@DisplayName("allows Tier 6 dense architectures including phi3 and qwen3")
	void allows_tier6() {
		ForwardPassHandlerLoader.requireLoraCompatibleArchitecture("llama");
		ForwardPassHandlerLoader.requireLoraCompatibleArchitecture("mistral");
		ForwardPassHandlerLoader.requireLoraCompatibleArchitecture("qwen2");
		ForwardPassHandlerLoader.requireLoraCompatibleArchitecture("qwen2.5");
		ForwardPassHandlerLoader.requireLoraCompatibleArchitecture("phi3");
		ForwardPassHandlerLoader.requireLoraCompatibleArchitecture("qwen3");
		assertThat(true).isTrue();
	}
}
