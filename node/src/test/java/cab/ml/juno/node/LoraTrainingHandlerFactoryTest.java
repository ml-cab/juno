package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("LoraTrainingHandlerFactory — architecture allowlist")
class LoraTrainingHandlerFactoryTest {

	@Test
	@DisplayName("accepts Tier 6 dense architectures")
	void acceptsSupported() {
		assertThat(LoraTrainingHandlerFactory.isSupported("llama")).isTrue();
		assertThat(LoraTrainingHandlerFactory.isSupported("mistral")).isTrue();
		assertThat(LoraTrainingHandlerFactory.isSupported("tinyllama")).isTrue();
		assertThat(LoraTrainingHandlerFactory.isSupported("qwen2")).isTrue();
		assertThat(LoraTrainingHandlerFactory.isSupported("qwen2.5")).isTrue();
		assertThat(LoraTrainingHandlerFactory.isSupported("phi3")).isTrue();
		assertThat(LoraTrainingHandlerFactory.isSupported("qwen3")).isTrue();
		assertThat(LoraTrainingHandlerFactory.isSupported(null)).isTrue(); // defaults to llama
	}

	@Test
	@DisplayName("rejects MoE, gemma, and unknown")
	void rejectsUnsupported() {
		assertThat(LoraTrainingHandlerFactory.isSupported("qwen3moe")).isFalse();
		assertThat(LoraTrainingHandlerFactory.isSupported("qwen35")).isFalse();
		assertThat(LoraTrainingHandlerFactory.isSupported("gemma")).isFalse();
		assertThat(LoraTrainingHandlerFactory.isSupported("unknown-arch")).isFalse();

		assertThatThrownBy(() -> LoraTrainingHandlerFactory.requireSupported("gemma"))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("gemma");
		assertThatThrownBy(() -> LoraTrainingHandlerFactory.requireSupported("qwen3moe"))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("qwen3moe");
		assertThatThrownBy(() -> LoraTrainingHandlerFactory.requireSupported("phi2"))
				.isInstanceOf(IllegalArgumentException.class);
	}
}
