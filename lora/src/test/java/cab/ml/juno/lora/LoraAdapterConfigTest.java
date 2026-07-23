package cab.ml.juno.lora;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("LoraAdapterConfig")
class LoraAdapterConfigTest {

	@Test
	@DisplayName("rejects invalid rank, non-finite alpha, and null enums")
	void validation() {
		assertThatThrownBy(() -> LoraAdapterConfig.of(0, 1f)).isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> LoraAdapterConfig.of(4, Float.NaN)).isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> LoraAdapterConfig.of(4, 1f, null, LoraInitialization.KAIMING_UNIFORM, LoraMode.LORA))
				.isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> LoraAdapterConfig.of(4, 1f, LoraScaling.STANDARD, null, LoraMode.LORA))
				.isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(
				() -> LoraAdapterConfig.of(4, 1f, LoraScaling.STANDARD, LoraInitialization.KAIMING_UNIFORM, null))
				.isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	@DisplayName("standard scale is alpha/rank; rsLoRA is alpha/sqrt(rank)")
	void effective_scales() {
		assertThat(LoraAdapterConfig.of(4, 8f).effectiveScale()).isEqualTo(2f);
		assertThat(LoraAdapterConfig
				.of(4, 8f, LoraScaling.RANK_STABILIZED, LoraInitialization.KAIMING_UNIFORM, LoraMode.LORA)
				.effectiveScale()).isCloseTo(8f / 2f, within(1e-6f));
		assertThat(LoraScaling.STANDARD.effectiveScale(16f, 8)).isEqualTo(2f);
		assertThat(LoraScaling.RANK_STABILIZED.effectiveScale(16f, 16)).isEqualTo(4f);
	}

	@Test
	@DisplayName("legacy() is standard + legacy-normal + LoRA")
	void legacy_defaults() {
		LoraAdapterConfig c = LoraAdapterConfig.legacy(8, 16f);
		assertThat(c.scaling()).isEqualTo(LoraScaling.STANDARD);
		assertThat(c.initialization()).isEqualTo(LoraInitialization.LEGACY_NORMAL);
		assertThat(c.mode()).isEqualTo(LoraMode.LORA);
		assertThat(c.effectiveScale()).isEqualTo(2f);
	}

	@Test
	@DisplayName("of(rank, alpha) defaults to Kaiming-uniform")
	void of_defaults_kaiming() {
		assertThat(LoraAdapterConfig.of(4, 4f).initialization()).isEqualTo(LoraInitialization.KAIMING_UNIFORM);
	}
}
