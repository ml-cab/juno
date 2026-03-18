package cab.ml.juno.sampler;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

class SamplingParamsTest {

	@Test
	void defaults_have_expected_values() {
		SamplingParams p = SamplingParams.defaults();
		assertThat(p.temperature()).isEqualTo(0.7f);
		assertThat(p.topK()).isEqualTo(50);
		assertThat(p.topP()).isEqualTo(0.9f);
		assertThat(p.repetitionPenalty()).isEqualTo(1.1f);
		assertThat(p.greedy()).isFalse();
		assertThat(p.maxTokens()).isEqualTo(512);
	}

	@Test
	void deterministic_profile_enables_greedy() {
		SamplingParams p = SamplingParams.deterministic();
		assertThat(p.greedy()).isTrue();
		assertThat(p.temperature()).isLessThanOrEqualTo(0.1f);
	}

	@Test
	void creative_profile_has_high_temperature() {
		SamplingParams p = SamplingParams.creative();
		assertThat(p.temperature()).isGreaterThan(1.0f);
		assertThat(p.topK()).isGreaterThanOrEqualTo(100);
	}

	@Test
	void fluent_builder_overrides_single_field() {
		SamplingParams p = SamplingParams.defaults().withTemperature(1.5f);
		assertThat(p.temperature()).isEqualTo(1.5f);
		assertThat(p.topK()).isEqualTo(50); // unchanged
	}

	@Test
	void rejects_negative_topK() {
		assertThatThrownBy(() -> SamplingParams.defaults().withTopK(-1)).isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	void rejects_temperature_out_of_range() {
		assertThatThrownBy(() -> SamplingParams.defaults().withTemperature(3.0f))
				.isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	void stopTokenIds_returns_defensive_copy() {
		SamplingParams p = SamplingParams.defaults().withStopTokenIds(1, 2, 3);
		p.stopTokenIds()[0] = 999; // mutate
		assertThat(p.stopTokenIds()[0]).isEqualTo(1); // original unchanged
	}
}
