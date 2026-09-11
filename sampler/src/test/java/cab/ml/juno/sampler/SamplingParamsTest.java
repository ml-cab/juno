package cab.ml.juno.sampler;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.util.Random;

import org.junit.jupiter.api.Test;

class SamplingParamsTest {

	@Test
	void defaults_have_expected_values() {
		SamplingParams p = SamplingParams.defaults();
		assertThat(p.temperature()).isEqualTo(0.7f);
		assertThat(p.topK()).isEqualTo(50);
		assertThat(p.topP()).isEqualTo(0.9f);
		assertThat(p.repetitionPenalty()).isEqualTo(1.1f);
		assertThat(p.presencePenalty()).isEqualTo(0.0f);
		assertThat(p.greedy()).isFalse();
		assertThat(p.maxTokens()).isEqualTo(200);
		assertThat(p.stopStrings()).isEmpty();
		assertThat(p.seed()).isNull();
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
	void rejects_presence_penalty_out_of_range() {
		assertThatThrownBy(() -> SamplingParams.defaults().withPresencePenalty(2.5f))
				.isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> SamplingParams.defaults().withPresencePenalty(-2.1f))
				.isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	void stopTokenIds_returns_defensive_copy() {
		SamplingParams p = SamplingParams.defaults().withStopTokenIds(1, 2, 3);
		p.stopTokenIds()[0] = 999; // mutate
		assertThat(p.stopTokenIds()[0]).isEqualTo(1); // original unchanged
	}

	@Test
	void stopStrings_returns_defensive_copy() {
		SamplingParams p = SamplingParams.defaults().withStopStrings("END", "STOP");
		p.stopStrings()[0] = "X";
		assertThat(p.stopStrings()[0]).isEqualTo("END");
	}

	@Test
	void seed_and_presence_fluent_builders() {
		SamplingParams p = SamplingParams.defaults().withSeed(42L).withPresencePenalty(0.8f);
		assertThat(p.seed()).isEqualTo(42L);
		assertThat(p.presencePenalty()).isCloseTo(0.8f, within(1e-6f));
	}

	@Test
	void clearSeed_sets_null() {
		assertThat(SamplingParams.defaults().withSeed(7L).withSeed(null).seed()).isNull();
	}

	@Test
	void seeded_sample_is_repeatable() {
		float[] logits = { 0.1f, 0.2f, 0.3f, 0.15f, 0.25f };
		SamplingParams params = SamplingParams.defaults().withTemperature(1.0f).withTopK(0).withTopP(1.0f)
				.withRepetitionPenalty(1.0f).withSeed(99L);
		Sampler sampler = Sampler.create();

		int[] a = new int[8];
		int[] b = new int[8];
		Random ra = new Random(99L);
		Random rb = new Random(99L);
		for (int i = 0; i < a.length; i++) {
			a[i] = sampler.sample(logits.clone(), params, new int[0], ra);
			b[i] = sampler.sample(logits.clone(), params, new int[0], rb);
		}
		assertThat(a).containsExactly(b);
	}
}
