package cab.ml.juno.sampler;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;

class SamplerTest {

	private final Sampler sampler = Sampler.create();

	@Test
	void greedy_pipeline_returns_highest_logit_token() {
		// Clear winner at index 3
		float[] logits = { 0.1f, 0.5f, 0.3f, 10.0f, 0.2f };
		int token = sampler.sample(logits, SamplingParams.deterministic());
		assertThat(token).isEqualTo(3);
	}

	@Test
	void stochastic_pipeline_returns_token_in_valid_range() {
		float[] logits = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
		SamplingParams params = SamplingParams.defaults();

		for (int i = 0; i < 200; i++) {
			int token = sampler.sample(logits, params, new int[0]);
			assertThat(token).isBetween(0, logits.length - 1);
		}
	}

	@Test
	void does_not_mutate_caller_logits() {
		float[] logits = { 1.0f, 2.0f, 3.0f };
		float[] copy = logits.clone();

		sampler.sample(logits, SamplingParams.defaults());

		assertThat(logits).containsExactly(copy); // original untouched
	}

	@Test
	void stop_token_detected_correctly() {
		SamplingParams params = SamplingParams.defaults().withStopTokenIds(2, 5);
		assertThat(sampler.isStopToken(2, params)).isTrue();
		assertThat(sampler.isStopToken(5, params)).isTrue();
		assertThat(sampler.isStopToken(3, params)).isFalse();
	}

	@Test
	void rejects_null_logits() {
		assertThatThrownBy(() -> sampler.sample(null, SamplingParams.defaults()))
				.isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	void rejects_empty_logits() {
		assertThatThrownBy(() -> sampler.sample(new float[0], SamplingParams.defaults()))
				.isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	void creative_profile_produces_varied_output() {
		// With a very flat distribution and creative settings,
		// 200 samples should not all return the same token
		float[] logits = { 1.0f, 1.01f, 1.02f, 1.03f, 1.04f };
		SamplingParams params = SamplingParams.creative();

		boolean[] seen = new boolean[logits.length];
		for (int i = 0; i < 200; i++) {
			seen[sampler.sample(logits, params)] = true;
		}

		long distinct = 0;
		for (boolean b : seen)
			if (b)
				distinct++;
		assertThat(distinct).isGreaterThan(1); // creative mode should vary
	}
}
