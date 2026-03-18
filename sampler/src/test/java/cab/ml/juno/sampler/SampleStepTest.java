package cab.ml.juno.sampler;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class SampleStepTest {

	private final SampleStep step = SampleStep.INSTANCE;

	@Test
	void greedy_always_returns_highest_probability_token() {
		float[] probs = { 0.1f, 0.6f, 0.3f };
		SamplingParams params = SamplingParams.deterministic();

		int token = step.sample(probs, params);

		assertThat(token).isEqualTo(1); // index 1 has prob 0.6
	}

	@Test
	void greedy_is_deterministic_across_multiple_calls() {
		float[] probs = { 0.2f, 0.5f, 0.3f };
		SamplingParams params = SamplingParams.deterministic();

		int first = step.sample(probs.clone(), params);
		int second = step.sample(probs.clone(), params);
		int third = step.sample(probs.clone(), params);

		assertThat(first).isEqualTo(second).isEqualTo(third).isEqualTo(1);
	}

	@Test
	void sampled_token_is_within_valid_range() {
		float[] probs = { 0.2f, 0.3f, 0.4f, 0.1f };
		SamplingParams params = SamplingParams.defaults();

		for (int i = 0; i < 100; i++) {
			int token = step.sample(probs.clone(), params);
			assertThat(token).isBetween(0, probs.length - 1);
		}
	}

	@Test
	void one_hot_distribution_always_returns_that_token() {
		float[] probs = { 0.0f, 0.0f, 1.0f, 0.0f }; // only token 2 has probability
		SamplingParams params = SamplingParams.defaults(); // stochastic mode

		for (int i = 0; i < 50; i++) {
			int token = step.sample(probs.clone(), params);
			assertThat(token).isEqualTo(2);
		}
	}
}
