package cab.ml.juno.sampler;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Test;

class TopPStepTest {

	private final TopPStep step = TopPStep.INSTANCE;
	private final int[] noHistory = new int[0];

	@Test
	void zeroes_out_low_probability_tokens() {
		// After softmax: roughly 0.6, 0.3, 0.07, 0.03 for these logits
		// topP=0.9 should zero out the 0.03 token
		float[] probs = { 0.60f, 0.30f, 0.07f, 0.03f };
		SamplingParams params = SamplingParams.defaults().withTopP(0.9f);

		step.apply(probs, params, noHistory);

		assertThat(probs[3]).isEqualTo(0.0f); // lowest prob zeroed
		assertThat(probs[0]).isGreaterThan(0.0f);
		assertThat(probs[1]).isGreaterThan(0.0f);
	}

	@Test
	void remaining_probabilities_renormalize_to_one() {
		float[] probs = { 0.60f, 0.30f, 0.07f, 0.03f };
		SamplingParams params = SamplingParams.defaults().withTopP(0.9f);

		step.apply(probs, params, noHistory);

		float sum = 0.0f;
		for (float v : probs)
			sum += v;
		assertThat(sum).isCloseTo(1.0f, within(1e-5f));
	}

	@Test
	void disabled_when_topP_is_one() {
		float[] probs = { 0.25f, 0.25f, 0.25f, 0.25f };
		float[] original = probs.clone();
		SamplingParams params = SamplingParams.defaults().withTopP(1.0f);

		step.apply(probs, params, noHistory);

		assertThat(probs).containsExactly(original);
	}

	@Test
	void skips_in_greedy_mode() {
		float[] probs = { 0.6f, 0.3f, 0.1f };
		float[] original = probs.clone();
		SamplingParams params = SamplingParams.deterministic().withTopP(0.7f);

		step.apply(probs, params, noHistory);

		assertThat(probs).containsExactly(original);
	}
}
