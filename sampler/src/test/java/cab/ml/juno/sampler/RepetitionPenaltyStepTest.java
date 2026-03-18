package cab.ml.juno.sampler;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Test;

class RepetitionPenaltyStepTest {

	private final RepetitionPenaltyStep step = RepetitionPenaltyStep.INSTANCE;

	@Test
	void reduces_probability_of_seen_tokens() {
		float[] probs = { 0.25f, 0.25f, 0.25f, 0.25f };
		int[] generated = { 0, 1 }; // tokens 0 and 1 already seen
		SamplingParams params = SamplingParams.defaults().withRepetitionPenalty(1.5f);

		step.apply(probs, params, generated);

		// Tokens 0 and 1 should have lower relative probability
		assertThat(probs[0]).isLessThan(probs[2]);
		assertThat(probs[1]).isLessThan(probs[3]);
	}

	@Test
	void skips_when_penalty_is_one() {
		float[] probs = { 0.25f, 0.25f, 0.25f, 0.25f };
		float[] original = probs.clone();
		SamplingParams params = SamplingParams.defaults().withRepetitionPenalty(1.0f);

		step.apply(probs, params, new int[] { 0, 1, 2 });

		assertThat(probs).containsExactly(original);
	}

	@Test
	void skips_when_no_generated_tokens() {
		float[] probs = { 0.25f, 0.25f, 0.25f, 0.25f };
		float[] original = probs.clone();
		SamplingParams params = SamplingParams.defaults().withRepetitionPenalty(1.3f);

		step.apply(probs, params, new int[0]);

		assertThat(probs).containsExactly(original);
	}

	@Test
	void negative_logits_are_pushed_further_down_after_penalty() {
		// RepetitionPenaltyStep operates on raw logits (before softmax).
		// Positive logits are divided, negative logits are multiplied —
		// both reduce the token's eventual probability. No renormalization;
		// that is softmax's job, which runs after this step.
		float[] logits = { 2.0f, -1.0f, 0.5f, -0.5f };
		SamplingParams params = SamplingParams.defaults().withRepetitionPenalty(1.3f);

		step.apply(logits, params, new int[] { 0, 1 }); // penalise tokens 0 (positive) and 1 (negative)

		// Token 0: positive logit → divided → less positive
		assertThat(logits[0]).isCloseTo(2.0f / 1.3f, within(1e-5f));
		// Token 1: negative logit → multiplied → more negative
		assertThat(logits[1]).isCloseTo(-1.0f * 1.3f, within(1e-5f));
		// Unseen tokens unchanged
		assertThat(logits[2]).isCloseTo(0.5f, within(1e-5f));
		assertThat(logits[3]).isCloseTo(-0.5f, within(1e-5f));
	}
}