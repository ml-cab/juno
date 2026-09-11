package cab.ml.juno.sampler;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Test;

class PresencePenaltyStepTest {

	private final PresencePenaltyStep step = PresencePenaltyStep.INSTANCE;

	@Test
	void subtracts_penalty_from_seen_token_logits() {
		float[] logits = { 2.0f, 1.0f, 0.5f, -0.25f };
		SamplingParams params = SamplingParams.defaults().withPresencePenalty(0.5f);

		step.apply(logits, params, new int[] { 0, 3 });

		assertThat(logits[0]).isCloseTo(1.5f, within(1e-5f));
		assertThat(logits[1]).isCloseTo(1.0f, within(1e-5f));
		assertThat(logits[2]).isCloseTo(0.5f, within(1e-5f));
		assertThat(logits[3]).isCloseTo(-0.75f, within(1e-5f));
	}

	@Test
	void changes_ranking_vs_baseline() {
		float[] baseline = { 1.0f, 0.9f, 0.2f };
		float[] penalized = baseline.clone();
		SamplingParams params = SamplingParams.defaults().withPresencePenalty(1.0f);

		step.apply(penalized, params, new int[] { 0 });

		assertThat(penalized[0]).isLessThan(baseline[0]);
		assertThat(penalized[0]).isLessThan(penalized[1]);
	}

	@Test
	void skips_when_penalty_is_zero() {
		float[] logits = { 1.0f, 2.0f };
		float[] original = logits.clone();
		SamplingParams params = SamplingParams.defaults().withPresencePenalty(0.0f);

		step.apply(logits, params, new int[] { 0 });

		assertThat(logits).containsExactly(original);
	}

	@Test
	void skips_when_no_history() {
		float[] logits = { 1.0f, 2.0f };
		float[] original = logits.clone();
		SamplingParams params = SamplingParams.defaults().withPresencePenalty(1.0f);

		step.apply(logits, params, new int[0]);

		assertThat(logits).containsExactly(original);
	}

	@Test
	void negative_penalty_boosts_seen_tokens() {
		float[] logits = { 1.0f, 1.0f };
		SamplingParams params = SamplingParams.defaults().withPresencePenalty(-0.5f);

		step.apply(logits, params, new int[] { 0 });

		assertThat(logits[0]).isCloseTo(1.5f, within(1e-5f));
		assertThat(logits[1]).isCloseTo(1.0f, within(1e-5f));
	}
}
