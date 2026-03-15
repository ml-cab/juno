package cab.ml.juno.sampler;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.sampler.TopKStep;

class TopKStepTest {

	private final TopKStep step = TopKStep.INSTANCE;
	private final int[] noHistory = new int[0];

	@Test
	void keeps_only_top_k_tokens() {
		// logits: indices 0,1,2,3,4 with values 1,5,3,9,7
		// topK=2 → keep indices 3 (9.0) and 4 (7.0)
		float[] logits = { 1.0f, 5.0f, 3.0f, 9.0f, 7.0f };
		SamplingParams params = SamplingParams.defaults().withTopK(2);

		step.apply(logits, params, noHistory);

		assertThat(logits[0]).isEqualTo(Float.NEGATIVE_INFINITY);
		assertThat(logits[1]).isEqualTo(Float.NEGATIVE_INFINITY);
		assertThat(logits[2]).isEqualTo(Float.NEGATIVE_INFINITY);
		assertThat(logits[3]).isEqualTo(9.0f);
		assertThat(logits[4]).isEqualTo(7.0f);
	}

	@Test
	void disabled_when_topK_is_zero() {
		float[] logits = { 1.0f, 2.0f, 3.0f };
		float[] original = logits.clone();
		SamplingParams params = SamplingParams.defaults().withTopK(0);

		step.apply(logits, params, noHistory);

		assertThat(logits).containsExactly(original);
	}

	@Test
	void skips_in_greedy_mode() {
		float[] logits = { 1.0f, 2.0f, 3.0f };
		float[] original = logits.clone();
		SamplingParams params = SamplingParams.deterministic().withTopK(1);

		step.apply(logits, params, noHistory);

		assertThat(logits).containsExactly(original);
	}
}
