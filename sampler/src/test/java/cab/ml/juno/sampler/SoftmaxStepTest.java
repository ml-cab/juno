package cab.ml.juno.sampler;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Test;

class SoftmaxStepTest {

	private final SoftmaxStep step = SoftmaxStep.INSTANCE;
	private final int[] noHistory = new int[0];
	private final SamplingParams params = SamplingParams.defaults();

	@Test
	void output_probabilities_sum_to_one() {
		float[] logits = { 1.0f, 2.0f, 3.0f, 4.0f };
		step.apply(logits, params, noHistory);

		float sum = 0.0f;
		for (float v : logits)
			sum += v;
		assertThat(sum).isCloseTo(1.0f, within(1e-5f));
	}

	@Test
	void all_probabilities_are_positive() {
		float[] logits = { -100.0f, 0.0f, 100.0f };
		step.apply(logits, params, noHistory);

		for (float v : logits) {
			assertThat(v).isGreaterThanOrEqualTo(0.0f);
		}
	}

	@Test
	void higher_logit_gets_higher_probability() {
		float[] logits = { 1.0f, 3.0f, 2.0f };
		step.apply(logits, params, noHistory);

		// index 1 had highest logit → highest probability
		assertThat(logits[1]).isGreaterThan(logits[2]);
		assertThat(logits[2]).isGreaterThan(logits[0]);
	}

	@Test
	void numerically_stable_with_large_logits() {
		// Without subtracting max, exp(1000) would overflow to Infinity
		float[] logits = { 1000.0f, 1001.0f, 1002.0f };
		assertThatCode(() -> step.apply(logits, params, noHistory)).doesNotThrowAnyException();

		float sum = 0.0f;
		for (float v : logits)
			sum += v;
		assertThat(sum).isCloseTo(1.0f, within(1e-5f));
	}
}
