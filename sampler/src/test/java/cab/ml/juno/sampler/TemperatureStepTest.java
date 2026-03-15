package cab.ml.juno.sampler;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.sampler.TemperatureStep;

class TemperatureStepTest {

	private final TemperatureStep step = TemperatureStep.INSTANCE;
	private final int[] noHistory = new int[0];

	@Test
	void scales_logits_by_temperature() {
		float[] logits = { 2.0f, 4.0f, 6.0f };
		SamplingParams params = SamplingParams.defaults().withTemperature(2.0f);

		step.apply(logits, params, noHistory);

		assertThat(logits).containsExactly(1.0f, 2.0f, 3.0f);
	}

	@Test
	void temperature_one_leaves_logits_unchanged() {
		float[] logits = { 1.0f, 2.0f, 3.0f };
		float[] original = logits.clone();
		SamplingParams params = SamplingParams.defaults().withTemperature(1.0f);

		step.apply(logits, params, noHistory);

		assertThat(logits).containsExactly(original);
	}

	@Test
	void skips_scaling_in_greedy_mode() {
		float[] logits = { 2.0f, 4.0f, 6.0f };
		float[] original = logits.clone();
		SamplingParams params = SamplingParams.deterministic(); // greedy=true

		step.apply(logits, params, noHistory);

		assertThat(logits).containsExactly(original);
	}
}
