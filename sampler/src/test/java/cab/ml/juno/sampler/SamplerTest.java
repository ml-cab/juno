package cab.ml.juno.sampler;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

class SamplerTest {

	private final Sampler sampler = Sampler.create();

	@Test
	void greedy_pipeline_returns_highest_logit_token() {
		// Clear winner at index 3
		float[] logits = { 0.1f, 0.5f, 0.3f, 10.0f, 0.2f };
		int token = sampler.sample(logits, SamplingParams.deterministic());
		assertThat(token).isEqualTo(3);
	}

	/**
	 * A request that sets temperature 0 (the REST surfaces only set the temperature, never the greedy
	 * flag) must decode deterministically: the documented meaning of temperature 0. The logits leave
	 * several competitors inside the default top-k / top-p nucleus, so a random draw would frequently
	 * pick something other than the argmax.
	 */
	@Test
	void temperature_zero_selects_the_argmax_without_the_greedy_flag() {
		float[] logits = { 3.0f, 2.9f, 2.8f, 1.0f, 0.5f };
		SamplingParams params = SamplingParams.defaults().withTemperature(0f);
		assertThat(params.greedy()).as("the flag itself is not set").isFalse();

		for (int i = 0; i < 300; i++)
			assertThat(sampler.sample(logits, params, new int[0], new java.util.Random(i))).isEqualTo(0);
	}

	@Test
	void near_zero_temperature_also_selects_the_argmax() {
		float[] logits = { 3.0f, 2.9f, 2.8f, 1.0f, 0.5f };
		SamplingParams params = SamplingParams.defaults().withTemperature(1e-7f);

		for (int i = 0; i < 300; i++)
			assertThat(sampler.sample(logits, params, new int[0], new java.util.Random(i))).isEqualTo(0);
	}

	@Test
	void positive_temperature_still_samples_other_tokens() {
		float[] logits = { 3.0f, 2.9f, 2.8f, 1.0f, 0.5f };
		SamplingParams params = SamplingParams.defaults().withTemperature(0.7f);

		boolean sawOther = false;
		for (int i = 0; i < 300 && !sawOther; i++)
			sawOther = sampler.sample(logits, params, new int[0], new java.util.Random(i)) != 0;
		assertThat(sawOther).as("a non-zero temperature must remain stochastic").isTrue();
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
