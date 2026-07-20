package cab.ml.juno.lora;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("LoraLearningRateSchedule")
class LoraLearningRateScheduleTest {

	@Test
	@DisplayName("constant schedule returns the configured rate")
	void constant_rate() {
		LoraLearningRateSchedule s = LoraLearningRateSchedule.constant(1e-4);
		assertThat(s.learningRate(1)).isEqualTo(1e-4);
		assertThat(s.learningRate(100)).isEqualTo(1e-4);
		assertThat(s.mode()).isEqualTo(LoraLearningRateSchedule.Mode.CONSTANT);
	}

	@Test
	@DisplayName("warmup is linear from first update to peak")
	void warmup_boundaries() {
		LoraLearningRateSchedule s = LoraLearningRateSchedule.warmupCosine(1e-3, 1e-5, 4, 20);
		assertThat(s.learningRate(1)).isCloseTo(1e-3 * 0.25, within(1e-12));
		assertThat(s.learningRate(2)).isCloseTo(1e-3 * 0.5, within(1e-12));
		assertThat(s.learningRate(4)).isCloseTo(1e-3, within(1e-12));
	}

	@Test
	@DisplayName("cosine midpoint and end match the closed-form schedule")
	void cosine_midpoint_and_end() {
		LoraLearningRateSchedule s = LoraLearningRateSchedule.warmupCosine(1.0, 0.0, 0, 5);
		// updates 1..5, cosineSpan=5, progress at update 3 = 2/4 = 0.5
		double mid = 0.5 * (1.0 + Math.cos(Math.PI * 0.5));
		assertThat(s.learningRate(3)).isCloseTo(mid, within(1e-12));
		assertThat(s.learningRate(5)).isCloseTo(0.0, within(1e-12));
	}

	@Test
	@DisplayName("no warmup starts cosine immediately at peak for update 1")
	void no_warmup() {
		LoraLearningRateSchedule s = LoraLearningRateSchedule.warmupCosine(2e-4, 1e-5, 0, 10);
		assertThat(s.learningRate(1)).isCloseTo(2e-4, within(1e-12));
	}

	@Test
	@DisplayName("rates clamp at minimum after totalUpdates")
	void clamps_after_total() {
		LoraLearningRateSchedule s = LoraLearningRateSchedule.warmupCosine(1e-3, 1e-5, 2, 8);
		assertThat(s.learningRate(8)).isCloseTo(1e-5, within(1e-12));
		assertThat(s.learningRate(100)).isCloseTo(1e-5, within(1e-12));
	}

	@Test
	@DisplayName("identical calls are deterministic")
	void deterministic_calls() {
		LoraLearningRateSchedule s = LoraLearningRateSchedule.warmupCosine(1e-3, 1e-5, 3, 30);
		double a = s.learningRate(17);
		double b = s.learningRate(17);
		assertThat(a).isEqualTo(b);
	}

	@Test
	@DisplayName("rejects invalid parameters")
	void invalid_parameters() {
		assertThatThrownBy(() -> LoraLearningRateSchedule.constant(-1))
				.isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> LoraLearningRateSchedule.constant(Double.NaN))
				.isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> LoraLearningRateSchedule.warmupCosine(1e-3, 2e-3, 0, 10))
				.isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> LoraLearningRateSchedule.warmupCosine(1e-3, 0, -1, 10))
				.isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> LoraLearningRateSchedule.warmupCosine(1e-3, 0, 5, 4))
				.isInstanceOf(IllegalArgumentException.class);
		LoraLearningRateSchedule s = LoraLearningRateSchedule.constant(1e-4);
		assertThatThrownBy(() -> s.learningRate(0)).isInstanceOf(IllegalArgumentException.class);
	}
}
