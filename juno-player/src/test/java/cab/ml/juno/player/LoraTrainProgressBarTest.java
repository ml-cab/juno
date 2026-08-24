package cab.ml.juno.player;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("LoraTrainProgressBar")
class LoraTrainProgressBarTest {

	@Test
	@DisplayName("percent is 0 before baseline (second pass) is set")
	void percent_zero_without_baseline() {
		assertThat(LoraTrainProgressBar.percentTowardTarget(Float.NaN, 3.0f, 1.2f)).isEqualTo(0);
		assertThat(LoraTrainProgressBar.percentTowardTarget(3.0f, Float.NaN, 1.2f)).isEqualTo(0);
	}

	@Test
	@DisplayName("percent measures closed fraction of baseline-to-target gap")
	void percent_from_loss_gap() {
		// baseline=3.0, target=1.0, span=2.0; current=2.0 → halfway
		assertThat(LoraTrainProgressBar.percentTowardTarget(3.0f, 2.0f, 1.0f)).isEqualTo(50);
		// current at baseline → 0%
		assertThat(LoraTrainProgressBar.percentTowardTarget(3.0f, 3.0f, 1.0f)).isEqualTo(0);
		// current at/below target → 100%
		assertThat(LoraTrainProgressBar.percentTowardTarget(3.0f, 1.0f, 1.0f)).isEqualTo(100);
		assertThat(LoraTrainProgressBar.percentTowardTarget(3.0f, 0.5f, 1.0f)).isEqualTo(100);
	}

	@Test
	@DisplayName("percent is 100 when current already at target even without baseline")
	void percent_target_hit_without_baseline() {
		assertThat(LoraTrainProgressBar.percentTowardTarget(Float.NaN, 1.0f, 1.2f)).isEqualTo(100);
		assertThat(LoraTrainProgressBar.percentTowardTarget(Float.NaN, 0.9f, 1.2f)).isEqualTo(100);
	}

	@Test
	@DisplayName("loss rise after baseline does not go negative")
	void percent_clamped_when_loss_rises() {
		assertThat(LoraTrainProgressBar.percentTowardTarget(2.0f, 2.5f, 1.0f)).isEqualTo(0);
	}

	@Test
	@DisplayName("filled bar count matches percentage")
	void filled_bars_match_percent() {
		assertThat(LoraTrainProgressBar.filledBars(24, 20)).isEqualTo(5);
		assertThat(LoraTrainProgressBar.filledBars(50, 20)).isEqualTo(10);
		assertThat(LoraTrainProgressBar.filledBars(100, 20)).isEqualTo(20);
		assertThat(LoraTrainProgressBar.filledBars(0, 20)).isEqualTo(0);
	}

	@Test
	@DisplayName("ETA from loss improvement rate since baseline")
	void eta_from_loss_rate() {
		// baseline 3→2 over 10s, remaining 2→1 → another 10s
		assertThat(LoraTrainProgressBar.etaMs(3.0f, 2.0f, 1.0f, 10_000L)).isEqualTo(10_000L);
		assertThat(LoraTrainProgressBar.etaMs(3.0f, 1.0f, 1.0f, 10_000L)).isEqualTo(0L);
		assertThat(LoraTrainProgressBar.etaMs(3.0f, 3.0f, 1.0f, 10_000L)).isEqualTo(0L);
		assertThat(LoraTrainProgressBar.etaMs(Float.NaN, 2.0f, 1.0f, 10_000L)).isEqualTo(0L);
	}

	@Test
	@DisplayName("ETA formats minutes and seconds")
	void eta_format() {
		assertThat(LoraTrainProgressBar.formatEta(0)).isEqualTo("0s");
		assertThat(LoraTrainProgressBar.formatEta(5_000)).isEqualTo("5s");
		assertThat(LoraTrainProgressBar.formatEta(125_000)).isEqualTo("2m05s");
	}

	@Test
	@DisplayName("render uses loss progress, not max passes")
	void render_loss_based_no_max_passes() {
		// baseline 3.0, current 1.0162, target 1.2 → past target → 100%
		String done = LoraTrainProgressBar.render(12, 1.0162f, 1.2f, 3.0f, 3754, 45_000);
		assertThat(done).startsWith("\r  pass ");
		assertThat(done).contains("pass  12");
		assertThat(done).doesNotContain("/50");
		assertThat(done).contains("100%");
		assertThat(done).contains("▓".repeat(20));
		assertThat(done).endsWith("\033[K");

		// baseline 3.0, current 2.1, target 1.2 → (3-2.1)/(3-1.2)=0.5 → 50%
		String mid = LoraTrainProgressBar.render(5, 2.1f, 1.2f, 3.0f, 3000, 12_000);
		assertThat(mid).contains("50%");
		assertThat(mid).contains("▓".repeat(10));
		assertThat(mid).doesNotContain("/50");
		assertThat(mid).doesNotContain("pass   5/");
	}
}
