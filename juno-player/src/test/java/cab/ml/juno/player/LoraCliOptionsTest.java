package cab.ml.juno.player;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.node.LoraProjection;
import cab.ml.juno.lora.LoraLearningRateSchedule;

@DisplayName("LoraCliOptions")
class LoraCliOptionsTest {

	@Test
	@DisplayName("parses Tier-1 LoRA flags")
	void parses_tier1_flags() {
		LoraCliOptions o = new LoraCliOptions();
		String[] args = { "--lora-targets", "all", "--lora-gradient-accumulation", "4", "--lora-max-grad-norm",
				"0.5" };
		int i = 0;
		i = o.applyFlag(args, i);
		assertThat(i).isEqualTo(1);
		i = o.applyFlag(args, i + 1);
		assertThat(i).isEqualTo(3);
		i = o.applyFlag(args, i + 1);
		assertThat(i).isEqualTo(5);
		assertThat(o.targets).isEqualTo("all");
		assertThat(o.gradientAccumulation).isEqualTo(4);
		assertThat(o.maxGradNorm).isEqualTo(0.5f);
		assertThat(o.parsedTargets()).isEqualTo(LoraProjection.allLinear());
	}

	@Test
	@DisplayName("rejects invalid targets")
	void rejects_bad_targets() {
		LoraCliOptions o = new LoraCliOptions();
		assertThatThrownBy(() -> o.applyFlag(new String[] { "--lora-targets", "nope" }, 0))
				.isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	@DisplayName("toTrainingConfig maps fields")
	void to_training_config() {
		LoraCliOptions o = new LoraCliOptions();
		o.rank = 4;
		o.alpha = 8f;
		o.lr = 2e-4;
		o.targets = "wq,wv,wo";
		o.gradientAccumulation = 2;
		o.maxGradNorm = 1.5f;
		LoraTrainingConfig c = o.toTrainingConfig();
		assertThat(c.rank()).isEqualTo(4);
		assertThat(c.alpha()).isEqualTo(8f);
		assertThat(c.gradientAccumulationSteps()).isEqualTo(2);
		assertThat(c.maxGradNorm()).isEqualTo(1.5f);
		assertThat(c.targets()).containsExactly(LoraProjection.WQ, LoraProjection.WV, LoraProjection.WO);
	}

	@Test
	@DisplayName("parses Tier-2 schedule and LoRA+ flags")
	void parses_tier2_flags() {
		LoraCliOptions o = new LoraCliOptions();
		String[] args = { "--lora-lr-schedule", "cosine", "--lora-warmup-steps", "10", "--lora-min-lr", "1e-6",
				"--lora-weight-decay", "0.05", "--lora-plus-ratio", "4", "--lora-dropout", "0.1", "--lora-seed", "7",
				"--lora-validation-split", "0.25", "--lora-validation-patience", "3", "--lora-validation-min-delta",
				"0.01" };
		for (int i = 0; i < args.length;) {
			int n = o.applyFlag(args, i);
			assertThat(n).isGreaterThan(i);
			i = n + 1;
		}
		LoraTrainingConfig c = o.toTrainingConfig();
		assertThat(c.lrSchedule()).isEqualTo(LoraLearningRateSchedule.Mode.COSINE);
		assertThat(c.warmupUpdates()).isEqualTo(10);
		assertThat(c.minLearningRate()).isEqualTo(1e-6);
		assertThat(c.weightDecay()).isEqualTo(0.05);
		assertThat(c.loraPlusRatio()).isEqualTo(4.0);
		assertThat(c.dropout()).isEqualTo(0.1f);
		assertThat(c.seed()).isEqualTo(7L);
		assertThat(c.validationSplit()).isEqualTo(0.25f);
		assertThat(c.validationPatience()).isEqualTo(3);
		assertThat(c.validationMinDelta()).isEqualTo(0.01f);
	}

	@Test
	@DisplayName("parses Tier-8 chunk and corpus-cap flags")
	void parses_tier8_flags() {
		LoraCliOptions o = new LoraCliOptions();
		String[] args = { "--lora-chunk-tokens", "128", "--lora-max-train-tokens", "2048" };
		for (int i = 0; i < args.length;) {
			int n = o.applyFlag(args, i);
			assertThat(n).isGreaterThan(i);
			i = n + 1;
		}
		assertThat(o.chunkTokens).isEqualTo(128);
		assertThat(o.maxTrainTokens).isEqualTo(2048);
		LoraTrainingConfig c = o.toTrainingConfig();
		assertThat(c.chunkTokens()).isEqualTo(128);
		assertThat(c.maxTrainTokens()).isEqualTo(2048);
	}

	@Test
	@DisplayName("rejects invalid Tier-8 chunk and corpus-cap values")
	void rejects_bad_tier8_bounds() {
		LoraCliOptions o = new LoraCliOptions();
		assertThatThrownBy(() -> o.applyFlag(new String[] { "--lora-chunk-tokens", "0" }, 0))
				.isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> o.applyFlag(new String[] { "--lora-chunk-tokens", "8193" }, 0))
				.isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> o.applyFlag(new String[] { "--lora-max-train-tokens", "-1" }, 0))
				.isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	@DisplayName("parses Tier-9 train-device flag")
	void parses_tier9_train_device() {
		LoraCliOptions o = new LoraCliOptions();
		assertThat(o.trainDevice).isEqualTo("auto");
		int n = o.applyFlag(new String[] { "--lora-train-device", "cpu" }, 0);
		assertThat(n).isEqualTo(1);
		assertThat(o.trainDevice).isEqualTo("cpu");
		assertThat(o.toTrainingConfig().trainDevice()).isEqualTo("cpu");
		assertThatThrownBy(() -> o.applyFlag(new String[] { "--lora-train-device", "tpu" }, 0))
				.isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	@DisplayName("parses Tier-11 microbatch flag")
	void parses_tier11_microbatch() {
		LoraCliOptions o = new LoraCliOptions();
		assertThat(o.microbatch).isEqualTo(8);
		int n = o.applyFlag(new String[] { "--lora-microbatch", "1" }, 0);
		assertThat(n).isEqualTo(1);
		assertThat(o.microbatch).isEqualTo(1);
		assertThat(o.toTrainingConfig().microbatch()).isEqualTo(1);
		assertThatThrownBy(() -> o.applyFlag(new String[] { "--lora-microbatch", "0" }, 0))
				.isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> o.applyFlag(new String[] { "--lora-microbatch", "129" }, 0))
				.isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	@DisplayName("Tier-8 defaults are 32 chunk and unlimited tokens")
	void tier8_defaults() {
		LoraCliOptions o = new LoraCliOptions();
		LoraTrainingConfig c = o.toTrainingConfig();
		assertThat(c.chunkTokens()).isEqualTo(32);
		assertThat(c.maxTrainTokens()).isEqualTo(0);
		assertThat(c.trainDevice()).isEqualTo("auto");
		assertThat(c.microbatch()).isEqualTo(8);
	}

	@Test
	@DisplayName("formats friendly LoRA train status line")
	void formats_lora_train_status() {
		assertThat(ConsoleMain.formatLoraTrainStatus("auto", "cuda", 1))
				.isEqualTo("Training on CUDA (auto-selected) · microbatch size 1");
		assertThat(ConsoleMain.formatLoraTrainStatus("gpu", "cuda", 8))
				.isEqualTo("Training on CUDA · microbatch size 8");
		assertThat(ConsoleMain.formatLoraTrainStatus("cpu", "cpu", 8))
				.isEqualTo("Training on CPU · microbatch size 8");
		assertThat(ConsoleMain.formatLoraTrainStatus("auto", "rocm", 8))
				.isEqualTo("Training on ROCm (auto-selected) · microbatch size 8");
	}
}
