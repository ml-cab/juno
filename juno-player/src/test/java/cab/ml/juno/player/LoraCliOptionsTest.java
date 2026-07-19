package cab.ml.juno.player;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.node.LoraProjection;

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
}
