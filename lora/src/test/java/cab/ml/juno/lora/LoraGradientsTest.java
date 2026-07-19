package cab.ml.juno.lora;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("LoraGradients")
class LoraGradientsTest {

	@Test
	@DisplayName("exact L2 norm after token normalization")
	void exact_norm_after_normalization() {
		LoraAdapterSet set = singleAdapter(2, 2, 2);
		LoraAdapter a = set.all().get(0);
		// gradA = [3, 4] → ||g||=5; gradB all zero
		a.gradA()[0] = 3f;
		a.gradA()[1] = 4f;

		LoraGradients.PrepResult r = LoraGradients.prepare(set, 2, 0f);
		assertThat(r.globalNorm()).isCloseTo(2.5, within(1e-6));
		assertThat(r.scale()).isCloseTo(0.5f, within(1e-6f));
		assertThat(r.clipped()).isFalse();
		assertThat(a.gradA()[0]).isCloseTo(1.5f, within(1e-6f));
		assertThat(a.gradA()[1]).isCloseTo(2.0f, within(1e-6f));
	}

	@Test
	@DisplayName("clipping scales jointly when norm exceeds maxNorm")
	void clipping_applies_when_norm_exceeds_max() {
		LoraAdapterSet set = singleAdapter(1, 1, 1);
		LoraAdapter a = set.all().get(0);
		a.gradA()[0] = 10f; // raw ||g||=10; after /1 → 10; clip to 2 → scale 0.2

		LoraGradients.PrepResult r = LoraGradients.prepare(set, 1, 2f);
		assertThat(r.globalNorm()).isCloseTo(10.0, within(1e-6));
		assertThat(r.clipped()).isTrue();
		assertThat(r.scale()).isCloseTo(0.2f, within(1e-6f));
		assertThat(a.gradA()[0]).isCloseTo(2f, within(1e-6f));
	}

	@Test
	@DisplayName("maxNorm == 0 disables clipping but still normalizes")
	void zero_max_norm_disables_clipping() {
		LoraAdapterSet set = singleAdapter(1, 1, 1);
		LoraAdapter a = set.all().get(0);
		a.gradA()[0] = 8f;

		LoraGradients.PrepResult r = LoraGradients.prepare(set, 4, 0f);
		assertThat(r.clipped()).isFalse();
		assertThat(r.scale()).isCloseTo(0.25f, within(1e-6f));
		assertThat(a.gradA()[0]).isCloseTo(2f, within(1e-6f));
	}

	@Test
	@DisplayName("all-zero gradients yield zero norm and unit-free scale")
	void zero_gradients() {
		LoraAdapterSet set = singleAdapter(2, 2, 2);
		LoraGradients.PrepResult r = LoraGradients.prepare(set, 3, 1f);
		assertThat(r.globalNorm()).isEqualTo(0.0);
		assertThat(r.clipped()).isFalse();
		assertThat(r.scale()).isCloseTo(1f / 3f, within(1e-6f));
	}

	@Test
	@DisplayName("NaN gradient is rejected before mutation completes")
	void rejects_nan() {
		LoraAdapterSet set = singleAdapter(1, 1, 1);
		set.all().get(0).gradA()[0] = Float.NaN;
		assertThatThrownBy(() -> LoraGradients.prepare(set, 1, 1f)).isInstanceOf(IllegalStateException.class)
				.hasMessageContaining("non-finite");
	}

	@Test
	@DisplayName("Inf gradient is rejected")
	void rejects_inf() {
		LoraAdapterSet set = singleAdapter(1, 1, 1);
		set.all().get(0).gradB()[0] = Float.POSITIVE_INFINITY;
		assertThatThrownBy(() -> LoraGradients.prepare(set, 1, 0f)).isInstanceOf(IllegalStateException.class);
	}

	@Test
	@DisplayName("predictionCount < 1 is rejected")
	void rejects_bad_prediction_count() {
		LoraAdapterSet set = singleAdapter(1, 1, 1);
		assertThatThrownBy(() -> LoraGradients.prepare(set, 0, 1f)).isInstanceOf(IllegalArgumentException.class);
	}

	private static LoraAdapterSet singleAdapter(int rank, int inDim, int outDim) {
		LoraAdapterSet set = new LoraAdapterSet();
		set.add(0, "wq", new LoraAdapter(rank, inDim, outDim, rank, new Random(1)));
		set.zeroAllGrads();
		return set;
	}
}
