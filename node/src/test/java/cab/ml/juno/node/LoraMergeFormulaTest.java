package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.lora.DoraMagnitude;
import cab.ml.juno.lora.DoraProjection;
import cab.ml.juno.lora.LoraAdapter;
import cab.ml.juno.lora.LoraAdapterConfig;
import cab.ml.juno.lora.LoraInitialization;
import cab.ml.juno.lora.LoraMode;
import cab.ml.juno.lora.LoraScaling;

@DisplayName("LoraMerge formulas")
class LoraMergeFormulaTest {

	@Test
	@DisplayName("LoRA merge equals W + scale*B*A")
	void lora_dense_reference() {
		int in = 4, out = 3, rank = 2;
		float[] w = random(out * in, 1);
		float[] orig = w.clone();
		LoraAdapter lora = LoraAdapter.fromWeights(LoraAdapterConfig.legacy(rank, 4f), in, out, random(rank * in, 2),
				random(out * rank, 3));
		LoraMerge.applyAdapter(w, lora, null, out, in);
		for (int r = 0; r < out; r++) {
			for (int c = 0; c < in; c++) {
				float delta = 0f;
				for (int k = 0; k < rank; k++)
					delta += lora.scale * lora.b()[r * rank + k] * lora.a()[k * in + c];
				assertThat(w[r * in + c]).isCloseTo(orig[r * in + c] + delta, within(1e-5f));
			}
		}
	}

	@Test
	@DisplayName("initial DoRA merge with B=0 and mag=row-norms is identity")
	void dora_initial_identity() {
		int in = 5, out = 4;
		float[] w = random(out * in, 10);
		float[] orig = w.clone();
		LoraAdapter lora = new LoraAdapter(
				LoraAdapterConfig.of(2, 2f, LoraScaling.STANDARD, LoraInitialization.LEGACY_NORMAL, LoraMode.DORA), in,
				out, new Random(11));
		DoraMagnitude mag = DoraProjection.magnitudeFromBaseRows(orig, out, in);
		LoraMerge.applyAdapter(w, lora, mag, out, in);
		assertThat(w).containsExactly(orig, within(1e-5f));
	}

	@Test
	@DisplayName("rsLoRA merge uses effective scale from adapter")
	void rslora_uses_adapter_scale() {
		int in = 3, out = 2, rank = 4;
		float[] w = new float[out * in];
		LoraAdapterConfig cfg = LoraAdapterConfig.of(rank, 8f, LoraScaling.RANK_STABILIZED,
				LoraInitialization.LEGACY_NORMAL, LoraMode.LORA);
		float[] a = new float[rank * in];
		float[] b = new float[out * rank];
		b[0] = 1f;
		a[0] = 1f;
		LoraAdapter lora = LoraAdapter.fromWeights(cfg, in, out, a, b);
		LoraMerge.applyAdapter(w, lora, null, out, in);
		assertThat(w[0]).isCloseTo(cfg.effectiveScale(), within(1e-6f));
	}

	private static float[] random(int n, long seed) {
		Random r = new Random(seed);
		float[] v = new float[n];
		for (int i = 0; i < n; i++)
			v[i] = (float) (r.nextGaussian() * 0.2);
		return v;
	}
}
