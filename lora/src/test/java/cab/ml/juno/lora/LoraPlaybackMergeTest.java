package cab.ml.juno.lora;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.lora.LoraPlaybackMerge.ScaledAdapterSet;

@DisplayName("LoraPlaybackMerge")
class LoraPlaybackMergeTest {

	@Test
	@DisplayName("single set at scale 1.0 is returned unchanged (identity)")
	void single_set_scale_one_is_identity() {
		LoraAdapterSet set = setWith(0, "wq", adapter(4, 3, 2, 8f, 1));
		LoraAdapterSet merged = LoraPlaybackMerge.merge(List.of(new ScaledAdapterSet(set, 1.0f, "a.lora")));
		assertThat(merged).isSameAs(set);
	}

	@Test
	@DisplayName("two adapters at different scales sum exactly: delta = s1*d1(x) + s2*d2(x)")
	void two_adapters_sum_reference_math() {
		int in = 5, out = 3;
		LoraAdapter a1 = adapter(in, out, 2, 6f, 11);
		LoraAdapter a2 = adapter(in, out, 3, 9f, 22);
		LoraAdapterSet setA = setWith(0, "wq", a1);
		LoraAdapterSet setB = setWith(0, "wq", a2);

		float scaleA = 0.5f;
		float scaleB = 1.25f;
		LoraAdapterSet merged = LoraPlaybackMerge.merge(
				List.of(new ScaledAdapterSet(setA, scaleA, "a.lora"), new ScaledAdapterSet(setB, scaleB, "b.lora")));

		float[] x = randomVector(in, 99);
		float[] expected = add(scale(a1.forward(x), scaleA), scale(a2.forward(x), scaleB));
		float[] actual = merged.get(0, "wq").forward(x);

		assertThat(actual).hasSize(out);
		for (int i = 0; i < out; i++)
			assertThat(actual[i]).isCloseTo(expected[i], within(1e-4f));
	}

	@Test
	@DisplayName("single-file custom scale still folds the scale in (not treated as identity)")
	void single_set_custom_scale_folds_scale() {
		int in = 4, out = 2;
		LoraAdapter a1 = adapter(in, out, 2, 4f, 5);
		LoraAdapterSet set = setWith(0, "wv", a1);
		LoraAdapterSet merged = LoraPlaybackMerge.merge(List.of(new ScaledAdapterSet(set, 2.0f, "a.lora")));

		float[] x = randomVector(in, 7);
		float[] expected = scale(a1.forward(x), 2.0f);
		float[] actual = merged.get(0, "wv").forward(x);
		for (int i = 0; i < out; i++)
			assertThat(actual[i]).isCloseTo(expected[i], within(1e-4f));
	}

	@Test
	@DisplayName("key present in only one of several sets still gets that set's scale applied")
	void key_unique_to_one_set_gets_its_scale() {
		int in = 3, out = 2;
		LoraAdapter onlyInA = adapter(in, out, 2, 4f, 3);
		LoraAdapterSet setA = setWith(0, "wq", onlyInA);
		LoraAdapterSet setB = new LoraAdapterSet();

		LoraAdapterSet merged = LoraPlaybackMerge
				.merge(List.of(new ScaledAdapterSet(setA, 0.5f, "a.lora"), new ScaledAdapterSet(setB, 1.0f, "b.lora")));

		float[] x = randomVector(in, 4);
		float[] expected = scale(onlyInA.forward(x), 0.5f);
		float[] actual = merged.get(0, "wq").forward(x);
		for (int i = 0; i < out; i++)
			assertThat(actual[i]).isCloseTo(expected[i], within(1e-4f));
	}

	@Test
	@DisplayName("shape mismatch at the same key across sets fails closed")
	void shape_mismatch_fails_closed() {
		LoraAdapterSet setA = setWith(0, "wq", adapter(4, 3, 2, 4f, 1));
		LoraAdapterSet setB = setWith(0, "wq", adapter(5, 3, 2, 4f, 2));
		assertThatThrownBy(() -> LoraPlaybackMerge
				.merge(List.of(new ScaledAdapterSet(setA, 1f, "a.lora"), new ScaledAdapterSet(setB, 1f, "b.lora"))))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("shape mismatch");
	}

	@Test
	@DisplayName("QA-LoRA entries in a multi/scaled playback set fail closed")
	void qa_lora_fails_closed() {
		QaLoraAdapter qa = new QaLoraAdapter(
				LoraAdapterConfig.of(2, 4f, LoraScaling.STANDARD, LoraInitialization.KAIMING_UNIFORM, LoraMode.QA_LORA),
				4, 3, 2, new Random(1));
		LoraAdapterSet set = new LoraAdapterSet();
		set.addQa(0, "wq", qa,
				new QaLoraEntryMeta(2, 2, 12, "enc", MergeCapability.SIDECAR_ONLY, QaLoraAdapter.PoolingOp.SUM));
		assertThatThrownBy(() -> LoraPlaybackMerge.merge(List.of(new ScaledAdapterSet(set, 2.0f, "a.lora"))))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("QA-LoRA");
	}

	@Test
	@DisplayName("empty list fails closed")
	void empty_list_fails_closed() {
		assertThatThrownBy(() -> LoraPlaybackMerge.merge(List.of())).isInstanceOf(IllegalArgumentException.class);
	}

	// ── helpers ────────────────────────────────────────────────────────────

	private static LoraAdapterSet setWith(int layer, String proj, LoraAdapter adapter) {
		LoraAdapterSet set = new LoraAdapterSet();
		set.add(layer, proj, adapter);
		return set;
	}

	private static LoraAdapter adapter(int in, int out, int rank, float alpha, long seed) {
		return LoraAdapter.fromWeights(LoraAdapterConfig.of(rank, alpha), in, out, randomVector(rank * in, seed),
				randomVector(out * rank, seed + 1000));
	}

	private static float[] randomVector(int n, long seed) {
		Random r = new Random(seed);
		float[] v = new float[n];
		for (int i = 0; i < n; i++)
			v[i] = (float) (r.nextGaussian() * 0.2);
		return v;
	}

	private static float[] scale(float[] v, float s) {
		float[] out = new float[v.length];
		for (int i = 0; i < v.length; i++)
			out[i] = v[i] * s;
		return out;
	}

	private static float[] add(float[] a, float[] b) {
		float[] out = new float[a.length];
		for (int i = 0; i < a.length; i++)
			out[i] = a[i] + b[i];
		return out;
	}
}
