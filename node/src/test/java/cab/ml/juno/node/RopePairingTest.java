/*
 * Copyright 2026 Dmytro Soloviov (soulaway)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.Random;

import org.junit.jupiter.api.Test;

/**
 * The two RoPE pairings differ only in which dimensions form a rotated pair:
 * adjacent {@code (x[2i], x[2i+1])} or split-half {@code (x[i], x[i+d/2])}. Pair
 * {@code i} uses the same frequency in both. So split-half applied to a head is
 * adjacent applied to the same head with its halves interleaved, then
 * de-interleaved - bit for bit, since the arithmetic per pair is identical.
 */
class RopePairingTest {

	private static final int HEAD_DIM = 64;
	private static final int HEADS = 4;
	private static final float THETA = 1_000_000f;

	@Test
	void adjacent_overload_is_bit_identical_to_the_legacy_rotation() {
		float[] x = random(HEADS * HEAD_DIM, 1);
		float[] legacy = x.clone();
		float[] viaPairing = x.clone();
		LlamaTransformerHandler.rope(legacy, 777, HEADS, HEAD_DIM, THETA);
		LlamaTransformerHandler.rope(viaPairing, 777, HEADS, HEAD_DIM, THETA, RopePairing.ADJACENT);
		assertThat(viaPairing).containsExactly(legacy);
	}

	@Test
	void split_half_is_adjacent_on_interleaved_halves() {
		for (int pos : new int[] { 0, 1, 17, 4095, 32767 }) {
			float[] x = random(HEADS * HEAD_DIM, pos);
			float[] split = x.clone();
			LlamaTransformerHandler.rope(split, pos, HEADS, HEAD_DIM, THETA, RopePairing.SPLIT_HALF);

			float[] interleaved = interleave(x);
			LlamaTransformerHandler.rope(interleaved, pos, HEADS, HEAD_DIM, THETA, RopePairing.ADJACENT);

			assertThat(split).as("pos %d", pos).containsExactly(deinterleave(interleaved));
		}
	}

	@Test
	void split_half_rotates_something_at_a_nonzero_position() {
		float[] x = random(HEADS * HEAD_DIM, 3);
		float[] adjacent = x.clone();
		float[] split = x.clone();
		LlamaTransformerHandler.rope(adjacent, 100, HEADS, HEAD_DIM, THETA, RopePairing.ADJACENT);
		LlamaTransformerHandler.rope(split, 100, HEADS, HEAD_DIM, THETA, RopePairing.SPLIT_HALF);
		assertThat(split).isNotEqualTo(adjacent).isNotEqualTo(x);
	}

	@Test
	void production_layout_per_architecture() {
		// Q/K rows are permuted for adjacent pairs only by the LLaMA conversion; the
		// Qwen conversions leave them in the rotate-half layout.
		for (String arch : new String[] { "llama", "mistral", "tinyllama" })
			assertThat(LlamaTransformerHandler.ropePairingFor(config(arch))).as(arch).isEqualTo(RopePairing.ADJACENT);
		for (String arch : new String[] { "qwen2", "qwen2.5" })
			assertThat(LlamaTransformerHandler.ropePairingFor(config(arch))).as(arch)
					.isEqualTo(RopePairing.SPLIT_HALF);
		assertThat(Qwen3RopeConfig.standard(qwen3Base()).pairing()).isEqualTo(RopePairing.SPLIT_HALF);
		assertThat(Qwen3RopeConfig.PAIRING).isEqualTo(RopePairing.SPLIT_HALF);
	}

	@Test
	void split_half_backward_is_the_adjoint_of_the_forward() {
		for (int pos : new int[] { 1, 250, 30000 }) {
			float[] x = random(HEADS * HEAD_DIM, pos);
			float[] g = random(HEADS * HEAD_DIM, pos + 1);
			float[] rx = x.clone();
			LlamaTransformerHandler.rope(rx, pos, HEADS, HEAD_DIM, THETA, RopePairing.SPLIT_HALF);
			float[] rtg = g.clone();
			LoraTrainingMath.ropeBackward(rtg, pos, HEADS, HEAD_DIM, THETA, RopePairing.SPLIT_HALF);
			assertThat(dot(rx, g)).as("pos %d", pos).isCloseTo(dot(x, rtg), org.assertj.core.api.Assertions.within(1e-3));
		}
	}

	@Test
	void adjacent_backward_overload_is_the_legacy_backward() {
		float[] g = random(HEADS * HEAD_DIM, 11);
		float[] legacy = g.clone();
		float[] viaPairing = g.clone();
		LoraTrainingMath.ropeBackward(legacy, 99, HEADS, HEAD_DIM, THETA);
		LoraTrainingMath.ropeBackward(viaPairing, 99, HEADS, HEAD_DIM, THETA, RopePairing.ADJACENT);
		assertThat(viaPairing).containsExactly(legacy);
	}

	@Test
	void qwen3_backward_is_the_adjoint_of_the_forward_in_both_layouts_with_and_without_yarn() {
		Qwen3RopeConfig plain = Qwen3RopeConfig.standard(qwen3Base());
		Qwen3RopeConfig yarn = new Qwen3RopeConfig(THETA, 0.25f, 1.0f, 32768, 131072, true, RopePairing.SPLIT_HALF);
		for (Qwen3RopeConfig c : new Qwen3RopeConfig[] { plain, plain.withPairing(RopePairing.ADJACENT), yarn,
				yarn.withPairing(RopePairing.ADJACENT) }) {
			float[] x = random(HEADS * HEAD_DIM, 21);
			float[] g = random(HEADS * HEAD_DIM, 22);
			float[] rx = x.clone();
			Qwen3Rope.apply(rx, 40_000, HEADS, HEAD_DIM, c);
			float[] rtg = g.clone();
			Qwen3Rope.applyBackward(rtg, 40_000, HEADS, HEAD_DIM, c);
			float[] rtg2 = g.clone();
			LoraTrainingMath.qwen3RopeBackward(rtg2, 40_000, HEADS, HEAD_DIM, c);
			assertThat(dot(rx, g)).as("%s", c).isCloseTo(dot(x, rtg), org.assertj.core.api.Assertions.within(1e-3));
			assertThat(dot(rx, g)).as("%s via LoraTrainingMath", c).isCloseTo(dot(x, rtg2),
					org.assertj.core.api.Assertions.within(1e-3));
		}
	}

	@Test
	void qwen3_rope_dispatches_on_the_config_pairing_without_yarn() {
		LlamaConfig base = qwen3Base();
		Qwen3RopeConfig adjacent = Qwen3RopeConfig.standard(base).withPairing(RopePairing.ADJACENT);
		Qwen3RopeConfig split = adjacent.withPairing(RopePairing.SPLIT_HALF);

		float[] x = random(HEADS * HEAD_DIM, 5);
		float[] viaQwen = x.clone();
		float[] direct = x.clone();
		Qwen3Rope.apply(viaQwen, 321, HEADS, HEAD_DIM, split);
		LlamaTransformerHandler.rope(direct, 321, HEADS, HEAD_DIM, base.ropeTheta(), RopePairing.SPLIT_HALF);
		assertThat(viaQwen).containsExactly(direct);
	}

	@Test
	void qwen3_yarn_split_half_is_yarn_adjacent_on_interleaved_halves() {
		Qwen3RopeConfig yarnAdjacent = new Qwen3RopeConfig(THETA, 0.25f, 1.0f, 32768, 131072, true,
				RopePairing.ADJACENT);
		Qwen3RopeConfig yarnSplit = yarnAdjacent.withPairing(RopePairing.SPLIT_HALF);

		float[] x = random(HEADS * HEAD_DIM, 9);
		float[] split = x.clone();
		Qwen3Rope.apply(split, 50_000, HEADS, HEAD_DIM, yarnSplit);
		float[] interleaved = interleave(x);
		Qwen3Rope.apply(interleaved, 50_000, HEADS, HEAD_DIM, yarnAdjacent);
		assertThat(split).containsExactly(deinterleave(interleaved));
	}

	private static LlamaConfig config(String arch) {
		return new LlamaConfig(HEADS * HEAD_DIM, 2, HEADS, HEADS, HEAD_DIM, 4 * HEADS * HEAD_DIM, 1000, 1e-6f,
				THETA, arch);
	}

	private static double dot(float[] a, float[] b) {
		double d = 0;
		for (int i = 0; i < a.length; i++)
			d += (double) a[i] * b[i];
		return d;
	}

	private static LlamaConfig qwen3Base() {
		return new LlamaConfig(HEADS * HEAD_DIM, 2, HEADS, HEADS, HEAD_DIM, 4 * HEADS * HEAD_DIM, 1000, 1e-6f,
				THETA, "qwen3");
	}

	/** Per head: split-half layout [a0..a(h-1), b0..b(h-1)] to [a0, b0, a1, b1, ...]. */
	private static float[] interleave(float[] x) {
		float[] out = new float[x.length];
		int half = HEAD_DIM / 2;
		for (int h = 0; h < HEADS; h++) {
			int base = h * HEAD_DIM;
			for (int i = 0; i < half; i++) {
				out[base + 2 * i] = x[base + i];
				out[base + 2 * i + 1] = x[base + i + half];
			}
		}
		return out;
	}

	private static float[] deinterleave(float[] x) {
		float[] out = new float[x.length];
		int half = HEAD_DIM / 2;
		for (int h = 0; h < HEADS; h++) {
			int base = h * HEAD_DIM;
			for (int i = 0; i < half; i++) {
				out[base + i] = x[base + 2 * i];
				out[base + i + half] = x[base + 2 * i + 1];
			}
		}
		return out;
	}

	private static float[] random(int n, long seed) {
		Random rng = new Random(seed);
		float[] a = new float[n];
		for (int i = 0; i < n; i++)
			a[i] = (float) rng.nextGaussian();
		return a;
	}
}
