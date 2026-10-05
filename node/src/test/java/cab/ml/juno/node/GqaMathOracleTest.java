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

import java.util.Arrays;
import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.DenseKvTensor;

/**
 * Checks the scalar attention path itself, which every GPU attention parity test
 * uses as its oracle ({@link GqaMath#attend}). Those tests compare the kernel
 * against this method, so a defect shared by both would pass them; here the
 * method is held to an independently written double-precision reference and to
 * properties that do not depend on any implementation: a single key returns its
 * value row, equal scores average the value rows, grouped heads read only their
 * own KV head, and positions at or past {@code seqLen} are never read.
 *
 * <p>Shapes are those of the sweep models and of the Qwen3 and Phi-3 handlers
 * (grouped and plain multi-head, power-of-two and 96-wide heads); lengths run
 * from one key to {@link DenseKvTensor#MAX_SEQ_LEN}. A replacement kernel is
 * validated against this oracle at the same lengths.
 */
@DisplayName("GqaMath.attend: scalar attention oracle vs an independent reference")
class GqaMathOracleTest {

	/**
	 * Absolute tolerance on outputs whose value rows lie in [-1, 1]: a fixed part
	 * for the per-dot rounding, plus a part growing with the square root of the
	 * number of summed keys (the float softmax sum and the float weighted-V sum).
	 */
	private static double tolerance(int seqLen) {
		return 2e-6 + 2e-6 * Math.sqrt(seqLen);
	}

	private record Shape(String name, int numHeads, int numKvHeads, int headDim) {
		int gqaRatio() {
			return numHeads / numKvHeads;
		}

		int kvDim() {
			return numKvHeads * headDim;
		}

		int qDim() {
			return numHeads * headDim;
		}
	}

	private static final Shape TINYLLAMA = new Shape("TinyLlama (32 heads, 4 KV, 64 wide)", 32, 4, 64);
	private static final Shape[] SHAPES = {
			TINYLLAMA,
			new Shape("Mistral 7B (32 heads, 8 KV, 128 wide)", 32, 8, 128),
			new Shape("Qwen2.5-3B (16 heads, 2 KV, 128 wide)", 16, 2, 128),
			new Shape("Phi-3.5-mini (32 heads, plain multi-head, 96 wide)", 32, 32, 96),
			new Shape("Qwen3-1.7B (16 heads, 8 KV, 128 wide)", 16, 8, 128),
	};

	/** Lengths below the long case: one key, a non-power of two, the sweep's prompt lengths. */
	private static final int[] LENGTHS = { 1, 2, 17, 128, 513, 2048 };

	@Test
	@DisplayName("matches a double-precision reference on every shape, one key to 2048 keys")
	void matchesReferenceAcrossShapesAndLengths() {
		for (Shape shape : SHAPES)
			for (int seqLen : LENGTHS)
				for (float qScale : new float[] { 1f, 6f })
					assertMatchesReference(shape, seqLen, qScale, 31L * seqLen + shape.name().hashCode());
	}

	@Test
	@DisplayName("matches a double-precision reference at MAX_SEQ_LEN")
	void matchesReferenceAtMaxSeqLen() {
		assertMatchesReference(TINYLLAMA, DenseKvTensor.MAX_SEQ_LEN, 1f, 7L);
		assertMatchesReference(TINYLLAMA, DenseKvTensor.MAX_SEQ_LEN, 6f, 8L);
	}

	@Test
	@DisplayName("stays finite and exact for logits far outside the exponent's range")
	void largeLogitsAreStable() {
		Shape shape = TINYLLAMA;
		int seqLen = 300;
		Random rng = new Random(11);
		float[] q = scaled(random(rng, shape.qDim()), 400f);
		float[] k = random(rng, seqLen * shape.kvDim());
		float[] v = random(rng, seqLen * shape.kvDim());

		float[] out = attend(shape, q, k, v, seqLen);

		for (float x : out)
			assertThat(Float.isFinite(x)).as("output finite with logits in the thousands").isTrue();
		assertClose("large logits", out, reference(shape, q, k, v, seqLen), tolerance(seqLen));
	}

	@Test
	@DisplayName("one key: every head returns its KV head's value row exactly")
	void singleKeyReturnsValueRow() {
		for (Shape shape : SHAPES) {
			Random rng = new Random(shape.name().hashCode());
			float[] q = random(rng, shape.qDim());
			float[] k = random(rng, shape.kvDim());
			float[] v = random(rng, shape.kvDim());

			float[] out = attend(shape, q, k, v, 1);

			for (int h = 0; h < shape.numHeads(); h++) {
				int kBase = (h / shape.gqaRatio()) * shape.headDim();
				for (int d = 0; d < shape.headDim(); d++)
					assertThat(out[h * shape.headDim() + d]).as("%s head %d dim %d", shape.name(), h, d)
							.isEqualTo(v[kBase + d]);
			}
		}
	}

	@Test
	@DisplayName("a zero query weights every key equally: output is the mean value row")
	void zeroQueryAveragesValues() {
		Shape shape = TINYLLAMA;
		int seqLen = 777;
		Random rng = new Random(5);
		float[] k = random(rng, seqLen * shape.kvDim());
		float[] v = random(rng, seqLen * shape.kvDim());

		float[] out = attend(shape, new float[shape.qDim()], k, v, seqLen);

		float[] expected = new float[shape.qDim()];
		for (int h = 0; h < shape.numHeads(); h++) {
			int kBase = (h / shape.gqaRatio()) * shape.headDim();
			for (int d = 0; d < shape.headDim(); d++) {
				double sum = 0;
				for (int t = 0; t < seqLen; t++)
					sum += v[t * shape.kvDim() + kBase + d];
				expected[h * shape.headDim() + d] = (float) (sum / seqLen);
			}
		}
		assertClose("zero query", out, expected, tolerance(seqLen));
	}

	@Test
	@DisplayName("grouped heads: changing one KV head moves only the query heads mapped to it")
	void groupedHeadsReadOnlyTheirKvHead() {
		Shape shape = new Shape("Mistral 7B (32 heads, 8 KV, 128 wide)", 32, 8, 128);
		int seqLen = 65;
		int changedKvHead = 3;
		Random rng = new Random(17);
		float[] q = random(rng, shape.qDim());
		float[] k = random(rng, seqLen * shape.kvDim());
		float[] v = random(rng, seqLen * shape.kvDim());
		float[] before = attend(shape, q, k, v, seqLen);

		float[] k2 = k.clone();
		float[] v2 = v.clone();
		for (int t = 0; t < seqLen; t++)
			for (int d = 0; d < shape.headDim(); d++) {
				int i = t * shape.kvDim() + changedKvHead * shape.headDim() + d;
				k2[i] = -k2[i];
				v2[i] = v2[i] * 0.5f + 0.25f;
			}
		float[] after = attend(shape, q, k2, v2, seqLen);

		for (int h = 0; h < shape.numHeads(); h++) {
			float[] headBefore = head(before, h, shape.headDim());
			float[] headAfter = head(after, h, shape.headDim());
			if (h / shape.gqaRatio() == changedKvHead)
				assertThat(headAfter).as("head %d reads KV head %d", h, changedKvHead).isNotEqualTo(headBefore);
			else
				assertThat(headAfter).as("head %d does not read KV head %d", h, changedKvHead).isEqualTo(headBefore);
		}
	}

	@Test
	@DisplayName("causal: rows at or past seqLen are never read, and the outputs are overwritten")
	void rowsPastSeqLenAreIgnored() {
		Shape shape = TINYLLAMA;
		int seqLen = 100;
		int capacity = 256;
		Random rng = new Random(23);
		float[] q = random(rng, shape.qDim());
		float[] k = random(rng, seqLen * shape.kvDim());
		float[] v = random(rng, seqLen * shape.kvDim());
		float[] exact = attend(shape, q, k, v, seqLen);

		float[] kPadded = Arrays.copyOf(k, capacity * shape.kvDim());
		float[] vPadded = Arrays.copyOf(v, capacity * shape.kvDim());
		Arrays.fill(kPadded, seqLen * shape.kvDim(), kPadded.length, Float.NaN);
		Arrays.fill(vPadded, seqLen * shape.kvDim(), vPadded.length, Float.NaN);
		float[] out = new float[shape.qDim()];
		Arrays.fill(out, Float.NaN);
		float[] scores = new float[capacity];
		Arrays.fill(scores, Float.NaN);
		GqaMath.attend(q, kPadded, vPadded, seqLen, out, scores,
				shape.numHeads(), shape.headDim(), shape.gqaRatio(), shape.kvDim());

		assertThat(out).isEqualTo(exact);
	}

	// -- helpers ------------------------------------------------------------

	private static void assertMatchesReference(Shape shape, int seqLen, float qScale, long seed) {
		Random rng = new Random(seed);
		float[] q = scaled(random(rng, shape.qDim()), qScale);
		float[] k = random(rng, seqLen * shape.kvDim());
		float[] v = random(rng, seqLen * shape.kvDim());
		assertClose(shape.name() + " seqLen=" + seqLen + " qScale=" + qScale,
				attend(shape, q, k, v, seqLen), reference(shape, q, k, v, seqLen), tolerance(seqLen));
	}

	private static float[] attend(Shape shape, float[] q, float[] k, float[] v, int seqLen) {
		float[] out = new float[shape.qDim()];
		GqaMath.attend(q, k, v, seqLen, out, new float[seqLen],
				shape.numHeads(), shape.headDim(), shape.gqaRatio(), shape.kvDim());
		return out;
	}

	/**
	 * Attention written from its definition, in double, sharing no code with
	 * {@link GqaMath}: per query head, scores {@code q . k_t / sqrt(headDim)}
	 * against KV head {@code h / gqaRatio}, a max-subtracted softmax, and the
	 * score-weighted sum of value rows.
	 */
	private static float[] reference(Shape shape, float[] q, float[] k, float[] v, int seqLen) {
		int hd = shape.headDim();
		int kvDim = shape.kvDim();
		double scale = 1.0 / Math.sqrt(hd);
		float[] out = new float[shape.qDim()];
		double[] logits = new double[seqLen];
		for (int h = 0; h < shape.numHeads(); h++) {
			int kvCol = (h / shape.gqaRatio()) * hd;
			double max = Double.NEGATIVE_INFINITY;
			for (int t = 0; t < seqLen; t++) {
				double dot = 0;
				for (int d = 0; d < hd; d++)
					dot += (double) q[h * hd + d] * k[t * kvDim + kvCol + d];
				logits[t] = dot * scale;
				max = Math.max(max, logits[t]);
			}
			double denom = 0;
			for (int t = 0; t < seqLen; t++) {
				logits[t] = Math.exp(logits[t] - max);
				denom += logits[t];
			}
			for (int d = 0; d < hd; d++) {
				double acc = 0;
				for (int t = 0; t < seqLen; t++)
					acc += logits[t] * v[t * kvDim + kvCol + d];
				out[h * hd + d] = (float) (acc / denom);
			}
		}
		return out;
	}

	private static void assertClose(String label, float[] actual, float[] expected, double tol) {
		assertThat(actual).as(label).hasSameSizeAs(expected);
		double worst = 0;
		int worstAt = -1;
		for (int i = 0; i < expected.length; i++) {
			double err = Math.abs((double) actual[i] - expected[i]);
			if (!(err <= worst)) {
				worst = err;
				worstAt = i;
			}
		}
		assertThat(worst).as("%s: largest error %.3e at [%d], tolerance %.3e", label, worst, worstAt, tol)
				.isLessThanOrEqualTo(tol);
	}

	private static float[] head(float[] row, int h, int headDim) {
		return Arrays.copyOfRange(row, h * headDim, (h + 1) * headDim);
	}

	private static float[] random(Random rng, int n) {
		float[] a = new float[n];
		for (int i = 0; i < n; i++)
			a[i] = rng.nextFloat() * 2f - 1f;
		return a;
	}

	private static float[] scaled(float[] a, float s) {
		for (int i = 0; i < a.length; i++)
			a[i] *= s;
		return a;
	}
}
