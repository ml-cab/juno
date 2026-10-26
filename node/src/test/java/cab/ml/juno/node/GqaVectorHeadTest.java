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

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * The vector attention head ({@link GqaVectorHead}) behind {@link GqaMath#attendPooled}. Its dot products sum in
 * four float lanes and then across them in a fixed order, so its output differs from the serial oracle
 * ({@link GqaMath#attend}) by float rounding only. It is held to {@code GqaMathOracleTest}'s bound against an
 * independently written double-precision reference, at the same head shapes and lengths, and must return the same
 * bits at every thread count and on every call (no dependence on which compilation tier ran it). The weighted value
 * sum keeps the serial loop's order per element.
 */
class GqaVectorHeadTest {

	private final int threadsBefore = CpuThreads.threads();

	@AfterEach
	void restoreThreads() {
		CpuThreads.setThreads(threadsBefore);
	}

	/** {@code GqaMathOracleTest}'s tolerance: absolute, on outputs whose value rows lie in [-1, 1]. */
	private static double tolerance(int keys) {
		return 2e-6 + 2e-6 * Math.sqrt(keys);
	}

	@ParameterizedTest(name = "heads {0}/{1}, dim {2}, seqLen {3}, window {4}, q scale {5}")
	@CsvSource({
			// the sweep models' and the other handlers' head shapes
			"32, 4, 64, 1, 0, 1", "32, 4, 64, 517, 0, 6", "32, 4, 64, 2048, 0, 1", "32, 8, 128, 2016, 300, 1",
			"16, 2, 128, 500, 0, 6", "32, 32, 96, 600, 7, 1", "16, 8, 128, 33, 0, 1", "32, 32, 80, 129, 0, 6",
			// widths with a remainder after the 16-wide dot blocks and the 32-wide value blocks
			"8, 2, 48, 77, 0, 1", "8, 2, 72, 65, 0, 6", "8, 4, 40, 31, 0, 1", "4, 2, 8, 19, 0, 1", "4, 1, 256, 300, 0, 1",
			"6, 3, 20, 9, 5, 1", "4, 2, 6, 40, 0, 1" })
	void withinTheOracleBoundOfADoubleReference(int numHeads, int numKvHeads, int headDim, int seqLen, int window,
			float qScale) {
		CpuThreads.setThreads(12);
		int gqaRatio = numHeads / numKvHeads;
		int kvDim = numKvHeads * headDim;
		Random rnd = new Random(17L * seqLen + 31L * headDim + window);
		float[] q = gaussian(numHeads * headDim, rnd, qScale);
		float[] k = gaussian(seqLen * kvDim, rnd, 1f);
		float[] v = uniform(seqLen * kvDim, rnd);

		float[] got = new float[numHeads * headDim];
		Arrays.fill(got, Float.NaN); // every element must be written
		GqaMath.attendPooled(q, k, v, seqLen, got, numHeads, headDim, gqaRatio, kvDim, window, true);
		double[] ref = reference(q, k, v, seqLen, numHeads, headDim, gqaRatio, kvDim, window);

		int keys = seqLen - GqaMath.firstKey(seqLen, window);
		double worst = 0;
		for (int i = 0; i < got.length; i++) {
			double err = Math.abs(got[i] - ref[i]);
			worst = Math.max(worst, err);
			assertThat(err).as("element %d (head %d, dim %d)", i, i / headDim, i % headDim)
					.isLessThanOrEqualTo(tolerance(keys));
		}
		System.out.printf("dim %d seqLen %d window %d: worst %.1e against bound %.1e%n", headDim, seqLen, window, worst,
				tolerance(keys));
	}

	@Test
	void sameBitsAtEveryThreadCountAndOnEveryCall() {
		int numHeads = 32, numKvHeads = 8, headDim = 128, seqLen = 333;
		int kvDim = numKvHeads * headDim;
		Random rnd = new Random(41);
		float[] q = gaussian(numHeads * headDim, rnd, 1f);
		float[] k = gaussian(seqLen * kvDim, rnd, 1f);
		float[] v = uniform(seqLen * kvDim, rnd);
		CpuThreads.setThreads(1);
		float[] first = new float[numHeads * headDim];
		GqaMath.attendPooled(q, k, v, seqLen, first, numHeads, headDim, 4, kvDim, 0, true);
		for (int threads : new int[] { 2, 12, 5 }) {
			CpuThreads.setThreads(threads);
			for (int call = 0; call < 400; call++) {
				float[] again = new float[numHeads * headDim];
				GqaMath.attendPooled(q, k, v, seqLen, again, numHeads, headDim, 4, kvDim, 0, true);
				for (int i = 0; i < first.length; i++)
					assertThat(Float.floatToRawIntBits(again[i])).as("threads %d call %d element %d", threads, call, i)
							.isEqualTo(Float.floatToRawIntBits(first[i]));
			}
		}
	}

	@Test
	void largeLogitsStayFiniteAndWithinBound() {
		CpuThreads.setThreads(4);
		int numHeads = 8, numKvHeads = 2, headDim = 64, seqLen = 300;
		int kvDim = numKvHeads * headDim;
		Random rnd = new Random(5);
		float[] q = gaussian(numHeads * headDim, rnd, 400f); // logits in the thousands
		float[] k = gaussian(seqLen * kvDim, rnd, 1f);
		float[] v = uniform(seqLen * kvDim, rnd);
		float[] got = new float[numHeads * headDim];
		GqaMath.attendPooled(q, k, v, seqLen, got, numHeads, headDim, 4, kvDim, 0, true);
		double[] ref = reference(q, k, v, seqLen, numHeads, headDim, 4, kvDim, 0);
		for (int i = 0; i < got.length; i++) {
			assertThat(Float.isFinite(got[i])).as("element %d finite", i).isTrue();
			assertThat(Math.abs(got[i] - ref[i])).as("element %d", i).isLessThanOrEqualTo(tolerance(seqLen));
		}
	}

	@Test
	void theVectorHeadRunsWhenTheVectorApiIsPresent() {
		// the node module's tests run with jdk.incubator.vector, so the default entry point takes the vector head
		assertThat(VectorQuantKernels.AVAILABLE).isTrue();
		assertThat(GqaMath.VECTOR_HEAD).isTrue();
	}

	/** Independent double-precision attention over keys {@code [firstKey(seqLen, window), seqLen)}. */
	private static double[] reference(float[] q, float[] k, float[] v, int seqLen, int numHeads, int headDim,
			int gqaRatio, int kvDim, int window) {
		int lo = GqaMath.firstKey(seqLen, window);
		double scale = 1.0 / Math.sqrt(headDim);
		double[] out = new double[numHeads * headDim];
		double[] s = new double[seqLen];
		for (int h = 0; h < numHeads; h++) {
			int kBase = (h / gqaRatio) * headDim;
			double max = Double.NEGATIVE_INFINITY;
			for (int t = lo; t < seqLen; t++) {
				double dot = 0;
				for (int d = 0; d < headDim; d++)
					dot += (double) q[h * headDim + d] * k[t * kvDim + kBase + d];
				s[t] = dot * scale;
				max = Math.max(max, s[t]);
			}
			double sum = 0;
			for (int t = lo; t < seqLen; t++) {
				s[t] = Math.exp(s[t] - max);
				sum += s[t];
			}
			for (int t = lo; t < seqLen; t++)
				for (int d = 0; d < headDim; d++)
					out[h * headDim + d] += s[t] / sum * v[t * kvDim + kBase + d];
		}
		return out;
	}

	private static float[] gaussian(int n, Random rnd, float scale) {
		float[] a = new float[n];
		for (int i = 0; i < n; i++)
			a[i] = (float) rnd.nextGaussian() * scale;
		return a;
	}

	private static float[] uniform(int n, Random rnd) {
		float[] a = new float[n];
		for (int i = 0; i < n; i++)
			a[i] = (float) (rnd.nextDouble() * 2 - 1);
		return a;
	}
}
