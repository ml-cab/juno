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

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * {@link GqaMath#attendPooled} spreads one row's query heads over the CPU
 * worker pool. With the serial head (the form taken without the Vector API
 * module) each head is computed exactly as {@link GqaMath#attend} computes it,
 * with its own score scratch, so the output must be bit-identical at every
 * thread count, head shape, context length and window. The vector head is held
 * to the oracle bound in {@link GqaVectorHeadTest}.
 */
class GqaMathPooledTest {

	private final int threadsBefore = CpuThreads.threads();

	@AfterEach
	void restoreThreads() {
		CpuThreads.setThreads(threadsBefore);
	}

	@ParameterizedTest(name = "heads {0}/{1}, dim {2}, seqLen {3}, window {4}, threads {5}")
	@CsvSource({
			// TinyLlama 1.1B
			"32, 4, 64, 1, 0, 12", "32, 4, 64, 7, 0, 2", "32, 4, 64, 517, 0, 12", "32, 4, 64, 2048, 0, 12",
			// Mistral 7B
			"32, 8, 128, 33, 0, 1", "32, 8, 128, 2016, 0, 12", "32, 8, 128, 2048, 300, 12",
			// Qwen2.5-3B
			"16, 2, 128, 500, 0, 12", "16, 2, 128, 128, 8, 2",
			// Phi-3.5-mini
			"32, 32, 96, 1, 0, 12", "32, 32, 96, 2048, 0, 12", "32, 32, 96, 600, 7, 12" })
	void bitIdenticalToTheSerialLoop(int numHeads, int numKvHeads, int headDim, int seqLen, int window, int threads) {
		CpuThreads.setThreads(threads);
		int gqaRatio = numHeads / numKvHeads;
		int kvDim = numKvHeads * headDim;
		Random rnd = new Random(31L * seqLen + numHeads + window);
		float[] q = gaussian(numHeads * headDim, rnd);
		float[] k = gaussian(seqLen * kvDim, rnd);
		float[] v = gaussian(seqLen * kvDim, rnd);

		float[] want = new float[numHeads * headDim];
		GqaMath.attend(q, k, v, seqLen, want, new float[seqLen], numHeads, headDim, gqaRatio, kvDim, window);
		float[] got = new float[numHeads * headDim];
		java.util.Arrays.fill(got, Float.NaN); // the pooled form must overwrite every element
		GqaMath.attendPooled(q, k, v, seqLen, got, numHeads, headDim, gqaRatio, kvDim, window, false);

		for (int i = 0; i < want.length; i++)
			assertThat(Float.floatToRawIntBits(got[i])).as("element %d", i).isEqualTo(Float.floatToRawIntBits(want[i]));
	}

	@Test
	void sameBitsWhenCalledFromInsideAPoolTask() {
		CpuThreads.setThreads(4);
		int numHeads = 32, headDim = 64, kvDim = 4 * 64, seqLen = 300;
		Random rnd = new Random(9);
		float[][] q = new float[4][];
		for (int i = 0; i < 4; i++)
			q[i] = gaussian(numHeads * headDim, rnd);
		float[] k = gaussian(seqLen * kvDim, rnd);
		float[] v = gaussian(seqLen * kvDim, rnd);
		float[][] want = new float[4][numHeads * headDim];
		for (int i = 0; i < 4; i++)
			GqaMath.attend(q[i], k, v, seqLen, want[i], new float[seqLen], numHeads, headDim, 8, kvDim, 0);
		float[][] got = new float[4][numHeads * headDim];
		CpuThreads.forEach(4, i -> GqaMath.attendPooled(q[i], k, v, seqLen, got[i], numHeads, headDim, 8, kvDim, 0, false));
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < want[i].length; j++)
				assertThat(Float.floatToRawIntBits(got[i][j])).as("row %d element %d", i, j)
						.isEqualTo(Float.floatToRawIntBits(want[i][j]));
	}

	private static float[] gaussian(int n, Random rnd) {
		float[] a = new float[n];
		for (int i = 0; i < n; i++)
			a[i] = (float) rnd.nextGaussian();
		return a;
	}
}
