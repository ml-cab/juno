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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.Random;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * Correctness of the cache-blocked integer matmul ({@link CpuIntegerGemm}).
 *
 * <p>Each activation row is quantized exactly as the decode kernel quantizes it,
 * so for every row the matmul must agree with the exact reference (dequantized
 * weights against the quantized activation, in double) to float rounding, and
 * with {@link CpuIntegerKernels} row by row to float rounding.
 */
class CpuIntegerGemmTest {

	@ParameterizedTest(name = "type {0}, {1} rows x {2} cols, B={3}")
	@CsvSource({ "12, 37, 256, 1", "12, 40, 1024, 8", "12, 33, 512, 128", "12, 20, 2048, 741",
			"13, 37, 256, 1", "13, 40, 1024, 8", "13, 33, 512, 128", "13, 20, 2048, 741",
			"14, 37, 256, 1", "14, 40, 1024, 8", "14, 33, 512, 128", "14, 20, 2048, 741",
			"8, 37, 256, 1", "8, 40, 1024, 8", "8, 33, 512, 128", "8, 20, 2048, 741" })
	void matchesExactReferenceAndDecodeKernel(int type, int rows, int cols, int batch) {
		Random rnd = new Random(97L * type + batch);
		GgufReader.QuantizedTensor w = CpuIntegerKernelsTest.synthetic(type, rows, cols, rnd);
		float[][] X = activations(batch, cols, rnd);
		float[][] Y = new float[batch][rows];
		CpuIntegerGemm.sgemmInto(w, X, Y, 0, rows, cols);

		float[] dq = LlamaTransformerHandler.dequantize(w, rows, cols);
		double worstExact = 0;
		double worstDecode = 0;
		for (int b = 0; b < batch; b++) {
			float[] xq = CpuIntegerKernels.Q8Activation.quantize(X[b], cols).dequantized();
			double[] exact = new double[rows];
			double scale = 0;
			for (int r = 0; r < rows; r++) {
				double s = 0;
				for (int c = 0; c < cols; c++)
					s += (double) dq[r * cols + c] * xq[c];
				exact[r] = s;
				scale = Math.max(scale, Math.abs(s));
			}
			float[] decode = new float[rows];
			CpuIntegerKernels.matVecInto(w, X[b], 0, rows, decode, cols);
			for (int r = 0; r < rows; r++) {
				worstExact = Math.max(worstExact, Math.abs(Y[b][r] - exact[r]) / scale);
				worstDecode = Math.max(worstDecode, Math.abs(Y[b][r] - decode[r]) / scale);
			}
		}
		assertThat(worstExact).as("against the exact reference").isLessThan(1e-5);
		assertThat(worstDecode).as("against the decode kernel").isLessThan(1e-5);
	}

	@Test
	void rowRangeWritesFromIndexZeroAndLeavesTheRestAlone() {
		for (int type : new int[] { 8, 12, 13, 14 }) {
			Random rnd = new Random(3 + type);
			int rows = 50, cols = 512, batch = 5;
			GgufReader.QuantizedTensor w = CpuIntegerKernelsTest.synthetic(type, rows, cols, rnd);
			float[][] X = activations(batch, cols, rnd);
			float[][] all = new float[batch][rows];
			CpuIntegerGemm.sgemmInto(w, X, all, 0, rows, cols);
			float[][] part = new float[batch][rows];
			for (float[] p : part)
				java.util.Arrays.fill(p, Float.NaN);
			CpuIntegerGemm.sgemmInto(w, X, part, 13, 44, cols);
			for (int b = 0; b < batch; b++) {
				for (int r = 0; r < 31; r++)
					assertThat(part[b][r]).as("type %d b %d row %d", type, b, r + 13).isEqualTo(all[b][r + 13]);
				for (int r = 31; r < rows; r++)
					assertThat(part[b][r]).isNaN();
			}
		}
	}

	@Test
	void reusedScratchGivesTheSameResultForASmallerCall() {
		Random rnd = new Random(41);
		GgufReader.QuantizedTensor w = CpuIntegerKernelsTest.synthetic(12, 24, 1024, rnd);
		float[][] big = activations(64, 1024, rnd);
		CpuIntegerGemm.sgemmInto(w, big, new float[64][24], 0, 24, 1024);
		float[][] small = activations(3, 1024, rnd);
		float[][] y1 = new float[3][24];
		CpuIntegerGemm.sgemmInto(w, small, y1, 0, 24, 1024);
		for (int b = 0; b < 3; b++) {
			float[] decode = new float[24];
			CpuIntegerKernels.matVecInto(w, small[b], 0, 24, decode, 1024);
			for (int r = 0; r < 24; r++)
				assertThat(y1[b][r]).isCloseTo(decode[r], org.assertj.core.data.Offset.offset(1e-4f * Math.max(1f, Math.abs(decode[r]))));
		}
	}

	@ParameterizedTest(name = "type {0}, {1} rows x {2} cols")
	@CsvSource({ "12, 2048, 11008", "14, 2048, 11008", "12, 11008, 2048", "14, 256, 2048" })
	void bitwiseDeterministicAcrossCallsAndFromInsideForkJoinTasks(int type, int rows, int cols) {
		Random rnd = new Random(type + rows);
		GgufReader.QuantizedTensor w = CpuIntegerKernelsTest.synthetic(type, rows, cols, rnd);
		float[][] X = activations(24, cols, rnd);
		float[][] ref = new float[24][rows];
		CpuIntegerGemm.sgemmInto(w, X, ref, 0, rows, cols);
		for (int rep = 0; rep < 3; rep++) {
			float[][] y = new float[24][rows];
			CpuIntegerGemm.sgemmInto(w, X, y, 0, rows, cols);
			for (int b = 0; b < 24; b++)
				assertThat(y[b]).as("repeat %d, row %d", rep, b).isEqualTo(ref[b]);
		}
		// Several matmuls issued from common-pool tasks at once: a worker waiting on its
		// own matmul may run another task's matmul; neither may disturb the other.
		float[][][] ys = new float[6][24][rows];
		java.util.stream.IntStream.range(0, 6).parallel()
				.forEach(k -> CpuIntegerGemm.sgemmInto(w, X, ys[k], 0, rows, cols));
		for (int k = 0; k < 6; k++)
			for (int b = 0; b < 24; b++)
				assertThat(ys[k][b]).as("concurrent call %d, row %d", k, b).isEqualTo(ref[b]);
		float[] dref = new float[rows];
		CpuIntegerKernels.matVecInto(w, X[0], 0, rows, dref, cols);
		float[][] ds = new float[6][rows];
		java.util.stream.IntStream.range(0, 6).parallel()
				.forEach(k -> CpuIntegerKernels.matVecInto(w, X[0], 0, rows, ds[k], cols));
		for (int k = 0; k < 6; k++)
			assertThat(ds[k]).as("concurrent decode call %d", k).isEqualTo(dref);
	}

	@Test
	void aBatchNotAMultipleOfFourIsPaddedWithoutChangingItsRows() {
		// The batch is padded with zero rows to a multiple of four; a real row's arithmetic does not depend on its
		// position or on the padding, and the reused buffers must not carry an earlier call's rows into the padding.
		for (int type : new int[] { 8, 12, 13, 14 }) {
			Random rnd = new Random(19 + type);
			int rows = 40, cols = 512;
			GgufReader.QuantizedTensor w = CpuIntegerKernelsTest.synthetic(type, rows, cols, rnd);
			float[][] eight = activations(8, cols, rnd);
			float[][] five = java.util.Arrays.copyOf(eight, 5);
			float[][] y5 = new float[5][rows];
			CpuIntegerGemm.sgemmInto(w, five, y5, 0, rows, cols);
			float[][] y8 = new float[8][rows];
			CpuIntegerGemm.sgemmInto(w, eight, y8, 0, rows, cols);
			float[][] y5Again = new float[5][rows];
			CpuIntegerGemm.sgemmInto(w, five, y5Again, 0, rows, cols);
			for (int b = 0; b < 5; b++) {
				assertThat(y5[b]).as("type %d row %d alone against inside a batch of 8", type, b).isEqualTo(y8[b]);
				assertThat(y5Again[b]).as("type %d row %d after a larger call", type, b).isEqualTo(y5[b]);
			}
		}
	}

	@Test
	void rejectsWidthsThatAreNotWholeSuperBlocks() {
		assertThat(CpuIntegerGemm.supports(8, 96)).isFalse();
		assertThat(CpuIntegerGemm.supports(12, 4096)).isTrue();
		assertThat(CpuIntegerGemm.supports(1, 4096)).isFalse();
		GgufReader.QuantizedTensor w = CpuIntegerKernelsTest.synthetic(8, 4, 96, new Random(1));
		assertThatThrownBy(() -> CpuIntegerGemm.sgemmInto(w, new float[2][96], new float[2][4], 0, 4, 96))
				.isInstanceOf(IllegalArgumentException.class);
	}

	private static float[][] activations(int batch, int cols, Random rnd) {
		float[][] X = new float[batch][cols];
		for (float[] x : X) {
			for (int i = 0; i < cols; i++)
				x[i] = (float) rnd.nextGaussian();
			for (int i = 0; i < cols / 128; i++)
				x[rnd.nextInt(cols)] *= 20f;
		}
		return X;
	}
}
