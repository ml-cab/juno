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
import java.util.stream.IntStream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * The decode-batch matmul gives every activation row, bit for bit, what the single-row kernel gives that row alone:
 * the property that keeps a request's greedy output independent of what else is in its decode batch.
 */
class CpuIntegerDecodeBatchTest {

	@ParameterizedTest(name = "type {0}, {1} rows x {2} cols, B={3}, wide={4}")
	@CsvSource({ "12, 37, 256, 1, false", "12, 37, 1024, 2, false", "12, 40, 512, 3, false", "12, 37, 1024, 4, false",
			"12, 33, 512, 5, false", "12, 20, 2048, 8, false", "12, 37, 1024, 9, false",
			"12, 37, 1024, 2, true", "12, 33, 512, 5, true", "12, 20, 2048, 8, true",
			"13, 37, 256, 1, false", "13, 37, 1024, 2, false", "13, 33, 512, 5, false", "13, 20, 2048, 8, false",
			"13, 37, 1024, 2, true", "13, 33, 512, 7, true",
			"14, 37, 256, 1, false", "14, 37, 1024, 2, false", "14, 33, 512, 5, false", "14, 20, 2048, 8, false",
			"14, 37, 1024, 2, true", "14, 33, 512, 7, true",
			"8, 37, 256, 1, false", "8, 37, 1024, 2, false", "8, 33, 96, 5, false", "8, 20, 2048, 8, false",
			"8, 33, 96, 3, true" })
	void everyRowIsBitIdenticalToTheSingleRowKernel(int type, int rows, int cols, int batch, boolean wide) {
		Random rnd = new Random(31L * type + 7L * batch + cols + (wide ? 1 : 0));
		GgufReader.QuantizedTensor w = CpuIntegerKernelsTest.synthetic(type, rows, cols, rnd);
		float[][] X = activations(batch, cols, rnd);
		float[][] Y = new float[batch][rows];
		CpuIntegerDecodeBatch.matmulInto(w, X, Y, 0, rows, cols, wide);
		for (int b = 0; b < batch; b++) {
			float[] single = new float[rows];
			CpuIntegerKernels.matVecInto(w, X[b], 0, rows, single, cols, wide);
			for (int r = 0; r < rows; r++)
				assertThat(Float.floatToRawIntBits(Y[b][r])).as("row %d of activation %d", r, b)
						.isEqualTo(Float.floatToRawIntBits(single[r]));
		}
	}

	@Test
	void aRowRangeWritesFromIndexZeroAndMatchesTheSingleRowKernelOverTheSameRange() {
		for (int type : new int[] { 12, 13, 14, 8 }) {
			Random rnd = new Random(type);
			GgufReader.QuantizedTensor w = CpuIntegerKernelsTest.synthetic(type, 40, 512, rnd);
			float[][] X = activations(6, 512, rnd);
			float[][] Y = new float[6][40];
			java.util.Arrays.stream(Y).forEach(y -> java.util.Arrays.fill(y, Float.NaN));
			CpuIntegerDecodeBatch.matmulInto(w, X, Y, 13, 29, 512, CpuIntegerWidth.WIDE);
			for (int b = 0; b < 6; b++) {
				float[] single = new float[16];
				CpuIntegerKernels.matVecInto(w, X[b], 13, 29, single, 512);
				for (int r = 0; r < 16; r++)
					assertThat(Float.floatToRawIntBits(Y[b][r])).as("type %d b %d row %d", type, b, r + 13)
							.isEqualTo(Float.floatToRawIntBits(single[r]));
				for (int r = 16; r < 40; r++)
					assertThat(Y[b][r]).as("type %d b %d untouched index %d", type, b, r).isNaN();
			}
		}
	}

	@Test
	void aSmallerCallAfterALargerOneReusesScratchWithoutChangingItsRows() {
		Random rnd = new Random(5);
		GgufReader.QuantizedTensor wide = CpuIntegerKernelsTest.synthetic(12, 24, 2048, rnd);
		GgufReader.QuantizedTensor narrow = CpuIntegerKernelsTest.synthetic(12, 24, 512, rnd);
		CpuIntegerDecodeBatch.matmulInto(wide, activations(9, 2048, rnd), new float[9][24], 0, 24, 2048,
				CpuIntegerWidth.WIDE);
		float[][] X = activations(3, 512, rnd);
		float[][] Y = new float[3][24];
		CpuIntegerDecodeBatch.matmulInto(narrow, X, Y, 0, 24, 512, CpuIntegerWidth.WIDE);
		for (int b = 0; b < 3; b++) {
			float[] single = new float[24];
			CpuIntegerKernels.matVecInto(narrow, X[b], 0, 24, single, 512);
			for (int r = 0; r < 24; r++)
				assertThat(Float.floatToRawIntBits(Y[b][r])).isEqualTo(Float.floatToRawIntBits(single[r]));
		}
	}

	@Test
	void concurrentCallersEachGetTheirOwnRowsBitForBit() {
		Random rnd = new Random(11);
		GgufReader.QuantizedTensor w = CpuIntegerKernelsTest.synthetic(14, 64, 1024, rnd);
		float[][][] Xs = new float[8][][];
		float[][][] want = new float[8][][];
		for (int k = 0; k < 8; k++) {
			Xs[k] = activations(1 + k % 5, 1024, rnd);
			want[k] = new float[Xs[k].length][64];
			for (int b = 0; b < Xs[k].length; b++)
				CpuIntegerKernels.matVecInto(w, Xs[k][b], 0, 64, want[k][b], 1024);
		}
		float[][][] got = new float[8][][];
		IntStream.range(0, 8).parallel().forEach(k -> {
			got[k] = new float[Xs[k].length][64];
			CpuIntegerDecodeBatch.matmulInto(w, Xs[k], got[k], 0, 64, 1024, CpuIntegerWidth.WIDE);
		});
		for (int k = 0; k < 8; k++)
			for (int b = 0; b < Xs[k].length; b++)
				for (int r = 0; r < 64; r++)
					assertThat(Float.floatToRawIntBits(got[k][b][r])).as("call %d, activation %d, row %d", k, b, r)
							.isEqualTo(Float.floatToRawIntBits(want[k][b][r]));
	}

	@Test
	void rejectsATypeTheSingleRowKernelDoesNotImplement() {
		GgufReader.QuantizedTensor f32 = new GgufReader.QuantizedTensor("w", 0, 256L * 4, new byte[256 * 4 * 4]);
		assertThatThrownBy(() -> CpuIntegerDecodeBatch.matmulInto(f32, new float[2][256], new float[2][4], 0, 4, 256,
				false)).isInstanceOf(IllegalArgumentException.class);
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
