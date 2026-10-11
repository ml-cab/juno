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

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.api.Test;

/**
 * Correctness of the integer-dot CPU decode kernels ({@link CpuIntegerKernels}).
 *
 * <p>Two references per format. The <em>exact</em> one dots the dequantized
 * weights with the activation as the kernel quantized it, in double: the
 * integer arithmetic is exact, so the kernel must agree to float rounding,
 * which catches any lane-mapping or bit-extraction error. The <em>float</em>
 * one dots the dequantized weights with the original activation: the
 * difference is the activation quantization error, bounded loosely here and
 * scored on real models by the model-quality test.
 */
class CpuIntegerKernelsTest {

	private static final int[] FORMATS = { 8, 12, 13, 14 };

	@ParameterizedTest(name = "type {0}, {1} x {2}")
	@CsvSource({ "12, 64, 256", "12, 96, 4096", "13, 64, 256", "13, 96, 4096", "14, 64, 256", "14, 96, 4096",
			"8, 64, 256", "8, 96, 4096", "12, 33, 14336", "14, 40, 14336" })
	void matchesExactReferenceAndBoundsQuantizationError(int type, int rows, int cols) {
		Random rnd = new Random(31L * type + cols);
		GgufReader.QuantizedTensor w = synthetic(type, rows, cols, rnd);
		float[] x = activation(cols, rnd);

		float[] y = new float[rows];
		CpuIntegerKernels.matVecInto(w, x, 0, rows, y, cols);

		float[] dq = LlamaTransformerHandler.dequantize(w, rows, cols);
		float[] xq = CpuIntegerKernels.Q8Activation.quantize(x, cols).dequantized();
		double[] exact = dot(dq, xq, rows, cols);
		double[] ref = dot(dq, x, rows, cols);

		double exactErr = maxRelative(exact, y);
		double quantErr = maxRelative(ref, y);
		System.out.printf("type %d %dx%d: vs exact %.2e, activation quantization %.2e%n", type, rows, cols, exactErr,
				quantErr);
		assertThat(exactErr).as("integer arithmetic against the exact reference").isLessThan(1e-5);
		assertThat(quantErr).as("activation quantization error").isLessThan(2e-2);
	}

	@Test
	void rowRangeWritesFromIndexZero() {
		for (int type : FORMATS) {
			Random rnd = new Random(7 + type);
			int rows = 48, cols = 512;
			GgufReader.QuantizedTensor w = synthetic(type, rows, cols, rnd);
			float[] x = activation(cols, rnd);
			float[] all = new float[rows];
			CpuIntegerKernels.matVecInto(w, x, 0, rows, all, cols);
			float[] part = new float[rows];
			CpuIntegerKernels.matVecInto(w, x, 17, 41, part, cols);
			for (int r = 0; r < 24; r++)
				assertThat(part[r]).as("type %d row %d", type, r + 17).isEqualTo(all[r + 17]);
		}
	}

	@Test
	void zeroActivationBlockQuantizesToZeroWithoutNaN() {
		float[] x = new float[256];
		for (int i = 64; i < 96; i++)
			x[i] = 0.5f * (i - 80);
		CpuIntegerKernels.Q8Activation a = CpuIntegerKernels.Q8Activation.quantize(x, 256);
		float[] back = a.dequantized();
		for (int i = 0; i < 64; i++)
			assertThat(back[i]).isZero();
		// Block maximum 8, so the step is 8 / 127 and rounding is off by at most half of it.
		for (int i = 64; i < 96; i++)
			assertThat(back[i]).isCloseTo(x[i], org.assertj.core.data.Offset.offset(8f / 127f / 2f + 1e-6f));

		Random rnd = new Random(5);
		for (int type : FORMATS) {
			GgufReader.QuantizedTensor w = synthetic(type, 8, 256, rnd);
			float[] y = new float[8];
			CpuIntegerKernels.matVecInto(w, new float[256], 0, 8, y, 256);
			for (float v : y)
				assertThat(v).as("type %d", type).isZero();
		}
	}

	@Test
	void supportsExactlyTheIntegerFormats() {
		for (int type : FORMATS)
			assertThat(CpuIntegerKernels.supports(type)).isTrue();
		for (int type : new int[] { 0, 1, 10, 11 })
			assertThat(CpuIntegerKernels.supports(type)).isFalse();
	}

	// ── helpers ─────────────────────────────────────────────────────────────

	private static double[] dot(float[] w, float[] x, int rows, int cols) {
		double[] y = new double[rows];
		for (int r = 0; r < rows; r++) {
			double s = 0;
			for (int c = 0; c < cols; c++)
				s += (double) w[r * cols + c] * x[c];
			y[r] = s;
		}
		return y;
	}

	private static double maxRelative(double[] ref, float[] got) {
		double scale = 0;
		for (double v : ref)
			scale = Math.max(scale, Math.abs(v));
		double max = 0;
		for (int i = 0; i < ref.length; i++)
			max = Math.max(max, Math.abs(got[i] - ref[i]));
		return max / scale;
	}

	/** Gaussian activation with a few large outliers, as transformer activations have. */
	private static float[] activation(int cols, Random rnd) {
		float[] x = new float[cols];
		for (int i = 0; i < cols; i++)
			x[i] = (float) rnd.nextGaussian();
		for (int i = 0; i < cols / 128; i++)
			x[rnd.nextInt(cols)] *= 20f;
		return x;
	}

	/** Random packed blocks with finite FP16 scales. */
	static GgufReader.QuantizedTensor synthetic(int type, int rows, int cols, Random rnd) {
		return CpuIntegerKernelsTestSupport.synthetic(type, rows, cols, rnd);
	}
}
