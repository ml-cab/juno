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

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * Parity of the 256-bit CPU integer kernels with the 128-bit kernels, which are
 * the oracle, forced to each width in the same JVM whatever the host prefers.
 *
 * <p>Integer sums are exact at either width. Where the float expression is the
 * same, the output is the same bits: the weight unpack (integers and their
 * scale factors), and the Q5_K, Q6_K and Q8_0 decode kernels. The Q4_K decode
 * kernel and the prefill matmul scale their sub-block sums in float lanes, eight
 * at 256 bits against four at 128, so they differ by float rounding only: Q4_K
 * within 1e-6 of the row's term magnitude ({@link CpuQ4kKernelParityTest}'s
 * bound), the matmul within 1e-6 of the output's term magnitude: the sum over its
 * sub-blocks of the absolute dot term and offset term the tile scales, from the
 * unpacked weights, in double. A largest-output bound would tighten as columns
 * grow, since the offset terms of Q6_K's unsigned weights are large.
 *
 * <p>On a host without 256-bit integer instructions the 256-bit kernels run in the
 * Vector API's portable form: correct, and slow, so the shapes here are small.
 */
class CpuIntegerWidthParityTest {

	private static final double BOUND = 1e-6;

	@ParameterizedTest(name = "type {0}, {1} rows x {2} cols")
	@CsvSource({ "12, 37, 256", "12, 9, 4096", "13, 37, 256", "13, 9, 4096", "14, 37, 256", "14, 9, 4096",
			"8, 37, 256", "8, 9, 4096" })
	void decodeKernelsAgreeAcrossWidths(int type, int rows, int cols) {
		Random rnd = new Random(31L * type + cols);
		GgufReader.QuantizedTensor w = CpuIntegerKernelsTest.synthetic(type, rows, cols, rnd);
		assertDecodeParity(type, w.data(), rows, cols, activation(cols, rnd));
	}

	@Test
	void decodeKernelsAgreeOnExtremeBlocks() {
		int cols = 512;
		float[] plus = new float[cols];
		Arrays.fill(plus, 3.5f);
		float[] minus = new float[cols];
		Arrays.fill(minus, -3.5f);
		float[] alternating = new float[cols];
		for (int i = 0; i < cols; i++)
			alternating[i] = i % 2 == 0 ? 3.5f : -3.5f;
		for (int type : new int[] { 12, 13, 14, 8 })
			for (byte fill : new byte[] { (byte) 0xFF, 0x00, (byte) 0x80, 0x7F, 0x55, (byte) 0xAA }) {
				byte[] raw = extreme(type, 3, cols, fill);
				for (float[] x : new float[][] { plus, minus, alternating })
					assertDecodeParity(type, raw, 3, cols, x);
			}
	}

	@Test
	void weightUnpackIsIdenticalAcrossWidths() {
		for (int type : new int[] { 12, 13, 14, 8 }) {
			Random rnd = new Random(5 + type);
			int rows = 3, cols = 1024;
			byte[][] raws = { CpuIntegerKernelsTest.synthetic(type, rows, cols, rnd).data(), extreme(type, rows, cols, (byte) 0xFF),
					extreme(type, rows, cols, (byte) 0xAA) };
			for (byte[] raw : raws)
				for (int r = 0; r < rows; r++)
					for (int kb = 0; kb < cols / 256; kb++) {
						short[] w128 = new short[3 * 256];
						short[] w256 = new short[3 * 256];
						Arrays.fill(w128, (short) 0x5A5A);
						Arrays.fill(w256, (short) 0x5A5A);
						float[] sc128 = new float[48], sc256 = new float[48], off128 = new float[48], off256 = new float[48];
						CpuIntegerGemm.unpack(type, raw, r, kb, cols, w128, 256, sc128, off128, 16);
						CpuIntegerGemm256.unpack(type, raw, r, kb, cols, w256, 256, sc256, off256, 16);
						assertThat(w256).as("type %d row %d block %d weights", type, r, kb).isEqualTo(w128);
						for (int i = 0; i < 48; i++) {
							assertThat(Float.floatToRawIntBits(sc256[i])).as("type %d scale %d", type, i)
									.isEqualTo(Float.floatToRawIntBits(sc128[i]));
							assertThat(Float.floatToRawIntBits(off256[i])).as("type %d offset %d", type, i)
									.isEqualTo(Float.floatToRawIntBits(off128[i]));
						}
					}
		}
	}

	@ParameterizedTest(name = "type {0}, {1} rows x {2} cols, B={3}")
	@CsvSource({ "12, 37, 256, 1", "12, 21, 512, 7", "12, 17, 1024, 32", "13, 37, 256, 1", "13, 21, 512, 7",
			"13, 17, 1024, 32", "14, 37, 256, 1", "14, 21, 512, 7", "14, 17, 1024, 32", "8, 37, 256, 1", "8, 21, 512, 7",
			"8, 17, 1024, 32", "12, 5, 4096, 4", "14, 5, 4096, 4" })
	void prefillMatmulAgreesAcrossWidths(int type, int rows, int cols, int batch) {
		Random rnd = new Random(53L * type + batch);
		GgufReader.QuantizedTensor w = CpuIntegerKernelsTest.synthetic(type, rows, cols, rnd);
		float[][] X = new float[batch][];
		for (int b = 0; b < batch; b++)
			X[b] = activation(cols, rnd);
		float[][] y128 = new float[batch][rows];
		float[][] y256 = new float[batch][rows];
		CpuIntegerGemm.sgemmInto(w, X, y128, 0, rows, cols, false);
		CpuIntegerGemm.sgemmInto(w, X, y256, 0, rows, cols, true);
		double worst = 0;
		for (int b = 0; b < batch; b++) {
			CpuIntegerKernels.Q8Activation a = CpuIntegerKernels.Q8Activation.quantize(X[b], cols);
			for (int r = 0; r < rows; r++) {
				double m = gemmTermMagnitude(type, w.data(), r, cols, a);
				double err = m == 0 ? Math.abs(y256[b][r] - y128[b][r]) : Math.abs(y256[b][r] - y128[b][r]) / m;
				worst = Math.max(worst, err);
				assertThat(err).as("b %d row %d (128-bit %s, 256-bit %s, term magnitude %s)", b, r, y128[b][r], y256[b][r], m)
						.isLessThanOrEqualTo(BOUND);
			}
		}
		System.out.printf("type %d %d x %d B=%d: worst 256-bit difference / term magnitude %.2e%n", type, rows, cols, batch,
				worst);
	}

	/** Sum over row {@code r}'s sub-blocks of |scale x activation scale x dot| + |offset x activation scale x sum|. */
	private static double gemmTermMagnitude(int type, byte[] raw, int r, int cols, CpuIntegerKernels.Q8Activation a) {
		int width = type == 14 || type == 8 ? 16 : 32;
		int[] sums = width == 16 ? a.sum16 : a.sum32;
		short[] w = new short[256];
		float[] sc = new float[16];
		float[] off = new float[16];
		double m = 0;
		for (int kb = 0; kb < cols / 256; kb++) {
			CpuIntegerGemm.unpack(type, raw, r, kb, cols, w, 0, sc, off, 0);
			for (int j = 0; j < 256 / width; j++) {
				int c0 = kb * 256 + j * width;
				long dot = 0;
				for (int i = 0; i < width; i++)
					dot += (long) w[j * width + i] * a.q[c0 + i];
				double xd = a.d[c0 >> 5];
				m += Math.abs(sc[j] * xd * dot) + Math.abs(off[j] * xd * sums[c0 / width]);
			}
		}
		return m;
	}

	@Test
	void defaultEntryPointsRunTheSelectedWidth() {
		Random rnd = new Random(77);
		int rows = 9, cols = 512;
		for (int type : new int[] { 12, 14 }) {
			GgufReader.QuantizedTensor w = CpuIntegerKernelsTest.synthetic(type, rows, cols, rnd);
			float[] x = activation(cols, rnd);
			float[] dflt = new float[rows];
			float[] sel = new float[rows];
			CpuIntegerKernels.matVecInto(w, x, 0, rows, dflt, cols);
			CpuIntegerKernels.matVecInto(w, x, 0, rows, sel, cols, CpuIntegerWidth.WIDE);
			assertThat(dflt).isEqualTo(sel);
			float[][] X = { x, activation(cols, rnd) };
			float[][] gd = new float[2][rows];
			float[][] gs = new float[2][rows];
			CpuIntegerGemm.sgemmInto(w, X, gd, 0, rows, cols);
			CpuIntegerGemm.sgemmInto(w, X, gs, 0, rows, cols, CpuIntegerWidth.WIDE);
			assertThat(gd).isEqualTo(gs);
		}
	}

	private static void assertDecodeParity(int type, byte[] raw, int rows, int cols, float[] x) {
		GgufReader.QuantizedTensor w = new GgufReader.QuantizedTensor("extreme", type, (long) rows * cols, raw);
		float[] y128 = new float[rows];
		float[] y256 = new float[rows];
		CpuIntegerKernels.matVecInto(w, x, 0, rows, y128, cols, false);
		CpuIntegerKernels.matVecInto(w, x, 0, rows, y256, cols, true);
		if (type != 12) {
			for (int r = 0; r < rows; r++)
				assertThat(Float.floatToRawIntBits(y256[r])).as("type %d row %d (128-bit %s, 256-bit %s)", type, r, y128[r], y256[r])
						.isEqualTo(Float.floatToRawIntBits(y128[r]));
			return;
		}
		CpuIntegerKernels.Q8Activation a = CpuIntegerKernels.Q8Activation.quantize(x, cols);
		int blocksPerRow = cols / 256;
		for (int r = 0; r < rows; r++) {
			int off = r * blocksPerRow * 144;
			double m = CpuQ4kKernelParityTest.termMagnitude(raw, off, blocksPerRow, a.q, a.d, a.sum32);
			double err = m == 0 ? Math.abs(y256[r] - y128[r]) : Math.abs(y256[r] - y128[r]) / m;
			assertThat(err).as("Q4_K row %d (128-bit %s, 256-bit %s, term magnitude %s)", r, y128[r], y256[r], m)
					.isLessThanOrEqualTo(BOUND);
		}
	}

	/** Every byte of every block set to {@code fill}, with scales of 1.0 (and a minimum of 0.5) so the values stay finite. */
	private static byte[] extreme(int type, int rows, int cols, byte fill) {
		int blockBytes = switch (type) {
		case 12 -> 144;
		case 13 -> 176;
		case 14 -> 210;
		default -> 34;
		};
		int blocks = rows * cols / (type == 8 ? 32 : 256);
		byte[] raw = new byte[blocks * blockBytes];
		Arrays.fill(raw, fill);
		for (int k = 0; k < blocks; k++) {
			int bo = k * blockBytes;
			switch (type) {
			case 12, 13 -> {
				raw[bo] = 0x00;
				raw[bo + 1] = 0x3C;
				raw[bo + 2] = 0x00;
				raw[bo + 3] = 0x38;
			}
			case 14 -> {
				raw[bo + 208] = 0x00;
				raw[bo + 209] = 0x3C;
			}
			default -> {
				raw[bo] = 0x00;
				raw[bo + 1] = 0x3C;
			}
			}
		}
		return raw;
	}

	private static float[] activation(int cols, Random rnd) {
		float[] x = new float[cols];
		for (int i = 0; i < cols; i++)
			x[i] = (float) rnd.nextGaussian();
		for (int i = 0; i < cols / 128; i++)
			x[rnd.nextInt(cols)] *= 20f;
		return x;
	}
}
