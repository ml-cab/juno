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

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * Bit-for-bit parity of the Q6_K integer kernels with the scalar integer loop
 * they replaced.
 *
 * <p>Integer accumulation is exact, so a kernel that forms the same four
 * integer sums per 16-weight sub-block and keeps the scalar loop's float
 * scaling expression must return the same float bits. The oracle is a verbatim
 * copy of that scalar loop ({@link #q6kRowScalar}). The prefill matmul's Q6_K
 * path is covered by {@link CpuIntegerGemmTest} against an exact reference at
 * B = 1, 8, 128 and 741.
 */
class CpuQ6kKernelParityTest {

	private static final int Q6_K = 14;
	private static final Path TINYLLAMA = Path.of("models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");

	@ParameterizedTest(name = "{0} rows x {1} cols")
	@CsvSource({ "64, 256", "96, 4096", "40, 14336", "1, 256" })
	void rowKernelIsBitIdenticalToTheScalarLoop(int rows, int cols) {
		Random rnd = new Random(7L * rows + cols);
		GgufReader.QuantizedTensor w = CpuIntegerKernelsTest.synthetic(Q6_K, rows, cols, rnd);
		assertRowsBitIdentical(w.data(), rows, cols, activation(cols, rnd));
	}

	@Test
	void extremeWeightsAndActivationsAreBitIdentical() {
		int cols = 512;
		int rows = 4;
		byte[] raw = new byte[rows * (cols / 256) * 210];
		Random rnd = new Random(5);
		for (int r = 0; r < rows; r++)
			for (int b = 0; b < cols / 256; b++) {
				int bo = (r * (cols / 256) + b) * 210;
				// rows 0 and 1: every weight 63 (the largest product); rows 2 and 3: every weight 0 (the offset alone)
				byte fill = r < 2 ? (byte) 0xFF : 0;
				Arrays.fill(raw, bo, bo + 192, fill);
				for (int s = 0; s < 16; s++)
					raw[bo + 192 + s] = (byte) (r % 2 == 0 ? 127 : -128 + rnd.nextInt(3));
				raw[bo + 208] = 0x00;
				raw[bo + 209] = 0x3C; // 1.0 in FP16
			}
		float[] plus = new float[cols];
		float[] minus = new float[cols];
		Arrays.fill(plus, 3.5f); // every block quantizes to +127
		Arrays.fill(minus, -3.5f); // and to -127
		assertRowsBitIdentical(raw, rows, cols, plus);
		assertRowsBitIdentical(raw, rows, cols, minus);
		float[] alternating = new float[cols];
		for (int i = 0; i < cols; i++)
			alternating[i] = i % 2 == 0 ? 3.5f : -3.5f;
		assertRowsBitIdentical(raw, rows, cols, alternating);
	}

	@Test
	void fullDispatchIsBitIdenticalRowByRow() {
		int rows = 741;
		int cols = 2048;
		Random rnd = new Random(741);
		GgufReader.QuantizedTensor w = CpuIntegerKernelsTest.synthetic(Q6_K, rows, cols, rnd);
		float[] x = activation(cols, rnd);
		float[] y = new float[rows];
		CpuIntegerKernels.matVecInto(w, x, 0, rows, y, cols);
		CpuIntegerKernels.Q8Activation a = CpuIntegerKernels.Q8Activation.quantize(x, cols);
		for (int r = 0; r < rows; r++)
			assertThat(Float.floatToRawIntBits(y[r]))
					.as("row %d", r)
					.isEqualTo(Float.floatToRawIntBits(q6kRowScalar(w.data(), r * (cols / 256) * 210, cols / 256, a.q, a.d)));
	}

	static boolean tinyLlamaPresent() {
		return resolve(TINYLLAMA).toFile().exists();
	}

	@Test
	@EnabledIf("tinyLlamaPresent")
	void realModelRowsAreBitIdentical() throws Exception {
		try (GgufReader r = GgufReader.open(resolve(TINYLLAMA))) {
			int checked = 0;
			for (String name : r.tensorNames()) {
				if (r.tensorType(name) != Q6_K || r.tensorDims(name).length != 2)
					continue;
				int cols = (int) r.tensorDims(name)[0];
				int rows = (int) Math.min(256, r.tensorDims(name)[1]);
				GgufReader.QuantizedTensor t = r.tensorRaw(name);
				byte[] raw = Arrays.copyOf(t.data(), rows * (cols / 256) * 210);
				assertRowsBitIdentical(raw, rows, cols, activation(cols, new Random(name.hashCode())));
				if (++checked == 4)
					break;
			}
			assertThat(checked).as("Q6_K tensors found in the model").isPositive();
		}
	}

	private static void assertRowsBitIdentical(byte[] raw, int rows, int cols, float[] x) {
		CpuIntegerKernels.Q8Activation a = CpuIntegerKernels.Q8Activation.quantize(x, cols);
		int blocksPerRow = cols / 256;
		for (int r = 0; r < rows; r++) {
			int off = r * blocksPerRow * 210;
			float want = q6kRowScalar(raw, off, blocksPerRow, a.q, a.d);
			float got = CpuIntegerKernels.q6kRow(raw, off, blocksPerRow, a.q, a.d, a.sum16);
			assertThat(Float.floatToRawIntBits(got)).as("row %d of %d x %d (want %s, got %s)", r, rows, cols, want, got)
					.isEqualTo(Float.floatToRawIntBits(want));
		}
	}

	private static Path resolve(Path p) {
		return p.toFile().exists() ? p : Path.of("..").resolve(p);
	}

	private static float[] activation(int cols, Random rnd) {
		float[] x = new float[cols];
		for (int i = 0; i < cols; i++)
			x[i] = (float) rnd.nextGaussian();
		for (int i = 0; i < cols / 128; i++)
			x[rnd.nextInt(cols)] *= 20f;
		return x;
	}

	/** The scalar integer Q6_K row kernel as shipped before the vector kernel, verbatim. */
	static float q6kRowScalar(byte[] raw, int rowByteOffset, int blocksPerRow, byte[] xq, float[] xd) {
		float acc = 0f;
		for (int b = 0; b < blocksPerRow; b++) {
			int bo = rowByteOffset + b * 210;
			float d = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo + 208));
			for (int half = 0; half < 2; half++) {
				int qlOff = bo + half * 64;
				int qhOff = bo + 128 + half * 32;
				int scOff = bo + 192 + half * 8;
				int xOff = b * 256 + half * 128;
				for (int is = 0; is < 2; is++) {
					int s1 = 0, s2 = 0, s3 = 0, s4 = 0;
					for (int l = is * 16; l < is * 16 + 16; l++) {
						int qlL = raw[qlOff + l] & 0xFF;
						int qlL2 = raw[qlOff + l + 32] & 0xFF;
						int qhL = raw[qhOff + l] & 0xFF;
						s1 += (((qlL & 0x0F) | ((qhL & 3) << 4)) - 32) * xq[xOff + l];
						s2 += (((qlL2 & 0x0F) | (((qhL >> 2) & 3) << 4)) - 32) * xq[xOff + l + 32];
						s3 += (((qlL >> 4) | (((qhL >> 4) & 3) << 4)) - 32) * xq[xOff + l + 64];
						s4 += (((qlL2 >> 4) | (((qhL >> 6) & 3) << 4)) - 32) * xq[xOff + l + 96];
					}
					int ab = (xOff + is * 16) >> 5;
					acc += d * (raw[scOff + is] * xd[ab] * s1 + raw[scOff + is + 2] * xd[ab + 1] * s2
							+ raw[scOff + is + 4] * xd[ab + 2] * s3 + raw[scOff + is + 6] * xd[ab + 3] * s4);
				}
			}
		}
		return acc;
	}
}
