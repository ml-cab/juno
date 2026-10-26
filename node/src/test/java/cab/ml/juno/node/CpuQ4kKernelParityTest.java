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

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Parity of the Q4_K integer decode kernel with the kernel it replaced.
 *
 * <p>The integer sums per 32-weight sub-block are exact in both. The kernel
 * under test scales them in float lanes and sums the row once, where the
 * replaced kernel reduced each sub-block to a scalar and summed in row order,
 * so the two differ by float rounding only. The bound is stated, not
 * bit-identity: 1e-6 of the row's term magnitude, the sum of the absolute
 * values of its sub-block scale and minimum terms (computed in double). A row
 * that is a small difference of large terms carries rounding of the terms'
 * size, not of the result's. The kernel must also return the same
 * bits on every call, from the first (interpreted) to compiled ones, since its
 * lane sum order is fixed.
 */
class CpuQ4kKernelParityTest {

	private static final int Q4_K = 12;
	private static final double BOUND = 1e-6;
	private static final Path TINYLLAMA = Path.of("models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
	private static final VectorSpecies<Byte> B = ByteVector.SPECIES_128;
	private static final VectorSpecies<Short> S = ShortVector.SPECIES_128;

	@ParameterizedTest(name = "{0} rows x {1} cols")
	@CsvSource({ "64, 256", "96, 4096", "40, 14336", "1, 256" })
	void rowKernelMatchesTheReplacedKernelToFloatRounding(int rows, int cols) {
		Random rnd = new Random(11L * rows + cols);
		GgufReader.QuantizedTensor w = CpuIntegerKernelsTest.synthetic(Q4_K, rows, cols, rnd);
		assertRowsWithinBound(w.data(), rows, cols, activation(cols, rnd));
	}

	@Test
	void extremeWeightsAndActivationsStayWithinBound() {
		int cols = 512;
		int rows = 4;
		byte[] raw = new byte[rows * (cols / 256) * 144];
		for (int r = 0; r < rows; r++)
			for (int b = 0; b < cols / 256; b++) {
				int bo = (r * (cols / 256) + b) * 144;
				raw[bo] = 0x00;
				raw[bo + 1] = 0x3C; // d = 1.0
				raw[bo + 2] = 0x00;
				raw[bo + 3] = 0x38; // dmin = 0.5
				Arrays.fill(raw, bo + 4, bo + 16, r % 2 == 0 ? (byte) 0xFF : (byte) 0x41); // every scale and min 63, or mixed
				Arrays.fill(raw, bo + 16, bo + 144, r < 2 ? (byte) 0xFF : 0); // every weight 15, or 0
			}
		float[] plus = new float[cols];
		Arrays.fill(plus, 3.5f);
		float[] alternating = new float[cols];
		for (int i = 0; i < cols; i++)
			alternating[i] = i % 2 == 0 ? 3.5f : -3.5f;
		assertRowsWithinBound(raw, rows, cols, plus);
		assertRowsWithinBound(raw, rows, cols, alternating);
	}

	@Test
	void sameBitsOnEveryCall() {
		int cols = 4096;
		Random rnd = new Random(3);
		GgufReader.QuantizedTensor w = CpuIntegerKernelsTest.synthetic(Q4_K, 8, cols, rnd);
		CpuIntegerKernels.Q8Activation a = CpuIntegerKernels.Q8Activation.quantize(activation(cols, rnd), cols);
		int first = Float.floatToRawIntBits(CpuIntegerKernels.q4kRow(w.data(), 0, cols / 256, a.xe, a.xo, a.d, a.sum32));
		for (int i = 0; i < 20_000; i++)
			assertThat(Float.floatToRawIntBits(CpuIntegerKernels.q4kRow(w.data(), 0, cols / 256, a.xe, a.xo, a.d, a.sum32)))
					.as("call %d", i).isEqualTo(first);
	}

	static boolean tinyLlamaPresent() {
		return resolve(TINYLLAMA).toFile().exists();
	}

	@Test
	@EnabledIf("tinyLlamaPresent")
	void realModelRowsStayWithinBound() throws Exception {
		try (GgufReader r = GgufReader.open(resolve(TINYLLAMA))) {
			int checked = 0;
			for (String name : r.tensorNames()) {
				if (r.tensorType(name) != Q4_K || r.tensorDims(name).length != 2)
					continue;
				int cols = (int) r.tensorDims(name)[0];
				int rows = (int) Math.min(256, r.tensorDims(name)[1]);
				byte[] raw = Arrays.copyOf(r.tensorRaw(name).data(), rows * (cols / 256) * 144);
				assertRowsWithinBound(raw, rows, cols, activation(cols, new Random(name.hashCode())));
				if (++checked == 4)
					break;
			}
			assertThat(checked).as("Q4_K tensors found in the model").isPositive();
		}
	}

	private static void assertRowsWithinBound(byte[] raw, int rows, int cols, float[] x) {
		CpuIntegerKernels.Q8Activation a = CpuIntegerKernels.Q8Activation.quantize(x, cols);
		int blocksPerRow = cols / 256;
		double worst = 0;
		for (int r = 0; r < rows; r++) {
			int off = r * blocksPerRow * 144;
			float want = q4kRowReplaced(raw, off, blocksPerRow, a.q, a.d, a.sum32);
			float got = CpuIntegerKernels.q4kRow(raw, off, blocksPerRow, a.xe, a.xo, a.d, a.sum32);
			double m = termMagnitude(raw, off, blocksPerRow, a.q, a.d, a.sum32);
			double err = m == 0 ? Math.abs(got - want) : Math.abs(got - want) / m;
			worst = Math.max(worst, err);
			assertThat(err).as("row %d of %d x %d (want %s, got %s, term magnitude %s)", r, rows, cols, want, got, m)
					.isLessThanOrEqualTo(BOUND);
		}
		System.out.printf("Q4_K %d x %d: worst difference / term magnitude %.2e%n", rows, cols, worst);
	}

	/** Sum over the row's sub-blocks of |scale term| + |minimum term|, in double, from scalar unpacking. */
	static double termMagnitude(byte[] raw, int rowByteOffset, int blocksPerRow, byte[] xq, float[] xd, int[] xs) {
		double m = 0;
		for (int b = 0; b < blocksPerRow; b++) {
			int bo = rowByteOffset + b * 144;
			double d = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo));
			double dmin = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo + 2));
			for (int j = 0; j < 8; j++) {
				int g = j / 2;
				int q = bo + 16 + g * 32;
				int x0 = b * 256 + g * 64 + (j % 2) * 32;
				long dot = 0;
				for (int i = 0; i < 32; i++) {
					int v = raw[q + i] & 0xFF;
					dot += (long) ((j % 2 == 0) ? (v & 0x0F) : (v >> 4)) * xq[x0 + i];
				}
				int ab = x0 >> 5;
				m += Math.abs(xd[ab] * d * LlamaTransformerHandler.q4kScaleRaw(raw, bo + 4, j) * dot)
						+ Math.abs(xd[ab] * dmin * LlamaTransformerHandler.q4kMinRaw(raw, bo + 4, j) * xs[ab]);
			}
		}
		return m;
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

	/** The Q4_K integer row kernel as shipped before the float-lane form, verbatim. */
	static float q4kRowReplaced(byte[] raw, int rowByteOffset, int blocksPerRow, byte[] xq, float[] xd, int[] xs) {
		float acc = 0f;
		for (int b = 0; b < blocksPerRow; b++) {
			int bo = rowByteOffset + b * 144;
			int scBase = bo + 4;
			float d = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo));
			float dmin = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo + 2));
			for (int g = 0; g < 4; g++) {
				int q = bo + 16 + g * 32;
				int x0 = b * 256 + g * 64;
				ShortVector aLo = ShortVector.zero(S);
				ShortVector aHi = ShortVector.zero(S);
				for (int h = 0; h < 32; h += 16) {
					ShortVector w = ByteVector.fromArray(B, raw, q + h).reinterpretAsShorts();
					ShortVector xl = ByteVector.fromArray(B, xq, x0 + h).reinterpretAsShorts();
					ShortVector xh = ByteVector.fromArray(B, xq, x0 + 32 + h).reinterpretAsShorts();
					aLo = aLo.add(w.and((short) 0x0F)
							.mul(xl.lanewise(VectorOperators.LSHL, 8).lanewise(VectorOperators.ASHR, 8)))
							.add(w.lanewise(VectorOperators.LSHR, 8).and((short) 0x0F)
									.mul(xl.lanewise(VectorOperators.ASHR, 8)));
					aHi = aHi.add(w.lanewise(VectorOperators.LSHR, 4).and((short) 0x0F)
							.mul(xh.lanewise(VectorOperators.LSHL, 8).lanewise(VectorOperators.ASHR, 8)))
							.add(w.lanewise(VectorOperators.LSHR, 12).mul(xh.lanewise(VectorOperators.ASHR, 8)));
				}
				IntVector iLo = aLo.reinterpretAsInts();
				IntVector iHi = aHi.reinterpretAsInts();
				int dotLo = iLo.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
						.add(iLo.lanewise(VectorOperators.ASHR, 16)).reduceLanes(VectorOperators.ADD);
				int dotHi = iHi.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
						.add(iHi.lanewise(VectorOperators.ASHR, 16)).reduceLanes(VectorOperators.ADD);
				int ab = x0 >> 5;
				acc += xd[ab] * (d * LlamaTransformerHandler.q4kScaleRaw(raw, scBase, 2 * g) * dotLo
						- dmin * LlamaTransformerHandler.q4kMinRaw(raw, scBase, 2 * g) * xs[ab]);
				acc += xd[ab + 1] * (d * LlamaTransformerHandler.q4kScaleRaw(raw, scBase, 2 * g + 1) * dotHi
						- dmin * LlamaTransformerHandler.q4kMinRaw(raw, scBase, 2 * g + 1) * xs[ab + 1]);
			}
		}
		return acc;
	}
}
