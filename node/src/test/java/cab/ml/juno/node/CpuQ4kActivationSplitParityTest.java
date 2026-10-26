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

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Random;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Bit-for-bit parity of the Q4_K decode kernel that reads the activation's
 * even and odd bytes from split, sign-extended 16-bit arrays with the kernel
 * that split them per row in vector lanes.
 *
 * <p>The split moves where each activation value is sign-extended, not what
 * is multiplied or in what order it is summed, so every integer sum and float
 * step is the same and the row's bits must match. The oracle
 * {@link #q4kRowShipped} is a verbatim copy of the replaced kernel.
 */
class CpuQ4kActivationSplitParityTest {

	private static final int Q4_K = 12;
	private static final VectorSpecies<Byte> B = ByteVector.SPECIES_128;
	private static final VectorSpecies<Short> S = ShortVector.SPECIES_128;
	private static final VectorSpecies<Float> F = FloatVector.SPECIES_128;
	private static final VarHandle LE_LONG = MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN);
	private static final VarHandle LE_INT = MethodHandles.byteArrayViewVarHandle(int[].class, ByteOrder.LITTLE_ENDIAN);

	@Test
	void activationSplitHoldsEvenAndOddBytesSignExtended() {
		int cols = 512;
		float[] x = new float[cols];
		for (int i = 0; i < cols; i++)
			x[i] = i % 3 == 0 ? -3.5f : (i % 3 == 1 ? 3.5f : 0.01f * i);
		CpuIntegerKernels.Q8Activation a = CpuIntegerKernels.Q8Activation.quantize(x, cols);
		for (int i = 0; i < cols / 2; i++) {
			assertThat(a.xe[i]).as("even %d", i).isEqualTo((short) a.q[2 * i]);
			assertThat(a.xo[i]).as("odd %d", i).isEqualTo((short) a.q[2 * i + 1]);
		}
		// a reused activation of a narrower row is split again over its own width
		CpuIntegerKernels.Q8Activation reused = CpuIntegerKernels.Q8Activation.quantize(new float[1024], 1024);
		reused.quantizeFrom(x, cols);
		assertThat(Arrays.copyOf(reused.xe, cols / 2)).isEqualTo(Arrays.copyOf(a.xe, cols / 2));
		assertThat(Arrays.copyOf(reused.xo, cols / 2)).isEqualTo(Arrays.copyOf(a.xo, cols / 2));
	}

	@ParameterizedTest(name = "{0} rows x {1} cols")
	@CsvSource({ "64, 256", "96, 4096", "40, 14336", "1, 256" })
	void rowKernelIsBitIdenticalToThePerRowSplit(int rows, int cols) {
		Random rnd = new Random(13L * rows + cols);
		GgufReader.QuantizedTensor w = CpuIntegerKernelsTest.synthetic(Q4_K, rows, cols, rnd);
		float[] x = new float[cols];
		for (int i = 0; i < cols; i++)
			x[i] = (float) rnd.nextGaussian();
		for (int i = 0; i < cols / 128; i++)
			x[rnd.nextInt(cols)] *= 20f;
		assertRowsBitIdentical(w.data(), rows, cols, x);
	}

	@Test
	void extremeWeightsAndActivationsAreBitIdentical() {
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
				Arrays.fill(raw, bo + 4, bo + 16, r % 2 == 0 ? (byte) 0xFF : (byte) 0x41);
				Arrays.fill(raw, bo + 16, bo + 144, r < 2 ? (byte) 0xFF : 0); // every weight 15, or 0
			}
		float[] plus = new float[cols];
		Arrays.fill(plus, 3.5f); // every block quantizes to +127
		float[] minus = new float[cols];
		Arrays.fill(minus, -3.5f); // and to -127
		float[] alternating = new float[cols];
		for (int i = 0; i < cols; i++)
			alternating[i] = i % 2 == 0 ? 3.5f : -3.5f;
		assertRowsBitIdentical(raw, rows, cols, plus);
		assertRowsBitIdentical(raw, rows, cols, minus);
		assertRowsBitIdentical(raw, rows, cols, alternating);
	}

	private static void assertRowsBitIdentical(byte[] raw, int rows, int cols, float[] x) {
		CpuIntegerKernels.Q8Activation a = CpuIntegerKernels.Q8Activation.quantize(x, cols);
		int blocksPerRow = cols / 256;
		for (int r = 0; r < rows; r++) {
			int off = r * blocksPerRow * 144;
			float want = q4kRowShipped(raw, off, blocksPerRow, a.q, a.d, a.sum32);
			float got = CpuIntegerKernels.q4kRow(raw, off, blocksPerRow, a.xe, a.xo, a.d, a.sum32);
			assertThat(Float.floatToRawIntBits(got)).as("row %d of %d x %d (want %s, got %s)", r, rows, cols, want, got)
					.isEqualTo(Float.floatToRawIntBits(want));
		}
	}

	static float f16(byte[] raw, int off) {
		return GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, off));
	}

	/** The Q4_K row kernel as shipped before the activation split, verbatim. */
	static float q4kRowShipped(byte[] raw, int rowByteOffset, int blocksPerRow, byte[] xq, float[] xd, int[] xs) {
		FloatVector accV = FloatVector.zero(F);
		float minTot = 0f;
		for (int b = 0; b < blocksPerRow; b++) {
			int bo = rowByteOffset + b * 144;
			float d = f16(raw, bo);
			float dmin = f16(raw, bo + 2);
			// scale byte k: k < 4 holds sc[k] and k 4..7 holds mn[k - 4] in its low 6 bits, and the top 2 bits of
			// sc[k + 4] (k < 4) or mn[k] (k 4..7); bytes 8..11 hold the low 4 bits of sc[4..7] and mn[4..7]
			long lo = (long) LE_LONG.get(raw, bo + 4);
			long hi = (int) LE_INT.get(raw, bo + 12) & 0xFFFFFFFFL;
			long low6 = lo & 0x3F3F3F3F3F3F3F3FL;
			long top2 = (lo >>> 6) & 0x0303030303030303L;
			long sc = (low6 & 0xFFFFFFFFL) | (((hi & 0x0F0F0F0FL) | ((top2 & 0xFFFFFFFFL) << 4)) << 32);
			long mn = (low6 >>> 32) | ((((hi >>> 4) & 0x0F0F0F0FL) | ((top2 >>> 32) << 4)) << 32);
			float minAcc = 0f;
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
				IntVector sLo = iLo.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
						.add(iLo.lanewise(VectorOperators.ASHR, 16));
				IntVector sHi = iHi.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
						.add(iHi.lanewise(VectorOperators.ASHR, 16));
				int ab = x0 >> 5;
				int sh = g * 16;
				float fLo = xd[ab] * d * (float) ((int) (sc >>> sh) & 0xFF);
				float fHi = xd[ab + 1] * d * (float) ((int) (sc >>> (sh + 8)) & 0xFF);
				accV = accV.add(((FloatVector) sLo.convertShape(VectorOperators.I2F, F, 0)).mul(fLo))
						.add(((FloatVector) sHi.convertShape(VectorOperators.I2F, F, 0)).mul(fHi));
				minAcc += xd[ab] * ((int) (mn >>> sh) & 0xFF) * xs[ab]
						+ xd[ab + 1] * ((int) (mn >>> (sh + 8)) & 0xFF) * xs[ab + 1];
			}
			minTot += dmin * minAcc;
		}
		return (accV.lane(0) + accV.lane(1)) + (accV.lane(2) + accV.lane(3)) - minTot;
	}
}
