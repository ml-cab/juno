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

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Exact parity of {@link CpuIntegerGemm}'s per-block weight unpack with the
 * scalar loop it replaced.
 *
 * <p>The unpack is pure integer work (nibble and bit-plane extraction, widened
 * to 16 bits) plus each sub-block's float factors, so the 16-bit weights and
 * the factors must equal the oracle's exactly, and every gemm output that
 * consumes them is unchanged. The oracle {@link #unpackScalar} is a verbatim
 * copy of the scalar unpack. The scratch is pre-filled with a sentinel, so a
 * missed or misplaced store shows as a difference.
 */
class CpuIntegerGemmUnpackParityTest {

	private static final int ROWS = 5;
	private static final int COLS = 1024;

	@ParameterizedTest(name = "type {0}")
	@ValueSource(ints = { 8, 12, 13, 14 })
	void randomBlocksUnpackExactly(int type) {
		GgufReader.QuantizedTensor t = CpuIntegerKernelsTestSupport.synthetic(type, ROWS, COLS, new Random(31L + type));
		assertEveryBlockMatches(type, t.data());
	}

	@ParameterizedTest(name = "type {0}")
	@ValueSource(ints = { 8, 12, 13, 14 })
	void extremeWeightBytesUnpackExactly(int type) {
		// Every weight byte at each of the values that exercise sign, top bit and nibble
		// edges; the scale bytes keep the synthetic tensor's finite values.
		for (byte fill : new byte[] { 0x00, (byte) 0xFF, (byte) 0x80, 0x7F, 0x0F, (byte) 0xF0, 0x55, (byte) 0xAA }) {
			byte[] raw = CpuIntegerKernelsTestSupport.synthetic(type, ROWS, COLS, new Random(fill)).data();
			fillWeightBytes(type, raw, fill);
			assertEveryBlockMatches(type, raw);
		}
	}

	private static void assertEveryBlockMatches(int type, byte[] raw) {
		int tileRows = CpuIntegerGemm.TILE_ROWS;
		for (int kb = 0; kb < COLS / 256; kb++) {
			short[] w = new short[tileRows * 256];
			short[] wRef = new short[tileRows * 256];
			float[] sc = new float[tileRows * 16];
			float[] scRef = new float[tileRows * 16];
			float[] off = new float[tileRows * 16];
			float[] offRef = new float[tileRows * 16];
			Arrays.fill(w, (short) 0x5A5A);
			Arrays.fill(wRef, (short) 0x5A5A);
			Arrays.fill(sc, Float.NaN);
			Arrays.fill(scRef, Float.NaN);
			Arrays.fill(off, Float.NaN);
			Arrays.fill(offRef, Float.NaN);
			for (int r = 0; r < ROWS; r++) {
				// tile slot r + 2, so neither the weight nor the factor offset is zero
				int slot = r + 2;
				CpuIntegerGemm.unpack(type, raw, r, kb, COLS, w, slot * 256, sc, off, slot * 16);
				unpackScalar(type, raw, r, kb, COLS, wRef, slot * 256, scRef, offRef, slot * 16);
			}
			assertThat(w).as("weights, type %d, block %d", type, kb).isEqualTo(wRef);
			assertThat(bits(sc)).as("scales, type %d, block %d", type, kb).isEqualTo(bits(scRef));
			assertThat(bits(off)).as("offsets, type %d, block %d", type, kb).isEqualTo(bits(offRef));
		}
	}

	private static void fillWeightBytes(int type, byte[] raw, byte fill) {
		switch (type) {
		case 8 -> {
			for (int o = 0; o < raw.length; o += 34)
				Arrays.fill(raw, o + 2, o + 34, fill);
		}
		case 12 -> {
			for (int o = 0; o < raw.length; o += 144)
				Arrays.fill(raw, o + 16, o + 144, fill);
		}
		case 13 -> {
			for (int o = 0; o < raw.length; o += 176)
				Arrays.fill(raw, o + 16, o + 176, fill);
		}
		case 14 -> {
			for (int o = 0; o < raw.length; o += 210)
				Arrays.fill(raw, o, o + 192, fill);
		}
		default -> throw new IllegalArgumentException("type " + type);
		}
	}

	private static int[] bits(float[] a) {
		int[] b = new int[a.length];
		for (int i = 0; i < a.length; i++)
			b[i] = Float.floatToRawIntBits(a[i]);
		return b;
	}

	private static float f16(byte[] raw, int off) {
		return GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, off));
	}

	/** Verbatim copy of the scalar unpack this test guards against. */
	private static void unpackScalar(int type, byte[] raw, int row, int kb, int cols, short[] w, int wo, float[] sc,
			float[] off, int so) {
		switch (type) {
		case 12 -> {
			int bo = (row * (cols / 256) + kb) * 144;
			int scBase = bo + 4;
			float d = f16(raw, bo);
			float dmin = f16(raw, bo + 2);
			for (int g = 0; g < 4; g++) {
				int q = bo + 16 + g * 32;
				int e = wo + g * 64;
				for (int i = 0; i < 32; i++) {
					int v = raw[q + i];
					w[e + i] = (short) (v & 0x0F);
					w[e + 32 + i] = (short) ((v >> 4) & 0x0F);
				}
				sc[so + 2 * g] = d * LlamaTransformerHandler.q4kScaleRaw(raw, scBase, 2 * g);
				sc[so + 2 * g + 1] = d * LlamaTransformerHandler.q4kScaleRaw(raw, scBase, 2 * g + 1);
				off[so + 2 * g] = dmin * LlamaTransformerHandler.q4kMinRaw(raw, scBase, 2 * g);
				off[so + 2 * g + 1] = dmin * LlamaTransformerHandler.q4kMinRaw(raw, scBase, 2 * g + 1);
			}
		}
		case 13 -> {
			int bo = (row * (cols / 256) + kb) * 176;
			int scBase = bo + 4;
			int qhBase = bo + 16;
			float d = f16(raw, bo);
			float dmin = f16(raw, bo + 2);
			for (int g = 0; g < 4; g++) {
				int q = bo + 48 + g * 32;
				int e = wo + g * 64;
				for (int i = 0; i < 32; i++) {
					int v = raw[q + i];
					int h = raw[qhBase + i];
					w[e + i] = (short) ((v & 0x0F) | (((h >> (2 * g)) & 1) << 4));
					w[e + 32 + i] = (short) (((v >> 4) & 0x0F) | (((h >> (2 * g + 1)) & 1) << 4));
				}
				sc[so + 2 * g] = d * LlamaTransformerHandler.q4kScaleRaw(raw, scBase, 2 * g);
				sc[so + 2 * g + 1] = d * LlamaTransformerHandler.q4kScaleRaw(raw, scBase, 2 * g + 1);
				off[so + 2 * g] = dmin * LlamaTransformerHandler.q4kMinRaw(raw, scBase, 2 * g);
				off[so + 2 * g + 1] = dmin * LlamaTransformerHandler.q4kMinRaw(raw, scBase, 2 * g + 1);
			}
		}
		case 14 -> {
			int bo = (row * (cols / 256) + kb) * 210;
			float d = f16(raw, bo + 208);
			for (int half = 0; half < 2; half++) {
				int qlOff = bo + half * 64;
				int qhOff = bo + 128 + half * 32;
				int scOff = bo + 192 + half * 8;
				int e = wo + half * 128;
				for (int l = 0; l < 32; l++) {
					int ql = raw[qlOff + l] & 0xFF;
					int ql2 = raw[qlOff + l + 32] & 0xFF;
					int qh = raw[qhOff + l] & 0xFF;
					w[e + l] = (short) ((ql & 0x0F) | ((qh & 3) << 4));
					w[e + l + 32] = (short) ((ql2 & 0x0F) | (((qh >> 2) & 3) << 4));
					w[e + l + 64] = (short) ((ql >> 4) | (((qh >> 4) & 3) << 4));
					w[e + l + 96] = (short) ((ql2 >> 4) | (((qh >> 6) & 3) << 4));
				}
				for (int jj = 0; jj < 8; jj++) {
					float s = d * raw[scOff + jj];
					sc[so + half * 8 + jj] = s;
					off[so + half * 8 + jj] = 32f * s;
				}
			}
		}
		case 8 -> {
			int bo = (row * (cols / 32) + kb * 8) * 34;
			for (int k = 0; k < 8; k++) {
				int o = bo + k * 34;
				float d = f16(raw, o);
				int e = wo + k * 32;
				for (int i = 0; i < 32; i++)
					w[e + i] = raw[o + 2 + i];
				sc[so + 2 * k] = d;
				sc[so + 2 * k + 1] = d;
				off[so + 2 * k] = 0f;
				off[so + 2 * k + 1] = 0f;
			}
		}
		default -> throw new IllegalArgumentException("type " + type);
		}
	}
}
