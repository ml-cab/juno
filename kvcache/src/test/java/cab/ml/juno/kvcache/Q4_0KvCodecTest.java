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
package cab.ml.juno.kvcache;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.Arrays;
import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * The GGUF {@code Q4_0} block as a KV row codec: 18 bytes per 32 values, an FP16 scale {@code d = max / -8} taken from
 * the element of largest magnitude, then 16 bytes holding element {@code j} in the low nibble and element
 * {@code j + 16} in the high nibble of byte {@code j}; a value decodes as {@code d * (q - 8)}.
 */
@DisplayName("Q4_0KvCodec")
class Q4_0KvCodecTest {

	@Test
	@DisplayName("encodedBytes pads to 32-value blocks of 18 bytes")
	void encodedBytesPadsToBlocks() {
		assertThat(Q4_0KvCodec.QK).isEqualTo(32);
		assertThat(Q4_0KvCodec.BLOCK_BYTES).isEqualTo(18);
		assertThat(Q4_0KvCodec.encodedBytes(0)).isEqualTo(0);
		assertThat(Q4_0KvCodec.encodedBytes(1)).isEqualTo(18);
		assertThat(Q4_0KvCodec.encodedBytes(32)).isEqualTo(18);
		assertThat(Q4_0KvCodec.encodedBytes(33)).isEqualTo(36);
		assertThat(Q4_0KvCodec.encodedBytes(256)).isEqualTo(8 * 18);
		assertThatThrownBy(() -> Q4_0KvCodec.encodedBytes(-1)).isInstanceOf(IllegalArgumentException.class);
	}

	/**
	 * A block worked by hand. The largest magnitude is -4 at element 0, so {@code d = -4 / -8 = 0.5} (FP16 0x3800) and
	 * every value below is a whole number of steps: each must take the nibble written next to it and decode exactly.
	 */
	@Test
	@DisplayName("a hand-worked block has the published byte layout and decodes exactly")
	void handWorkedBlock() {
		float[] x = new float[32];
		x[0] = -4f; // q 0
		x[1] = 2f; // q 12
		x[2] = 1f; // q 10
		x[16] = 3f; // q 14, high nibble of byte 0
		x[17] = -1f; // q 6, high nibble of byte 1
		byte[] enc = new byte[18];
		Q4_0KvCodec.encode(x, 0, 32, enc, 0);

		byte[] expected = new byte[18];
		expected[0] = 0x00; // FP16 0.5 = 0x3800, little-endian
		expected[1] = 0x38;
		expected[2] = (byte) 0xE0; // element 0: q 0, element 16: q 14
		expected[3] = (byte) 0x6C; // element 1: q 12, element 17: q 6
		expected[4] = (byte) 0x8A; // element 2: q 10, element 18 (zero): q 8
		for (int j = 3; j < 16; j++)
			expected[2 + j] = (byte) 0x88; // zeros: q 8 in both nibbles
		assertThat(enc).containsExactly(expected);

		float[] got = new float[32];
		Q4_0KvCodec.decode(enc, 0, got, 0, 32);
		for (int i = 0; i < 32; i++)
			assertThat(got[i]).as("element %d", i).isEqualTo(x[i]);
	}

	/**
	 * The scale takes the sign of the element of largest magnitude, so that element is exact and a value as large on
	 * the other side, which would need 16 steps, clamps to the 7 steps the nibble holds: an error of one step.
	 */
	@Test
	@DisplayName("the extreme decodes exactly and its opposite clamps to seven steps")
	void oppositeExtremeClamps() {
		float[] x = new float[32];
		x[0] = 4f;
		x[1] = -4f;
		byte[] enc = new byte[18];
		Q4_0KvCodec.encode(x, 0, 32, enc, 0);
		float[] got = new float[32];
		Q4_0KvCodec.decode(enc, 0, got, 0, 32);
		assertThat(got[0]).isEqualTo(4f);
		assertThat(got[1]).isEqualTo(-3.5f);
		assertThat(enc[2] & 0x0f).as("element 0 is q 0").isEqualTo(0);
		assertThat(enc[3] & 0x0f).as("element 1 clamps to q 15").isEqualTo(15);
	}

	/**
	 * Against the float rows the {@code f16} path stores: every value within one step of its block's scale (half a
	 * step when not clamped, plus the scale's FP16 rounding), and over Gaussian rows an error energy near the
	 * uniform-quantization model ({@code step^2 / 12} per value, the step being the block's largest magnitude over 8),
	 * coarser than {@code Q8_0} on the same rows.
	 */
	@Test
	@DisplayName("round trip against the float rows: within one step per block, RMS error as 16 levels give")
	void roundTripAgainstFloat() {
		Random rng = new Random(17);
		double err2 = 0;
		double x2 = 0;
		double err2q8 = 0;
		double model2 = 0;
		for (int n : new int[] { 32, 64, 80, 96, 128, 256 }) {
			for (int rep = 0; rep < 50; rep++) {
				float[] x = new float[n];
				for (int i = 0; i < n; i++)
					x[i] = (float) (rng.nextGaussian() * (1 + (i % 5)));
				byte[] enc = new byte[Q4_0KvCodec.encodedBytes(n)];
				Q4_0KvCodec.encode(x, 0, n, enc, 0);
				float[] got = new float[n];
				Q4_0KvCodec.decode(enc, 0, got, 0, n);
				byte[] enc8 = new byte[Q8_0KvCodec.encodedBytes(n)];
				Q8_0KvCodec.encode(x, 0, n, enc8, 0);
				float[] got8 = new float[n];
				Q8_0KvCodec.decode(enc8, 0, got8, 0, n);
				for (int b = 0; b < n; b += 32) {
					int blockEnd = Math.min(n, b + 32);
					float amax = 0f;
					for (int i = b; i < blockEnd; i++)
						amax = Math.max(amax, Math.abs(x[i]));
					float step = amax / 8f;
					for (int i = b; i < blockEnd; i++) {
						float e = Math.abs(got[i] - x[i]);
						assertThat(e).as("n %d, element %d: %f decodes to %f, step %f", n, i, x[i], got[i], step)
								.isLessThanOrEqualTo(step * 1.01f + 1e-6f);
						err2 += (double) e * e;
						model2 += (double) step * step / 12;
						x2 += (double) x[i] * x[i];
						err2q8 += (double) (got8[i] - x[i]) * (got8[i] - x[i]);
					}
				}
			}
		}
		double relRms = Math.sqrt(err2 / x2);
		double relRmsQ8 = Math.sqrt(err2q8 / x2);
		assertThat(err2 / model2).as("error energy over the uniform-quantization model (relative RMS %f)", relRms)
				.isBetween(0.8, 1.25);
		assertThat(relRms).as("coarser than Q8_0 (%f)", relRmsQ8).isGreaterThan(8 * relRmsQ8);
	}

	@Test
	@DisplayName("a zero block decodes to zeros; a short row pads its last block and decodes only its own values")
	void zerosAndPartialBlock() {
		float[] zeros = new float[32];
		byte[] enc = new byte[18];
		Q4_0KvCodec.encode(zeros, 0, 32, enc, 0);
		float[] got = new float[32];
		Arrays.fill(got, Float.NaN);
		Q4_0KvCodec.decode(enc, 0, got, 0, 32);
		for (float v : got)
			assertThat(v == 0f).as("%s is zero (a zero block's scale is -0, as the reference encoder gives)", v).isTrue();

		float[] src = { 0f, 0.5f, -0.25f, 1f, -1f };
		byte[] packed = new byte[Q4_0KvCodec.encodedBytes(src.length) + 4];
		Arrays.fill(packed, (byte) 0x5a);
		Q4_0KvCodec.encode(src, 0, src.length, packed, 2);
		assertThat(packed[0]).as("nothing written before dstOff").isEqualTo((byte) 0x5a);
		assertThat(packed[packed.length - 1]).as("nothing written past the row").isEqualTo((byte) 0x5a);
		float[] out = new float[src.length + 1];
		Arrays.fill(out, Float.NaN);
		Q4_0KvCodec.decode(packed, 2, out, 0, src.length);
		for (int i = 0; i < src.length; i++)
			assertThat(Math.abs(out[i] - src[i])).as("element %d", i).isLessThanOrEqualTo(1f / 8f);
		assertThat(out[src.length]).as("nothing decoded past n").isNaN();
	}

	/**
	 * Attention decodes one head's slice of a stored row at a time, and a slice can start and end anywhere in a block,
	 * on either side of the block's nibble split at 16. Every slice must read the values the whole-row decode gives,
	 * bit for bit, at the row's offset inside a larger page.
	 */
	@Test
	@DisplayName("decodeRange gives decode's values for every slice of a row, bit for bit")
	void decodeRangeMatchesDecode() {
		Random rng = new Random(43);
		for (int n : new int[] { 1, 15, 16, 17, 31, 32, 33, 48, 80, 96, 256 }) {
			float[] src = new float[n];
			for (int i = 0; i < n; i++)
				src[i] = (float) rng.nextGaussian() * (1 + (i % 7));
			int rowBytes = Q4_0KvCodec.encodedBytes(n);
			int rowOff = 3 * rowBytes;
			byte[] page = new byte[5 * rowBytes];
			Q4_0KvCodec.encode(src, 0, n, page, rowOff);
			float[] whole = new float[n];
			Q4_0KvCodec.decode(page, rowOff, whole, 0, n);
			float[] got = new float[n + 2];
			for (int from = 0; from < n; from++)
				for (int len = 1; from + len <= n; len++) {
					Arrays.fill(got, Float.NaN);
					Q4_0KvCodec.decodeRange(page, rowOff, from, len, got, 1);
					for (int i = 0; i < len; i++)
						if (Float.floatToRawIntBits(got[1 + i]) != Float.floatToRawIntBits(whole[from + i]))
							throw new AssertionError("n " + n + ", slice [" + from + ", " + (from + len) + "): element "
									+ (from + i) + " reads " + got[1 + i] + ", decode reads " + whole[from + i]);
					assertThat(got[0]).as("nothing written before dstOff").isNaN();
					assertThat(got[1 + len]).as("nothing written past the slice").isNaN();
				}
		}
	}

	@Test
	@DisplayName("about 7.1x smaller than float32 rows")
	void compressionAgainstFloat32() {
		assertThat(Q4_0KvCodec.compressionRatioVsF32(256)).isEqualTo(256.0 * 4 / (8 * 18));
		assertThat(KvElementType.Q4_0.bytesPerElement()).isEqualTo(18.0 / 32);
	}

	@Test
	@DisplayName("rejects ranges outside the arrays")
	void rejectsBadRanges() {
		float[] src = new float[8];
		assertThatThrownBy(() -> Q4_0KvCodec.encode(src, 0, 64, new byte[36], 0))
				.isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> Q4_0KvCodec.encode(src, 0, 8, new byte[17], 0))
				.isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> Q4_0KvCodec.decode(new byte[17], 0, new float[8], 0, 8))
				.isInstanceOf(IllegalArgumentException.class);
	}
}
