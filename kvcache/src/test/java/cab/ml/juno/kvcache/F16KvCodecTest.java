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

import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * {@link F16KvCodec}: an {@code f16} KV row is its values' IEEE half-precision bits, two little-endian bytes each, the
 * rounding {@link Float#floatToFloat16} gives (round to nearest even, overflow to infinity, gradual underflow), and
 * reads back as {@link Float#float16ToFloat} of those bits; a head slice decodes to exactly what the whole row does.
 */
@DisplayName("F16KvCodec - KV rows as IEEE half floats")
class F16KvCodecTest {

	private static final int KV_DIM = 96;

	@Test
	@DisplayName("a row takes two bytes per value, through KvRowCodec as well")
	void rowBytes() {
		assertThat(F16KvCodec.encodedBytes(KV_DIM)).isEqualTo(2 * KV_DIM);
		assertThat(F16KvCodec.encodedBytes(33)).isEqualTo(66);
		assertThat(KvRowCodec.rowBytes(KvElementType.F16, KV_DIM)).isEqualTo(2 * KV_DIM);
	}

	@Test
	@DisplayName("each value is stored as floatToFloat16's bits, low byte first")
	void storesHalfBitsLittleEndian() {
		float[] src = values(KV_DIM, 3);
		byte[] dst = new byte[3 + 2 * KV_DIM];
		F16KvCodec.encode(src, 0, KV_DIM, dst, 3);
		for (int i = 0; i < KV_DIM; i++) {
			short bits = Float.floatToFloat16(src[i]);
			assertThat(dst[3 + 2 * i]).as("low byte of value %d", i).isEqualTo((byte) bits);
			assertThat(dst[3 + 2 * i + 1]).as("high byte of value %d", i).isEqualTo((byte) (bits >> 8));
		}
		byte[] one = new byte[2];
		F16KvCodec.encode(new float[] { 1f }, 0, 1, one, 0);
		assertThat(one).containsExactly(0x00, 0x3c);
	}

	@Test
	@DisplayName("decode gives float16ToFloat of the stored bits: halves round-trip exactly, others round to nearest")
	void decodesTheRounding() {
		float[] src = values(KV_DIM, 5);
		byte[] enc = new byte[2 * KV_DIM];
		F16KvCodec.encode(src, 0, KV_DIM, enc, 0);
		float[] got = new float[KV_DIM + 4];
		F16KvCodec.decode(enc, 0, got, 4, KV_DIM);
		for (int i = 0; i < KV_DIM; i++) {
			float want = Float.float16ToFloat(Float.floatToFloat16(src[i]));
			assertThat(Float.floatToRawIntBits(got[4 + i])).as("value %d (%s)", i, src[i])
					.isEqualTo(Float.floatToRawIntBits(want));
		}
		float[] again = new float[KV_DIM];
		byte[] enc2 = new byte[2 * KV_DIM];
		F16KvCodec.encode(got, 4, KV_DIM, enc2, 0);
		F16KvCodec.decode(enc2, 0, again, 0, KV_DIM);
		for (int i = 0; i < KV_DIM; i++)
			assertThat(Float.floatToRawIntBits(again[i])).isEqualTo(Float.floatToRawIntBits(got[4 + i]));
	}

	@Test
	@DisplayName("relative error at most 2^-11 over the half's normal range")
	void relativeError() {
		Random r = new Random(7);
		float[] src = new float[4096];
		for (int i = 0; i < src.length; i++)
			src[i] = (float) (r.nextGaussian() * Math.pow(10, r.nextInt(7) - 3));
		byte[] enc = new byte[2 * src.length];
		F16KvCodec.encode(src, 0, src.length, enc, 0);
		float[] got = new float[src.length];
		F16KvCodec.decode(enc, 0, got, 0, src.length);
		for (int i = 0; i < src.length; i++) {
			float a = Math.abs(src[i]);
			if (a >= 6.104e-5f && a <= 65504f)
				assertThat((double) Math.abs(got[i] - src[i])).as("value %s", src[i])
						.isLessThanOrEqualTo(a * Math.scalb(1.0, -11));
		}
	}

	@Test
	@DisplayName("zeros keep their sign; out-of-range values become infinities; tiny ones underflow gradually")
	void edges() {
		float[] src = { 0f, -0f, 65504f, 65520f, -1e9f, 5.96e-8f, 1e-9f, -3.0e-6f };
		byte[] enc = new byte[2 * src.length];
		F16KvCodec.encode(src, 0, src.length, enc, 0);
		float[] got = new float[src.length];
		F16KvCodec.decode(enc, 0, got, 0, src.length);
		assertThat(Float.floatToRawIntBits(got[0])).isZero();
		assertThat(Float.floatToRawIntBits(got[1])).isEqualTo(Float.floatToRawIntBits(-0f));
		assertThat(got[2]).isEqualTo(65504f);
		assertThat(got[3]).isEqualTo(Float.POSITIVE_INFINITY);
		assertThat(got[4]).isEqualTo(Float.NEGATIVE_INFINITY);
		assertThat(got[5]).isEqualTo(Float.float16ToFloat((short) 1));
		assertThat(got[6]).isZero();
		assertThat(got[7]).isEqualTo(Float.float16ToFloat(Float.floatToFloat16(-3.0e-6f))).isNegative();
	}

	@Test
	@DisplayName("decodeRange of any slice equals the whole-row decode, at any row offset")
	void decodeRangeMatchesWholeRow() {
		float[] src = values(KV_DIM, 11);
		int rowOff = 10;
		byte[] page = new byte[rowOff + 2 * KV_DIM];
		F16KvCodec.encode(src, 0, KV_DIM, page, rowOff);
		float[] whole = new float[KV_DIM];
		F16KvCodec.decode(page, rowOff, whole, 0, KV_DIM);
		for (int from = 0; from < KV_DIM; from += 7)
			for (int n = 1; from + n <= KV_DIM; n += 13) {
				float[] dst = new float[n + 2];
				F16KvCodec.decodeRange(page, rowOff, from, n, dst, 2);
				for (int i = 0; i < n; i++)
					assertThat(Float.floatToRawIntBits(dst[2 + i])).as("from %d n %d i %d", from, n, i)
							.isEqualTo(Float.floatToRawIntBits(whole[from + i]));
			}
	}

	@Test
	@DisplayName("KvRowCodec encodes and decodes f16 rows through it")
	void rowCodecDispatch() {
		float[] src = values(KV_DIM, 13);
		byte[] a = new byte[2 * KV_DIM];
		byte[] b = new byte[2 * KV_DIM];
		F16KvCodec.encode(src, 0, KV_DIM, a, 0);
		KvRowCodec.encode(KvElementType.F16, src, 0, KV_DIM, b, 0);
		assertThat(b).isEqualTo(a);
		float[] x = new float[KV_DIM];
		float[] y = new float[KV_DIM];
		F16KvCodec.decode(a, 0, x, 0, KV_DIM);
		KvRowCodec.decode(KvElementType.F16, a, 0, y, 0, KV_DIM);
		assertThat(y).isEqualTo(x);
	}

	static float[] values(int n, long seed) {
		Random r = new Random(seed);
		float[] v = new float[n];
		for (int i = 0; i < n; i++)
			v[i] = (float) (r.nextGaussian() * 3.0);
		return v;
	}
}
