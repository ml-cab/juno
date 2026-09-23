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

/**
 * GGUF / ggml {@code Q8_0} block codec for KV tensors.
 *
 * <p>Block layout (34 bytes, little-endian): {@code fp16 d} + {@code int8 qs[32]}.
 * Dequant: {@code x[i] = float16ToFloat(d) * qs[i]}.
 */
public final class Q8_0KvCodec {

	public static final int QK = 32;
	public static final int BLOCK_BYTES = 34; // 2 + 32

	private Q8_0KvCodec() {
	}

	/** Encoded size for {@code nFloats} elements (pads the last block with zeros). */
	public static int encodedBytes(int nFloats) {
		if (nFloats < 0)
			throw new IllegalArgumentException("nFloats must be >= 0");
		int blocks = (nFloats + QK - 1) / QK;
		return blocks * BLOCK_BYTES;
	}

	/**
	 * Encode {@code n} floats from {@code src[srcOff..)} into {@code dst[dstOff..)}.
	 * Destination must have at least {@link #encodedBytes(int)} bytes from
	 * {@code dstOff}.
	 */
	public static void encode(float[] src, int srcOff, int n, byte[] dst, int dstOff) {
		checkRange(src.length, srcOff, n);
		int need = encodedBytes(n);
		if (dstOff < 0 || dstOff + need > dst.length)
			throw new IllegalArgumentException("dst too small for " + n + " floats");
		int full = n / QK;
		int rem = n % QK;
		int so = srcOff;
		int doff = dstOff;
		for (int b = 0; b < full; b++) {
			encodeBlock(src, so, QK, dst, doff);
			so += QK;
			doff += BLOCK_BYTES;
		}
		if (rem > 0) {
			float[] pad = new float[QK];
			System.arraycopy(src, so, pad, 0, rem);
			encodeBlock(pad, 0, QK, dst, doff);
		}
	}

	/**
	 * Decode {@code n} floats into {@code dst[dstOff..)} from packed
	 * {@code src[srcOff..)}.
	 */
	public static void decode(byte[] src, int srcOff, float[] dst, int dstOff, int n) {
		checkRange(dst.length, dstOff, n);
		int need = encodedBytes(n);
		if (srcOff < 0 || srcOff + need > src.length)
			throw new IllegalArgumentException("src too small for " + n + " floats");
		int full = n / QK;
		int rem = n % QK;
		int so = srcOff;
		int doff = dstOff;
		for (int b = 0; b < full; b++) {
			decodeBlock(src, so, dst, doff, QK);
			so += BLOCK_BYTES;
			doff += QK;
		}
		if (rem > 0) {
			float[] pad = new float[QK];
			decodeBlock(src, so, pad, 0, QK);
			System.arraycopy(pad, 0, dst, doff, rem);
		}
	}

	/** Max |encode→decode − original| over {@code n} elements. */
	public static float maxAbsError(float[] src, int srcOff, int n) {
		byte[] enc = new byte[encodedBytes(n)];
		encode(src, srcOff, n, enc, 0);
		float[] got = new float[n];
		decode(enc, 0, got, 0, n);
		float max = 0f;
		for (int i = 0; i < n; i++) {
			float e = Math.abs(got[i] - src[srcOff + i]);
			if (e > max)
				max = e;
		}
		return max;
	}

	/** Persistent size ratio float32 / q8_0 (about 3.8 for whole 32-value blocks). */
	public static double compressionRatioVsF32(int nFloats) {
		if (nFloats <= 0)
			throw new IllegalArgumentException("nFloats must be > 0");
		double f32 = (double) nFloats * Float.BYTES;
		return f32 / encodedBytes(nFloats);
	}

	private static void encodeBlock(float[] src, int srcOff, int n, byte[] dst, int dstOff) {
		float amax = 0f;
		for (int i = 0; i < n; i++) {
			float a = Math.abs(src[srcOff + i]);
			if (a > amax)
				amax = a;
		}
		float d = amax / 127f;
		float id = d > 0f ? 1f / d : 0f;
		short dBits = Float.floatToFloat16(d);
		dst[dstOff] = (byte) (dBits & 0xff);
		dst[dstOff + 1] = (byte) ((dBits >>> 8) & 0xff);
		for (int i = 0; i < n; i++) {
			int q = Math.round(src[srcOff + i] * id);
			if (q > 127)
				q = 127;
			if (q < -128)
				q = -128;
			dst[dstOff + 2 + i] = (byte) q;
		}
		for (int i = n; i < QK; i++)
			dst[dstOff + 2 + i] = 0;
	}

	private static void decodeBlock(byte[] src, int srcOff, float[] dst, int dstOff, int n) {
		int lo = src[srcOff] & 0xff;
		int hi = src[srcOff + 1] & 0xff;
		float d = Float.float16ToFloat((short) (lo | (hi << 8)));
		for (int i = 0; i < n; i++)
			dst[dstOff + i] = d * src[srcOff + 2 + i];
	}

	private static void checkRange(int len, int off, int n) {
		if (off < 0 || n < 0 || off + n > len)
			throw new IllegalArgumentException("range off=" + off + " n=" + n + " len=" + len);
	}
}
