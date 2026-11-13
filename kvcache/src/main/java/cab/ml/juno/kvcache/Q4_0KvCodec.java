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
 * GGUF {@code Q4_0} block codec for KV tensors.
 *
 * <p>Block layout (18 bytes, little-endian): {@code fp16 d} + {@code uint8 qs[16]}. Byte {@code j} of {@code qs} holds
 * element {@code j} in its low nibble and element {@code j + 16} in its high nibble. Dequant:
 * {@code x[i] = float16ToFloat(d) * (q[i] - 8)}. Encoding takes the element of largest magnitude, keeping its sign,
 * as {@code max} and sets {@code d = max / -8}, so that element is stored exactly as step {@code -8} and a value as
 * large on the other side clamps to step {@code 7}; every other value rounds to the nearest of the 16 steps.
 */
public final class Q4_0KvCodec {

	public static final int QK = 32;
	public static final int BLOCK_BYTES = 18; // 2 + 16

	private Q4_0KvCodec() {
	}

	/** Encoded size for {@code nFloats} elements (pads the last block with zeros). */
	public static int encodedBytes(int nFloats) {
		if (nFloats < 0)
			throw new IllegalArgumentException("nFloats must be >= 0");
		return (nFloats + QK - 1) / QK * BLOCK_BYTES;
	}

	/**
	 * Encode {@code n} floats from {@code src[srcOff..)} into {@code dst[dstOff..)}. Destination must have at least
	 * {@link #encodedBytes(int)} bytes from {@code dstOff}.
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
			encodeBlock(src, so, dst, doff);
			so += QK;
			doff += BLOCK_BYTES;
		}
		if (rem > 0) {
			float[] pad = new float[QK];
			System.arraycopy(src, so, pad, 0, rem);
			encodeBlock(pad, 0, dst, doff);
		}
	}

	/** Decode {@code n} floats into {@code dst[dstOff..)} from packed {@code src[srcOff..)}. */
	public static void decode(byte[] src, int srcOff, float[] dst, int dstOff, int n) {
		checkRange(dst.length, dstOff, n);
		int need = encodedBytes(n);
		if (srcOff < 0 || srcOff + need > src.length)
			throw new IllegalArgumentException("src too small for " + n + " floats");
		decodeRange(src, srcOff, 0, n, dst, dstOff);
	}

	/**
	 * Decode elements {@code [from, from + n)} of the row encoded at {@code src[rowOff..)} into
	 * {@code dst[dstOff, dstOff + n)}: the values {@link #decode} gives for them, bit for bit (it is the same loop).
	 * Attention calls this with one head's slice of a stored row, which may start and end anywhere in a block. No range
	 * checks beyond the arrays' own: the caller is the attention loop.
	 */
	public static void decodeRange(byte[] src, int rowOff, int from, int n, float[] dst, int dstOff) {
		int e = from;
		int end = from + n;
		int j = dstOff - from;
		while (e < end) {
			int block = e / QK;
			int bo = rowOff + block * BLOCK_BYTES;
			float d = Float.float16ToFloat((short) ((src[bo] & 0xff) | ((src[bo + 1] & 0xff) << 8)));
			int base = block * QK;
			int qs = bo + 2;
			int lowStop = Math.min(end, base + QK / 2);
			for (; e < lowStop; e++)
				dst[j + e] = d * ((src[qs + e - base] & 0x0f) - 8);
			int stop = Math.min(end, base + QK);
			for (; e < stop; e++)
				dst[j + e] = d * (((src[qs + e - base - QK / 2] >> 4) & 0x0f) - 8);
		}
	}

	/** Persistent size ratio float32 / q4_0 (about 7.1 for whole 32-value blocks). */
	public static double compressionRatioVsF32(int nFloats) {
		if (nFloats <= 0)
			throw new IllegalArgumentException("nFloats must be > 0");
		return (double) nFloats * Float.BYTES / encodedBytes(nFloats);
	}

	private static void encodeBlock(float[] src, int srcOff, byte[] dst, int dstOff) {
		float amax = 0f;
		float max = 0f;
		for (int i = 0; i < QK; i++) {
			float v = src[srcOff + i];
			if (amax < Math.abs(v)) {
				amax = Math.abs(v);
				max = v;
			}
		}
		float d = max / -8f;
		float id = d != 0f ? 1f / d : 0f;
		short dBits = Float.floatToFloat16(d);
		dst[dstOff] = (byte) (dBits & 0xff);
		dst[dstOff + 1] = (byte) ((dBits >>> 8) & 0xff);
		for (int i = 0; i < QK / 2; i++) {
			int lo = Math.min(15, (int) (src[srcOff + i] * id + 8.5f));
			int hi = Math.min(15, (int) (src[srcOff + QK / 2 + i] * id + 8.5f));
			dst[dstOff + 2 + i] = (byte) (lo | (hi << 4));
		}
	}

	private static void checkRange(int len, int off, int n) {
		if (off < 0 || n < 0 || off + n > len)
			throw new IllegalArgumentException("range off=" + off + " n=" + n + " len=" + len);
	}
}
