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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Encodes and decodes activation tensors (float[]) to/from compressed bytes.
 *
 * Used exclusively at the gRPC transport boundary: - compress before writing
 * {@code ForwardRequest.activation} - decompress after reading
 * {@code ForwardResponse.activation}
 *
 * The rest of the pipeline (ForwardRequest record, ForwardResult record,
 * ForwardPassHandler) always works with float[] — compression is purely a
 * network-wire concern and is invisible to business logic.
 *
 * Wire layouts: FLOAT32 → big-endian IEEE 754 float, 4 bytes/element FLOAT16 →
 * big-endian IEEE 754 half-precision, 2 bytes/element INT8 → [scale:float32
 * big-endian (4 bytes)] [quantised:signed byte × N]
 *
 * Thread-safe: all methods are stateless.
 */
public final class ActivationCodec {

	private ActivationCodec() {
	} // utility class — no instances

	// ── Public API ────────────────────────────────────────────────────────────

	/**
	 * Encode {@code floats} into bytes using the requested dtype.
	 *
	 * @param floats source activation tensor (may be empty, must not be null)
	 * @param dtype  target wire format
	 * @return compressed byte representation; length depends on dtype
	 */
	public static byte[] encode(float[] floats, ActivationDtype dtype) {
		if (floats == null || floats.length == 0)
			return new byte[0];
		return switch (dtype) {
		case FLOAT32 -> encodeFloat32(floats);
		case FLOAT16 -> encodeFloat16(floats);
		case INT8 -> encodeInt8(floats);
		};
	}

	/**
	 * Decode {@code bytes} back to a float[]. Must be called with the same dtype
	 * that was used for encoding.
	 *
	 * @param bytes encoded bytes (from {@link #encode}); null or empty returns
	 *              float[0]
	 * @param dtype the format used during encoding
	 * @return reconstructed float array
	 */
	public static float[] decode(byte[] bytes, ActivationDtype dtype) {
		if (bytes == null || bytes.length == 0)
			return new float[0];
		return switch (dtype) {
		case FLOAT32 -> decodeFloat32(bytes);
		case FLOAT16 -> decodeFloat16(bytes);
		case INT8 -> decodeInt8(bytes);
		};
	}

	// ── FLOAT32 ───────────────────────────────────────────────────────────────

	private static byte[] encodeFloat32(float[] floats) {
		ByteBuffer buf = ByteBuffer.allocate(floats.length * 4).order(ByteOrder.BIG_ENDIAN);
		for (float f : floats)
			buf.putFloat(f);
		return buf.array();
	}

	private static float[] decodeFloat32(byte[] bytes) {
		ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN);
		float[] out = new float[bytes.length / 4];
		for (int i = 0; i < out.length; i++)
			out[i] = buf.getFloat();
		return out;
	}

	// ── FLOAT16 ───────────────────────────────────────────────────────────────

	/**
	 * Encode to IEEE 754 half-precision (FP16). 2 bytes per element, big-endian.
	 */
	private static byte[] encodeFloat16(float[] floats) {
		ByteBuffer buf = ByteBuffer.allocate(floats.length * 2).order(ByteOrder.BIG_ENDIAN);
		for (float f : floats)
			buf.putShort(floatToHalf(f));
		return buf.array();
	}

	private static float[] decodeFloat16(byte[] bytes) {
		ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN);
		float[] out = new float[bytes.length / 2];
		for (int i = 0; i < out.length; i++)
			out[i] = halfToFloat(buf.getShort());
		return out;
	}

	/**
	 * Convert IEEE 754 single-precision float to half-precision (16-bit).
	 *
	 * Bit layout (float32): s[1] e[8] m[23] Bit layout (float16): s[1] e[5] m[10]
	 *
	 * Handles: normals, subnormals, ±0, ±∞, NaN, overflow (→ ±∞), underflow (→ ±0).
	 *
	 * Package-private for unit testing.
	 */
	static short floatToHalf(float f) {
		int bits = Float.floatToIntBits(f);
		int sign = (bits >>> 16) & 0x8000;
		int exp = ((bits >>> 23) & 0xFF) - 127 + 15; // rebias exponent
		int mant = bits & 0x007FFFFF;

		if (exp <= 0) {
			// Subnormal or underflow to zero
			if (exp < -10)
				return (short) sign; // too small → ±0
			// Shift mantissa into FP16 subnormal range
			mant = (mant | 0x00800000) >> (1 - exp);
			return (short) (sign | (mant >> 13));
		}
		if (exp >= 31) {
			return (short) (sign | 0x7C00); // overflow → ±infinity
		}
		return (short) (sign | (exp << 10) | (mant >> 13));
	}

	/**
	 * Convert IEEE 754 half-precision (16-bit) to single-precision float.
	 *
	 * Package-private for unit testing.
	 */
	static float halfToFloat(short half) {
		int h = half & 0xFFFF;
		int sign = (h & 0x8000) << 16;
		int exp = (h >>> 10) & 0x1F;
		int mant = h & 0x03FF;

		if (exp == 0) {
			if (mant == 0)
				return Float.intBitsToFloat(sign); // ±0
			// Subnormal FP16 → normalised float32
			while ((mant & 0x0400) == 0) {
				mant <<= 1;
				exp--;
			}
			exp++;
			mant &= 0x03FF;
		} else if (exp == 31) {
			// ±Infinity or NaN — propagate mantissa
			return Float.intBitsToFloat(sign | 0x7F800000 | (mant << 13));
		}
		exp = (exp + 127 - 15) << 23; // rebias exponent
		mant = mant << 13;
		return Float.intBitsToFloat(sign | exp | mant);
	}

	// ── INT8 ──────────────────────────────────────────────────────────────────

	/**
	 * Symmetric INT8 quantisation.
	 *
	 * Wire layout:
	 * {@code [scale:float32 big-endian (4 bytes)][quantised:signed byte × N]}.
	 *
	 * Symmetric (not asymmetric) quantisation is used to keep the zero point at 0,
	 * which preserves true-zero activations and simplifies reconstruction. Using
	 * 127 (not 128) as the max avoids asymmetry of signed byte range.
	 */
	private static byte[] encodeInt8(float[] floats) {
		float maxAbs = 0f;
		for (float f : floats) {
			float abs = Math.abs(f);
			if (abs > maxAbs)
				maxAbs = abs;
		}
		// Guard against all-zero arrays: use scale=1 so reconstruction returns 0
		float scale = maxAbs == 0f ? 1f : maxAbs / 127f;

		ByteBuffer buf = ByteBuffer.allocate(4 + floats.length).order(ByteOrder.BIG_ENDIAN);
		buf.putFloat(scale);
		for (float f : floats) {
			int q = Math.round(f / scale);
			buf.put((byte) Math.max(-127, Math.min(127, q)));
		}
		return buf.array();
	}

	private static float[] decodeInt8(byte[] bytes) {
		ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN);
		float scale = buf.getFloat();
		float[] out = new float[bytes.length - 4];
		for (int i = 0; i < out.length; i++)
			out[i] = buf.get() * scale;
		return out;
	}
}