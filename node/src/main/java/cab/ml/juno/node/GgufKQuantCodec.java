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
import java.util.Arrays;

/**
 * Versioned Q4_K / Q5_K / Q6_K encode and decode codecs for GGUF tensors.
 *
 * <p>Decoder math matches llama.cpp {@code dequantize_row_q*_K}. Encoder strategy
 * ID is {@link #ENCODER_ID} — do not assume binary identity with other encoders
 * without differential tests.
 *
 * <p>Thread-safe: all methods are static and stateless.
 */
public final class GgufKQuantCodec {

	/**
	 * Named encoder strategy. Change only when encode semantics change intentionally.
	 */
	public static final String ENCODER_ID = "juno-kquant-v1";

	private GgufKQuantCodec() {}

	// ── Public facade ─────────────────────────────────────────────────────────

	/** Decode a contiguous K-quant tensor ({@code nelems} must be block-aligned). */
	public static float[] decode(byte[] raw, int typeId) {
		QuantizationLayout layout = QuantizationLayout.require(typeId);
		validateRaw(raw, layout, -1);
		long nelems = (raw.length / (long) layout.blockBytes()) * layout.blockWidth();
		layout.validateElementCount(nelems);
		return switch (typeId) {
			case QuantizationLayout.TYPE_Q4_K -> decodeQ4K(raw, (int) nelems);
			case QuantizationLayout.TYPE_Q5_K -> decodeQ5K(raw, (int) nelems);
			case QuantizationLayout.TYPE_Q6_K -> decodeQ6K(raw, (int) nelems);
			default -> throw new IllegalArgumentException("unsupported type " + typeId);
		};
	}

	/**
	 * Decode a row-major matrix where each row is an integer number of K-blocks.
	 * Matches {@link LlamaTransformerHandler} weight layout.
	 */
	public static float[] decodeRows(byte[] raw, int typeId, int rows, int cols) {
		QuantizationLayout layout = QuantizationLayout.require(typeId);
		layout.validateMatrix(rows, cols);
		long expected = layout.encodedBytes((long) rows * cols);
		if (raw == null || raw.length != expected) {
			throw new IllegalArgumentException(
					layout.name() + ": raw length " + (raw == null ? -1 : raw.length)
							+ " != expected " + expected);
		}
		return switch (typeId) {
			case QuantizationLayout.TYPE_Q4_K -> decodeQ4KRows(raw, rows, cols);
			case QuantizationLayout.TYPE_Q5_K -> decodeQ5KRows(raw, rows, cols);
			case QuantizationLayout.TYPE_Q6_K -> decodeQ6KRows(raw, rows, cols);
			default -> throw new IllegalArgumentException("unsupported type " + typeId);
		};
	}

	/** Encode floats with the named {@link #ENCODER_ID} strategy. */
	public static byte[] encode(float[] data, int typeId) {
		if (data == null) {
			throw new IllegalArgumentException("data must be non-null");
		}
		QuantizationLayout layout = QuantizationLayout.require(typeId);
		layout.validateElementCount(data.length);
		return switch (typeId) {
			case QuantizationLayout.TYPE_Q4_K -> encodeQ4K(data);
			case QuantizationLayout.TYPE_Q5_K -> encodeQ5K(data);
			case QuantizationLayout.TYPE_Q6_K -> encodeQ6K(data);
			default -> throw new IllegalArgumentException("unsupported type " + typeId);
		};
	}

	/**
	 * No-op / sidecar path: return a defensive copy of raw bytes without
	 * decode/re-encode. Required for byte-identical preservation.
	 */
	public static byte[] copyRawUnchanged(byte[] raw) {
		if (raw == null) {
			throw new IllegalArgumentException("raw must be non-null");
		}
		return Arrays.copyOf(raw, raw.length);
	}

	// ── Contiguous decode ─────────────────────────────────────────────────────

	public static float[] decodeQ4K(byte[] raw, int nelems) {
		QuantizationLayout.Q4_K.validateElementCount(nelems);
		validateRaw(raw, QuantizationLayout.Q4_K, nelems);
		float[] out = new float[nelems];
		final int QK = QuantizationLayout.QK_K;
		final int BB = QuantizationLayout.Q4_K.blockBytes();
		int nBlocks = nelems / QK;
		int oi = 0;
		for (int b = 0; b < nBlocks; b++) {
			int bo = b * BB;
			float d = GgufReader.f16ToF32(readLE16(raw, bo));
			float dmin = GgufReader.f16ToF32(readLE16(raw, bo + 2));
			int scBase = bo + 4;
			int qsBase = bo + 16;
			int qi = 0;
			for (int g = 0; g < QK; g += 64) {
				int s0 = g / 32, s1 = s0 + 1;
				float scale0 = d * scale4K(raw, scBase, s0);
				float min0 = dmin * min4K(raw, scBase, s0);
				float scale1 = d * scale4K(raw, scBase, s1);
				float min1 = dmin * min4K(raw, scBase, s1);
				for (int i = 0; i < 32; i++)
					out[oi++] = scale0 * (raw[qsBase + qi + i] & 0x0F) - min0;
				for (int i = 0; i < 32; i++)
					out[oi++] = scale1 * ((raw[qsBase + qi + i] >> 4) & 0x0F) - min1;
				qi += 32;
			}
		}
		return out;
	}

	public static float[] decodeQ5K(byte[] raw, int nelems) {
		QuantizationLayout.Q5_K.validateElementCount(nelems);
		validateRaw(raw, QuantizationLayout.Q5_K, nelems);
		float[] out = new float[nelems];
		final int QK = QuantizationLayout.QK_K;
		final int BB = QuantizationLayout.Q5_K.blockBytes();
		int nBlocks = nelems / QK;
		int oi = 0;
		for (int b = 0; b < nBlocks; b++) {
			int bo = b * BB;
			float d = GgufReader.f16ToF32(readLE16(raw, bo));
			float dmin = GgufReader.f16ToF32(readLE16(raw, bo + 2));
			int scBase = bo + 4;
			int qhBase = bo + 16;
			int qsBase = bo + 48;
			int qi = 0;
			for (int g = 0; g < 4; g++) {
				int s0 = g * 2, s1 = s0 + 1;
				float scale0 = d * scale4K(raw, scBase, s0);
				float min0 = dmin * min4K(raw, scBase, s0);
				float scale1 = d * scale4K(raw, scBase, s1);
				float min1 = dmin * min4K(raw, scBase, s1);
				int hiBit0 = g * 2, hiBit1 = g * 2 + 1;
				for (int l = 0; l < 32; l++) {
					int lo = raw[qsBase + qi + l] & 0x0F;
					int hi = (raw[qhBase + l] >>> hiBit0) & 1;
					out[oi++] = scale0 * (lo | (hi << 4)) - min0;
				}
				for (int l = 0; l < 32; l++) {
					int lo = (raw[qsBase + qi + l] >>> 4) & 0x0F;
					int hi = (raw[qhBase + l] >>> hiBit1) & 1;
					out[oi++] = scale1 * (lo | (hi << 4)) - min1;
				}
				qi += 32;
			}
		}
		return out;
	}

	public static float[] decodeQ6K(byte[] raw, int nelems) {
		QuantizationLayout.Q6_K.validateElementCount(nelems);
		validateRaw(raw, QuantizationLayout.Q6_K, nelems);
		float[] out = new float[nelems];
		final int QK = QuantizationLayout.QK_K;
		final int BB = QuantizationLayout.Q6_K.blockBytes();
		int nBlocks = nelems / QK;
		int oi = 0;
		for (int b = 0; b < nBlocks; b++) {
			int bo = b * BB;
			float d = GgufReader.f16ToF32(readLE16(raw, bo + 208));
			for (int half = 0; half < 2; half++) {
				int qlOff = bo + half * 64;
				int qhOff = bo + 128 + half * 32;
				int scOff = bo + 192 + half * 8;
				for (int l = 0; l < 32; l++) {
					int is = l / 16;
					int qlL = raw[qlOff + l] & 0xFF;
					int qlL2 = raw[qlOff + l + 32] & 0xFF;
					int qhL = raw[qhOff + l] & 0xFF;
					int q1 = ((qlL & 0x0F) | (((qhL >> 0) & 3) << 4)) - 32;
					int q2 = ((qlL2 & 0x0F) | (((qhL >> 2) & 3) << 4)) - 32;
					int q3 = ((qlL >> 4) | (((qhL >> 4) & 3) << 4)) - 32;
					int q4 = ((qlL2 >> 4) | (((qhL >> 6) & 3) << 4)) - 32;
					float d1 = d * raw[scOff + is];
					float d2 = d * raw[scOff + is + 2];
					float d3 = d * raw[scOff + is + 4];
					float d4 = d * raw[scOff + is + 6];
					out[oi + l] = d1 * q1;
					out[oi + l + 32] = d2 * q2;
					out[oi + l + 64] = d3 * q3;
					out[oi + l + 96] = d4 * q4;
				}
				oi += 128;
			}
		}
		return out;
	}

	// ── Row-strided decode (weight matrices) ──────────────────────────────────

	public static float[] decodeQ4KRows(byte[] raw, int rows, int cols) {
		QuantizationLayout.Q4_K.validateMatrix(rows, cols);
		float[] out = new float[rows * cols];
		final int BLOCK_SIZE = QuantizationLayout.QK_K;
		final int BLOCK_BYTES = QuantizationLayout.Q4_K.blockBytes();
		int blocksPerRow = cols / BLOCK_SIZE;
		int bytesPerRow = blocksPerRow * BLOCK_BYTES;
		for (int r = 0; r < rows; r++) {
			int rowByteOff = r * bytesPerRow;
			int xBase = r * cols;
			for (int b = 0; b < blocksPerRow; b++) {
				int bo = rowByteOff + b * BLOCK_BYTES;
				int scBase = bo + 4;
				int qsBase = bo + 16;
				float d = GgufReader.f16ToF32(readLE16(raw, bo));
				float dmin = GgufReader.f16ToF32(readLE16(raw, bo + 2));
				int qi = 0;
				for (int g = 0; g < BLOCK_SIZE; g += 64) {
					int s0 = g / 32, s1 = s0 + 1;
					float scale0 = d * scale4K(raw, scBase, s0);
					float min0 = dmin * min4K(raw, scBase, s0);
					float scale1 = d * scale4K(raw, scBase, s1);
					float min1 = dmin * min4K(raw, scBase, s1);
					int outOff = xBase + b * BLOCK_SIZE + g;
					for (int i = 0; i < 32; i++)
						out[outOff + i] = scale0 * (raw[qsBase + qi + i] & 0x0F) - min0;
					for (int i = 0; i < 32; i++)
						out[outOff + 32 + i] = scale1 * ((raw[qsBase + qi + i] >> 4) & 0x0F) - min1;
					qi += 32;
				}
			}
		}
		return out;
	}

	public static float[] decodeQ5KRows(byte[] raw, int rows, int cols) {
		QuantizationLayout.Q5_K.validateMatrix(rows, cols);
		float[] out = new float[rows * cols];
		final int BLOCK_SIZE = QuantizationLayout.QK_K;
		final int BLOCK_BYTES = QuantizationLayout.Q5_K.blockBytes();
		int blocksPerRow = cols / BLOCK_SIZE;
		int bytesPerRow = blocksPerRow * BLOCK_BYTES;
		for (int r = 0; r < rows; r++) {
			int rowByteOff = r * bytesPerRow;
			int xBase = r * cols;
			for (int b = 0; b < blocksPerRow; b++) {
				int bo = rowByteOff + b * BLOCK_BYTES;
				int scBase = bo + 4;
				int qhBase = bo + 16;
				int qsBase = bo + 48;
				float d = GgufReader.f16ToF32(readLE16(raw, bo));
				float dmin = GgufReader.f16ToF32(readLE16(raw, bo + 2));
				int qi = 0;
				for (int g = 0; g < 4; g++) {
					int s0 = g * 2, s1 = s0 + 1;
					float scale0 = d * scale4K(raw, scBase, s0);
					float min0 = dmin * min4K(raw, scBase, s0);
					float scale1 = d * scale4K(raw, scBase, s1);
					float min1 = dmin * min4K(raw, scBase, s1);
					int hiBit0 = g * 2, hiBit1 = g * 2 + 1;
					int outOff = xBase + b * BLOCK_SIZE + g * 64;
					for (int l = 0; l < 32; l++) {
						int lo = raw[qsBase + qi + l] & 0x0F;
						int hi = (raw[qhBase + l] >>> hiBit0) & 1;
						out[outOff + l] = scale0 * (lo | (hi << 4)) - min0;
					}
					for (int l = 0; l < 32; l++) {
						int lo = (raw[qsBase + qi + l] >>> 4) & 0x0F;
						int hi = (raw[qhBase + l] >>> hiBit1) & 1;
						out[outOff + 32 + l] = scale1 * (lo | (hi << 4)) - min1;
					}
					qi += 32;
				}
			}
		}
		return out;
	}

	public static float[] decodeQ6KRows(byte[] raw, int rows, int cols) {
		QuantizationLayout.Q6_K.validateMatrix(rows, cols);
		float[] out = new float[rows * cols];
		final int BLOCK_SIZE = QuantizationLayout.QK_K;
		final int BLOCK_BYTES = QuantizationLayout.Q6_K.blockBytes();
		int blocksPerRow = cols / BLOCK_SIZE;
		int bytesPerRow = blocksPerRow * BLOCK_BYTES;
		for (int r = 0; r < rows; r++) {
			int rowByteOff = r * bytesPerRow;
			int xBase = r * cols;
			for (int b = 0; b < blocksPerRow; b++) {
				int bo = rowByteOff + b * BLOCK_BYTES;
				float d = GgufReader.f16ToF32(readLE16(raw, bo + 208));
				for (int half = 0; half < 2; half++) {
					int qlOff = bo + half * 64;
					int qhOff = bo + 128 + half * 32;
					int scOff = bo + 192 + half * 8;
					int xOff = xBase + b * BLOCK_SIZE + half * 128;
					for (int l = 0; l < 32; l++) {
						int is = l / 16;
						int qlL = raw[qlOff + l] & 0xFF;
						int qlL2 = raw[qlOff + l + 32] & 0xFF;
						int qhL = raw[qhOff + l] & 0xFF;
						int q1 = ((qlL & 0x0F) | (((qhL >> 0) & 3) << 4)) - 32;
						int q2 = ((qlL2 & 0x0F) | (((qhL >> 2) & 3) << 4)) - 32;
						int q3 = ((qlL >> 4) | (((qhL >> 4) & 3) << 4)) - 32;
						int q4 = ((qlL2 >> 4) | (((qhL >> 6) & 3) << 4)) - 32;
						float d1 = d * raw[scOff + is];
						float d2 = d * raw[scOff + is + 2];
						float d3 = d * raw[scOff + is + 4];
						float d4 = d * raw[scOff + is + 6];
						out[xOff + l] = d1 * q1;
						out[xOff + l + 32] = d2 * q2;
						out[xOff + l + 64] = d3 * q3;
						out[xOff + l + 96] = d4 * q4;
					}
				}
			}
		}
		return out;
	}

	// ── Encode (juno-kquant-v1) ───────────────────────────────────────────────

	public static byte[] encodeQ4K(float[] data) {
		int n = data.length;
		QuantizationLayout.Q4_K.validateElementCount(n);
		int QK_K = QuantizationLayout.QK_K;
		int nBlocks = n / QK_K;
		ByteBuffer buf = ByteBuffer.allocate(nBlocks * QuantizationLayout.Q4_K.blockBytes())
				.order(ByteOrder.LITTLE_ENDIAN);

		for (int b = 0; b < nBlocks; b++) {
			int base = b * QK_K;
			float[] mins = new float[8], ranges = new float[8];
			for (int s = 0; s < 8; s++) {
				float mn = Float.MAX_VALUE, mx = -Float.MAX_VALUE;
				for (int i = 0; i < 32; i++) {
					float v = data[base + s * 32 + i];
					if (v < mn) mn = v;
					if (v > mx) mx = v;
				}
				if (mn > mx) { mn = 0f; mx = 0f; }
				mins[s] = mn;
				ranges[s] = mx - mn;
			}
			float maxRange = 0f, maxAbsMin = 0f;
			for (int s = 0; s < 8; s++) {
				if (ranges[s] > maxRange) maxRange = ranges[s];
				float am = -mins[s];
				if (am > maxAbsMin) maxAbsMin = am;
			}

			float d = maxRange > 0f ? maxRange / (63f * 15f) : 0f;
			float dmin = maxAbsMin > 0f ? maxAbsMin / 63f : 0f;

			int[] ls = new int[8], lm = new int[8];
			for (int s = 0; s < 8; s++) {
				ls[s] = maxRange > 0f ? clamp6(Math.round(ranges[s] * 63f / maxRange)) : 0;
				lm[s] = maxAbsMin > 0f ? clamp6(Math.round(-mins[s] * 63f / maxAbsMin)) : 0;
			}

			buf.putShort(LoraMerge.f32ToF16(d));
			buf.putShort(LoraMerge.f32ToF16(dmin));

			byte[] sc = new byte[12];
			for (int j = 0; j < 4; j++) {
				sc[j] = (byte) ((ls[j] & 0x3F) | ((ls[j + 4] & 0x30) << 2));
				sc[j + 4] = (byte) ((lm[j] & 0x3F) | ((lm[j + 4] & 0x30) << 2));
				sc[j + 8] = (byte) ((ls[j + 4] & 0x0F) | ((lm[j + 4] & 0x0F) << 4));
			}
			buf.put(sc);

			byte[] qs = new byte[128];
			for (int g = 0; g < 4; g++) {
				int s0 = g * 2, s1 = s0 + 1;
				float sc0 = d * ls[s0], mn0 = dmin * lm[s0];
				float sc1 = d * ls[s1], mn1 = dmin * lm[s1];
				int qi = g * 32;
				for (int i = 0; i < 32; i++) {
					int q0 = sc0 > 0f ? clamp(Math.round((data[base + s0 * 32 + i] + mn0) / sc0), 0, 15) : 0;
					int q1 = sc1 > 0f ? clamp(Math.round((data[base + s1 * 32 + i] + mn1) / sc1), 0, 15) : 0;
					qs[qi + i] = (byte) (q0 | (q1 << 4));
				}
			}
			buf.put(qs);
		}
		return buf.array();
	}

	public static byte[] encodeQ5K(float[] data) {
		int n = data.length;
		QuantizationLayout.Q5_K.validateElementCount(n);
		int QK_K = QuantizationLayout.QK_K;
		int nBlocks = n / QK_K;
		ByteBuffer buf = ByteBuffer.allocate(nBlocks * QuantizationLayout.Q5_K.blockBytes())
				.order(ByteOrder.LITTLE_ENDIAN);

		for (int b = 0; b < nBlocks; b++) {
			int base = b * QK_K;
			float[] mins = new float[8], ranges = new float[8];
			for (int s = 0; s < 8; s++) {
				float mn = Float.MAX_VALUE, mx = -Float.MAX_VALUE;
				for (int i = 0; i < 32; i++) {
					float v = data[base + s * 32 + i];
					if (v < mn) mn = v;
					if (v > mx) mx = v;
				}
				if (mn > mx) { mn = 0f; mx = 0f; }
				mins[s] = mn;
				ranges[s] = mx - mn;
			}
			float maxRange = 0f, maxAbsMin = 0f;
			for (int s = 0; s < 8; s++) {
				if (ranges[s] > maxRange) maxRange = ranges[s];
				float am = -mins[s];
				if (am > maxAbsMin) maxAbsMin = am;
			}

			float d = maxRange > 0f ? maxRange / (63f * 31f) : 0f;
			float dmin = maxAbsMin > 0f ? maxAbsMin / 63f : 0f;

			int[] ls = new int[8], lm = new int[8];
			for (int s = 0; s < 8; s++) {
				ls[s] = maxRange > 0f ? clamp6(Math.round(ranges[s] * 63f / maxRange)) : 0;
				lm[s] = maxAbsMin > 0f ? clamp6(Math.round(-mins[s] * 63f / maxAbsMin)) : 0;
			}

			buf.putShort(LoraMerge.f32ToF16(d));
			buf.putShort(LoraMerge.f32ToF16(dmin));

			byte[] sc = new byte[12];
			for (int j = 0; j < 4; j++) {
				sc[j] = (byte) ((ls[j] & 0x3F) | ((ls[j + 4] & 0x30) << 2));
				sc[j + 4] = (byte) ((lm[j] & 0x3F) | ((lm[j + 4] & 0x30) << 2));
				sc[j + 8] = (byte) ((ls[j + 4] & 0x0F) | ((lm[j + 4] & 0x0F) << 4));
			}
			buf.put(sc);

			byte[] qh = new byte[32], qs = new byte[128];
			for (int g = 0; g < 4; g++) {
				int s0 = g * 2, s1 = s0 + 1;
				float sc0 = d * ls[s0], mn0 = dmin * lm[s0];
				float sc1 = d * ls[s1], mn1 = dmin * lm[s1];
				int qi = g * 32, hiBit0 = g * 2, hiBit1 = g * 2 + 1;
				for (int l = 0; l < 32; l++) {
					int q0 = sc0 > 0f ? clamp(Math.round((data[base + s0 * 32 + l] + mn0) / sc0), 0, 31) : 0;
					int q1 = sc1 > 0f ? clamp(Math.round((data[base + s1 * 32 + l] + mn1) / sc1), 0, 31) : 0;
					qs[qi + l] = (byte) ((q0 & 0x0F) | ((q1 & 0x0F) << 4));
					qh[l] |= (byte) (((q0 >> 4) & 1) << hiBit0);
					qh[l] |= (byte) (((q1 >> 4) & 1) << hiBit1);
				}
			}
			buf.put(qh);
			buf.put(qs);
		}
		return buf.array();
	}

	public static byte[] encodeQ6K(float[] data) {
		int n = data.length;
		QuantizationLayout.Q6_K.validateElementCount(n);
		int QK_K = QuantizationLayout.QK_K;
		int nBlocks = n / QK_K;
		ByteBuffer buf = ByteBuffer.allocate(nBlocks * QuantizationLayout.Q6_K.blockBytes())
				.order(ByteOrder.LITTLE_ENDIAN);
		for (int b = 0; b < nBlocks; b++) {
			int base = b * QK_K;
			float[] subMax = new float[16];
			for (int s = 0; s < 16; s++) {
				float m = 0f;
				for (int i = 0; i < 16; i++)
					m = Math.max(m, Math.abs(data[base + s * 16 + i]));
				subMax[s] = m;
			}
			float globalMax = 0f;
			for (float m : subMax)
				if (m > globalMax) globalMax = m;

			float d = globalMax > 0f ? globalMax / (127f * 32f) : 0f;
			float invD = d > 0f ? 1f / d : 0f;
			byte[] sc = new byte[16];
			for (int s = 0; s < 16; s++)
				sc[s] = (byte) (d > 0f ? clamp(Math.round(subMax[s] * invD / 32f), -127, 127) : 0);

			byte[] ql = new byte[128], qh = new byte[64];
			for (int p = 0; p < QK_K; p++) {
				int half = p / 128, lp = p % 128, quad = lp / 32, l = lp % 32;
				int sub = half * 8 + (l / 16) + quad * 2;
				float effScale = d * sc[sub];
				int q6 = effScale != 0f ? clamp(Math.round(data[base + p] / effScale), -32, 31) : 0;
				int unsigned = q6 + 32;
				int qlBase = half * 64, qhBase = half * 32;
				int qlIdx = qlBase + l + ((quad % 2 == 1) ? 32 : 0);
				if (quad < 2)
					ql[qlIdx] = (byte) ((ql[qlIdx] & 0xF0) | (unsigned & 0x0F));
				else
					ql[qlIdx] = (byte) ((ql[qlIdx] & 0x0F) | ((unsigned & 0x0F) << 4));
				int shift = quad * 2;
				qh[qhBase + l] = (byte) ((qh[qhBase + l] & ~(0x3 << shift))
						| (((unsigned >> 4) & 0x3) << shift));
			}
			buf.put(ql);
			buf.put(qh);
			buf.put(sc);
			buf.putShort(LoraMerge.f32ToF16(d));
		}
		return buf.array();
	}

	// ── Scale helpers (shared Q4_K / Q5_K packing) ────────────────────────────

	static int scale4K(byte[] sc, int scBase, int j) {
		if (j < 4)
			return sc[scBase + j] & 0x3F;
		return ((sc[scBase + j + 4] & 0x0F) | ((sc[scBase + j - 4] & 0xC0) >> 2)) & 0x3F;
	}

	static int min4K(byte[] sc, int scBase, int j) {
		if (j < 4)
			return sc[scBase + j + 4] & 0x3F;
		return (((sc[scBase + j + 4] & 0xFF) >> 4) | ((sc[scBase + j] & 0xC0) >> 2)) & 0x3F;
	}

	static short readLE16(byte[] raw, int off) {
		return (short) ((raw[off] & 0xFF) | ((raw[off + 1] & 0xFF) << 8));
	}

	private static void validateRaw(byte[] raw, QuantizationLayout layout, int nelems) {
		if (raw == null) {
			throw new IllegalArgumentException(layout.name() + ": raw is null");
		}
		if (raw.length % layout.blockBytes() != 0) {
			throw new IllegalArgumentException(
					layout.name() + ": raw length " + raw.length
							+ " not divisible by blockBytes=" + layout.blockBytes());
		}
		if (nelems > 0) {
			long expected = layout.encodedBytes(nelems);
			if (raw.length != expected) {
				throw new IllegalArgumentException(
						layout.name() + ": raw length " + raw.length + " != expected " + expected);
			}
		}
	}

	private static int clamp(int v, int lo, int hi) {
		return v < lo ? lo : (v > hi ? hi : v);
	}

	private static int clamp6(int v) {
		return clamp(v, 0, 63);
	}
}
