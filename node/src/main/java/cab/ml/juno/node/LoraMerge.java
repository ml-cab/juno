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

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Merges a trained LoRA adapter set into a GGUF model file, producing a new
 * standalone GGUF that no longer requires the {@code .lora} file at inference
 * time.
 *
 * <h3>Algorithm</h3>
 * For each frozen weight matrix {@code W} that has a LoRA adapter:
 * <pre>
 *   W_merged = W + (alpha / rank) × B × A
 * </pre>
 * The merged weights are re-quantised back to the tensor's original format
 * (Q4_K, Q6_K, Q8_0, F16, …) so the output GGUF is byte-for-byte compatible
 * with the original — same file size, same tensor layout, only the data bytes
 * of the adapted projection weights differ.
 *
 * <h3>Supported quantisation types</h3>
 * F32, F16, BF16, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, Q2_K, Q3_K.
 * All types that {@link GgufReader} can read are also re-quantisable here.
 *
 * <h3>Projection name mapping (LoRA key → GGUF tensor name)</h3>
 * <pre>
 *   "L:wq" → blk.L.attn_q.weight
 *   "L:wk" → blk.L.attn_k.weight
 *   "L:wv" → blk.L.attn_v.weight
 *   "L:wo" → blk.L.attn_output.weight
 * </pre>
 *
 * <h3>Strategy</h3>
 * The source file (GGUF or llamafile) is copied verbatim to {@code outputPath}.
 * Only the raw bytes for the adapted tensors are overwritten in-place; all
 * metadata, tokeniser data, and non-adapted tensors are untouched.
 *
 * <h3>Usage</h3>
 * <pre>
 *   LoraMerge.Result r = LoraMerge.merge(
 *       Path.of("model.gguf"),
 *       Path.of("model.lora"),
 *       Path.of("model-merged.gguf"));
 *   System.out.println("Merged " + r.adaptersApplied() + " adapters");
 * </pre>
 */
public final class LoraMerge {

	// ── GGML type constants (mirrors GgufReader) ──────────────────────────────
	private static final int TYPE_F32  =  0;
	private static final int TYPE_F16  =  1;
	private static final int TYPE_Q4_0 =  2;
	private static final int TYPE_Q8_0 =  8;
	private static final int TYPE_Q2_K = 10;
	private static final int TYPE_Q3_K = 11;
	private static final int TYPE_Q4_K = 12;
	private static final int TYPE_Q5_K = 13;
	private static final int TYPE_Q6_K = 14;
	private static final int TYPE_BF16 = 30;

	// ── LoRA projection key → GGUF tensor name suffix ────────────────────────
	private static final Map<String, String> PROJ_SUFFIX = Map.of(
			"wq", "attn_q.weight",
			"wk", "attn_k.weight",
			"wv", "attn_v.weight",
			"wo", "attn_output.weight"
	);

	private LoraMerge() {}

	// ── Public result type ────────────────────────────────────────────────────

	/**
	 * Summary of a completed merge operation.
	 *
	 * @param adaptersApplied number of LoRA adapters successfully baked in
	 * @param tensorsPatched  GGUF tensor names that were re-quantised and patched
	 * @param skipped         adapter keys skipped (tensor absent in model)
	 */
	public record Result(int adaptersApplied, List<String> tensorsPatched, List<String> skipped) {}

	// ── Public API ────────────────────────────────────────────────────────────

	/**
	 * Merge {@code loraPath} into {@code modelPath} and write the result to
	 * {@code outputPath}.
	 *
	 * <p>The output file is a valid GGUF (or llamafile with the merged GGUF
	 * embedded) that stands alone — no {@code .lora} sidecar needed.
	 *
	 * @param modelPath  source GGUF or llamafile (read-only, never modified)
	 * @param loraPath   trained {@code .lora} checkpoint produced by Juno
	 * @param outputPath destination; overwritten if it exists
	 * @return summary of what was merged
	 * @throws IOException on any I/O failure or unsupported tensor type
	 */
	public static Result merge(Path modelPath, Path loraPath, Path outputPath) throws IOException {
		LoraAdapterSet adapters = LoraAdapterSet.load(loraPath);

		// Step 1 — copy the source file to the output destination.
		// We will patch tensor data in-place; all other bytes remain identical.
		Files.copy(modelPath, outputPath, StandardCopyOption.REPLACE_EXISTING);

		List<String> patched = new ArrayList<>();
		List<String> skipped = new ArrayList<>();

		try (GgufReader reader = GgufReader.open(modelPath);
			 FileChannel outCh = FileChannel.open(outputPath,
					 StandardOpenOption.READ, StandardOpenOption.WRITE)) {

			for (Map.Entry<String, LoraAdapter> entry : adapters.asMap().entrySet()) {
				String key  = entry.getKey();
				LoraAdapter lora = entry.getValue();

				// Resolve "layer:proj" → GGUF tensor name
				int    layer = LoraAdapterSet.keyLayer(key);
				String proj  = LoraAdapterSet.keyProj(key);
				String suffix = PROJ_SUFFIX.get(proj);
				if (suffix == null) {
					skipped.add(key + " (unknown projection)");
					continue;
				}
				String tensorName = "blk." + layer + "." + suffix;
				if (!reader.hasTensor(tensorName)) {
					skipped.add(key + " → " + tensorName + " (not in model)");
					continue;
				}

				// Step 2 — dequantise the frozen weight matrix to float32.
				float[] w = reader.tensor(tensorName);

				// GGUF stores weight dims as [inDim, outDim] (inner-first).
				long[] dims = reader.tensorDims(tensorName);
				int inDim  = (int) dims[0];
				int outDim = (int) dims[1];

				// Step 3 — apply LoRA delta: W += scale × B × A (in-place)
				applyDelta(w, lora, outDim, inDim);

				// Step 4 — re-quantise to the original format.
				int   type = reader.tensorType(tensorName);
				byte[] raw = requantize(w, type, (int) lora.inDim, (int) lora.outDim);

				// Step 5 — overwrite the tensor bytes in the output file.
				long absOffset = reader.tensorAbsoluteOffset(tensorName);
				ByteBuffer buf = ByteBuffer.wrap(raw);
				while (buf.hasRemaining())
					outCh.write(buf, absOffset + (raw.length - buf.remaining()));

				patched.add(tensorName);
			}
		}

		return new Result(patched.size(), List.copyOf(patched), List.copyOf(skipped));
	}

	// ── LoRA delta application ────────────────────────────────────────────────

	/**
	 * Apply {@code ΔW = scale × B × A} to the dequantised weight matrix {@code w}
	 * in-place.
	 *
	 * <p>W is stored row-major [outDim × inDim]. A is [rank × inDim], B is
	 * [outDim × rank]. The loop order (outer=row, mid=rank, inner=col) keeps
	 * memory access patterns cache-friendly.
	 */
	static void applyDelta(float[] w, LoraAdapter lora, int outDim, int inDim) {
		float[] a     = lora.a();
		float[] b     = lora.b();
		float   scale = lora.scale;
		int     rank  = lora.rank;

		for (int r = 0; r < outDim; r++) {
			int wRowBase = r * inDim;
			int bRowBase = r * rank;
			for (int k = 0; k < rank; k++) {
				float bscale = b[bRowBase + k] * scale;
				if (bscale == 0f) continue; // B starts at zero; skip untrained steps
				int aRowBase = k * inDim;
				for (int c = 0; c < inDim; c++) {
					w[wRowBase + c] += bscale * a[aRowBase + c];
				}
			}
		}
	}

	// ── Re-quantisation dispatcher ────────────────────────────────────────────

	/**
	 * Re-quantise {@code data} (float32) back to the given GGML type.
	 *
	 * @param data   dequantised + LoRA-merged weights, length {@code outDim × inDim}
	 * @param type   GGML quantisation type ID
	 * @param inDim  columns of the weight matrix
	 * @param outDim rows of the weight matrix
	 * @return raw bytes in the original quantised encoding, same length as the
	 *         original tensor's encoded form
	 */
	static byte[] requantize(float[] data, int type, int inDim, int outDim) {
		int n = data.length;
		return switch (type) {
			case TYPE_F32  -> quantizeF32(data);
			case TYPE_F16  -> quantizeF16(data);
			case TYPE_BF16 -> quantizeBF16(data);
			case TYPE_Q8_0 -> quantizeQ8_0(data, n);
			case TYPE_Q4_0 -> quantizeQ4_0(data, n);
			case TYPE_Q4_K -> quantizeQ4_K(data, n);
			case TYPE_Q5_K -> quantizeQ5_K(data, n);
			case TYPE_Q6_K -> quantizeQ6_K(data, n);
			case TYPE_Q2_K -> quantizeQ2_K(data, n);
			case TYPE_Q3_K -> quantizeQ3_K(data, n);
			default -> throw new UnsupportedOperationException(
					"Re-quantisation not implemented for GGML type " + type);
		};
	}

	// ── F32 / F16 / BF16 ─────────────────────────────────────────────────────

	private static byte[] quantizeF32(float[] data) {
		ByteBuffer buf = ByteBuffer.allocate(data.length * 4).order(ByteOrder.LITTLE_ENDIAN);
		for (float f : data) buf.putFloat(f);
		return buf.array();
	}

	private static byte[] quantizeF16(float[] data) {
		ByteBuffer buf = ByteBuffer.allocate(data.length * 2).order(ByteOrder.LITTLE_ENDIAN);
		for (float f : data) buf.putShort(f32ToF16(f));
		return buf.array();
	}

	private static byte[] quantizeBF16(float[] data) {
		ByteBuffer buf = ByteBuffer.allocate(data.length * 2).order(ByteOrder.LITTLE_ENDIAN);
		for (float f : data) buf.putShort((short) (Float.floatToRawIntBits(f) >>> 16));
		return buf.array();
	}

	// ── Q8_0 ─────────────────────────────────────────────────────────────────
	// Block of 32: [d: f16][qi: 32 × int8]  →  34 bytes/block

	private static byte[] quantizeQ8_0(float[] data, int n) {
		int nBlocks = n / 32;
		ByteBuffer buf = ByteBuffer.allocate(nBlocks * 34).order(ByteOrder.LITTLE_ENDIAN);
		for (int b = 0; b < nBlocks; b++) {
			int base = b * 32;
			float absMax = 0f;
			for (int i = 0; i < 32; i++) absMax = Math.max(absMax, Math.abs(data[base + i]));
			float d    = absMax / 127f;
			float invD = d > 0f ? 1f / d : 0f;
			buf.putShort(f32ToF16(d));
			for (int i = 0; i < 32; i++) {
				int q = Math.round(data[base + i] * invD);
				buf.put((byte) Math.max(-127, Math.min(127, q)));
			}
		}
		return buf.array();
	}

	// ── Q4_0 ─────────────────────────────────────────────────────────────────
	// Block of 32: [d: f16][qs: 16 bytes packed nibbles]  →  18 bytes/block
	// Range [0..15], bias 8 (so signed range is [-8..7]).

	private static byte[] quantizeQ4_0(float[] data, int n) {
		int nBlocks = n / 32;
		ByteBuffer buf = ByteBuffer.allocate(nBlocks * 18).order(ByteOrder.LITTLE_ENDIAN);
		for (int b = 0; b < nBlocks; b++) {
			int   base   = b * 32;
			float absMax = 0f;
			for (int i = 0; i < 32; i++) absMax = Math.max(absMax, Math.abs(data[base + i]));
			float d    = absMax / 8f;
			float invD = d > 0f ? 1f / d : 0f;
			buf.putShort(f32ToF16(d));
			byte[] qs = new byte[16];
			for (int i = 0; i < 16; i++) {
				int lo = Math.max(0, Math.min(15, Math.round(data[base +    i] * invD) + 8));
				int hi = Math.max(0, Math.min(15, Math.round(data[base + 16 + i] * invD) + 8));
				qs[i]  = (byte) (lo | (hi << 4));
			}
			buf.put(qs);
		}
		return buf.array();
	}

	// ── Q4_K ─────────────────────────────────────────────────────────────────
	// Superblock of 256: [d:f16][dmin:f16][scales:12 bytes][qs:128 bytes] = 144 bytes
	// 8 sub-blocks of 32 elements. Quant range [0..15].

	private static byte[] quantizeQ4_K(float[] data, int n) {
		int QK_K    = 256;
		int nBlocks = n / QK_K;
		ByteBuffer buf = ByteBuffer.allocate(nBlocks * 144).order(ByteOrder.LITTLE_ENDIAN);

		for (int b = 0; b < nBlocks; b++) {
			int     base   = b * QK_K;
			float[] mins   = new float[8]; // per-subblock minimum
			float[] ranges = new float[8]; // per-subblock (max − min)

			for (int s = 0; s < 8; s++) {
				float min = Float.MAX_VALUE, max = -Float.MAX_VALUE;
				for (int i = 0; i < 32; i++) {
					float v = data[base + s * 32 + i];
					if (v < min) min = v;
					if (v > max) max = v;
				}
				if (min > max) { min = 0f; max = 0f; } // degenerate block
				mins[s]   = min;
				ranges[s] = max - min;
			}

			// Superblock d and dmin: scale sub-block scales/mins to 6-bit [0..63]
			float maxRange = 0f, maxAbsMin = 0f;
			for (int s = 0; s < 8; s++) {
				if (ranges[s] > maxRange) maxRange  = ranges[s];
				float absMin = -mins[s];
				if (absMin   > maxAbsMin) maxAbsMin = absMin;
			}
			float d    = maxRange  / 63f;
			float dmin = maxAbsMin / 63f;

			// 6-bit subscale and submin per sub-block
			int[] ls = new int[8]; // subscale [0..63]
			int[] lm = new int[8]; // submin   [0..63]
			for (int s = 0; s < 8; s++) {
				ls[s] = d    > 0f ? clamp6(Math.round(ranges[s]  / d))    : 0;
				lm[s] = dmin > 0f ? clamp6(Math.round(-mins[s]   / dmin)) : 0;
			}

			buf.putShort(f32ToF16(d));
			buf.putShort(f32ToF16(dmin));

			// Pack subscales + submins into 12 bytes
			// j=0..3: sc[j] = ls[j] in bits 0..5, high 2 bits of ls[j+4] in bits 6..7
			//         sc[j+4] = lm[j] in bits 0..5, high 2 bits of lm[j+4] in bits 6..7
			//         sc[j+8] = ls[j+4] low nibble | lm[j+4] low nibble << 4
			byte[] sc = new byte[12];
			for (int j = 0; j < 4; j++) {
				sc[j]     = (byte) ((ls[j]   & 0x3F) | ((ls[j + 4] & 0x30) << 2));
				sc[j + 4] = (byte) ((lm[j]   & 0x3F) | ((lm[j + 4] & 0x30) << 2));
				sc[j + 8] = (byte) ((ls[j+4] & 0x0F) | ((lm[j + 4] & 0x0F) << 4));
			}
			buf.put(sc);

			// Pack quant values: 4 groups of 64 elements each split into two
			// sub-blocks of 32. Low nibbles = first sub-block, high = second.
			byte[] qs = new byte[128];
			for (int g = 0; g < 4; g++) {
				int s0 = g * 2, s1 = s0 + 1;
				float sc0 = d * ls[s0], mn0 = dmin * lm[s0];
				float sc1 = d * ls[s1], mn1 = dmin * lm[s1];
				int   qi  = g * 32;
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

	// ── Q5_K ─────────────────────────────────────────────────────────────────
	// Superblock of 256: [d:f16][dmin:f16][scales:12 bytes][qh:32 bytes][qs:128 bytes] = 176 bytes
	// Same subscale structure as Q4_K; 5-bit quant [0..31] split into 4+1 bits.

	private static byte[] quantizeQ5_K(float[] data, int n) {
		int QK_K    = 256;
		int nBlocks = n / QK_K;
		ByteBuffer buf = ByteBuffer.allocate(nBlocks * 176).order(ByteOrder.LITTLE_ENDIAN);

		for (int b = 0; b < nBlocks; b++) {
			int     base   = b * QK_K;
			float[] mins   = new float[8];
			float[] ranges = new float[8];

			for (int s = 0; s < 8; s++) {
				float min = Float.MAX_VALUE, max = -Float.MAX_VALUE;
				for (int i = 0; i < 32; i++) {
					float v = data[base + s * 32 + i];
					if (v < min) min = v;
					if (v > max) max = v;
				}
				if (min > max) { min = 0f; max = 0f; }
				mins[s]   = min;
				ranges[s] = max - min;
			}

			float maxRange = 0f, maxAbsMin = 0f;
			for (int s = 0; s < 8; s++) {
				if (ranges[s] > maxRange)   maxRange  = ranges[s];
				float am = -mins[s];
				if (am > maxAbsMin)          maxAbsMin = am;
			}
			float d    = maxRange  / 63f;
			float dmin = maxAbsMin / 63f;

			int[] ls = new int[8], lm = new int[8];
			for (int s = 0; s < 8; s++) {
				ls[s] = d    > 0f ? clamp6(Math.round(ranges[s]  / d))    : 0;
				lm[s] = dmin > 0f ? clamp6(Math.round(-mins[s]   / dmin)) : 0;
			}

			buf.putShort(f32ToF16(d));
			buf.putShort(f32ToF16(dmin));

			byte[] sc = new byte[12];
			for (int j = 0; j < 4; j++) {
				sc[j]     = (byte) ((ls[j]   & 0x3F) | ((ls[j + 4] & 0x30) << 2));
				sc[j + 4] = (byte) ((lm[j]   & 0x3F) | ((lm[j + 4] & 0x30) << 2));
				sc[j + 8] = (byte) ((ls[j+4] & 0x0F) | ((lm[j + 4] & 0x0F) << 4));
			}
			buf.put(sc);

			// 5-bit quants split: low 4 bits in qs (nibble), bit 4 in qh
			byte[] qh = new byte[32];
			byte[] qs = new byte[128];
			// 4 groups of 64 elements, 2 sub-blocks of 32 each
			for (int g = 0; g < 4; g++) {
				int s0 = g * 2, s1 = s0 + 1;
				float sc0 = d * ls[s0], mn0 = dmin * lm[s0];
				float sc1 = d * ls[s1], mn1 = dmin * lm[s1];
				int qi    = g * 32;
				int hiBit0 = g * 2, hiBit1 = g * 2 + 1;

				for (int l = 0; l < 32; l++) {
					// First sub-block: low nibble + hiBit0
					int q0 = sc0 > 0f ? clamp(Math.round((data[base + s0 * 32 + l] + mn0) / sc0), 0, 31) : 0;
					// Second sub-block: high nibble + hiBit1
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

	// ── Q6_K ─────────────────────────────────────────────────────────────────
	// Superblock of 256: [ql:128 bytes][qh:64 bytes][scales:16 bytes][d:f16] = 210 bytes
	// 16 sub-blocks of 16 elements. 6-bit signed quant [-32..31].
	// int8 per-subblock scale × global f16 d.

	private static byte[] quantizeQ6_K(float[] data, int n) {
		int QK_K    = 256;
		int nBlocks = n / QK_K;
		ByteBuffer buf = ByteBuffer.allocate(nBlocks * 210).order(ByteOrder.LITTLE_ENDIAN);

		for (int b = 0; b < nBlocks; b++) {
			int base = b * QK_K;

			// 16 sub-blocks of 16 elements; find per-subblock max |x|
			float[] subMax = new float[16];
			for (int s = 0; s < 16; s++) {
				float m = 0f;
				for (int i = 0; i < 16; i++) m = Math.max(m, Math.abs(data[base + s * 16 + i]));
				subMax[s] = m;
			}

			// Global scale: d * 127 * 31 = global_max ensures all sub-blocks fit
			float globalMax = 0f;
			for (float m : subMax) if (m > globalMax) globalMax = m;
			float d    = globalMax / (127f * 31f);
			float invD = d > 0f ? 1f / d : 0f;

			// Per-subblock int8 scales: sc[s] = round(subMax[s] / (d * 31))
			byte[] sc = new byte[16];
			for (int s = 0; s < 16; s++) {
				int sv = d > 0f ? clamp(Math.round(subMax[s] * invD / 31f), -127, 127) : 0;
				sc[s]  = (byte) sv;
			}

			// Quantise and pack into ql[128] + qh[64]
			// Element at block-relative position p (0..255):
			//   half = p/128, lp = p%128, quad = lp/32, l = lp%32, is = l/16
			//   sub-block index: half*8 + is + quad*2
			//   ql: half*64 + l + (quad is odd ? 32 : 0), nibble = low if quad<2, high if quad>=2
			//   qh: half*32 + l, bit pair = quad*2
			byte[] ql = new byte[128];
			byte[] qh = new byte[64];
			for (int p = 0; p < 256; p++) {
				int half = p / 128;
				int lp   = p % 128;
				int quad = lp / 32;
				int l    = lp % 32;
				int is   = l / 16;
				int sub  = half * 8 + is + quad * 2;

				int scVal = sc[sub] & 0xFF; // treat as unsigned to get |scale|
				// For signed int8 scale, use the signed value directly
				int scSigned = sc[sub]; // could be negative if rounding went wrong
				float effScale = d * scSigned;
				int q6 = effScale != 0f
						? clamp(Math.round(data[base + p] / effScale), -32, 31)
						: 0;
				int unsigned = q6 + 32; // [0..63]

				// Pack into ql and qh
				int qlBase  = half * 64;
				int qhBase  = half * 32;
				int qlIdx   = qlBase + l + ((quad % 2 == 1) ? 32 : 0);
				boolean hiNibble = (quad >= 2);
				if (!hiNibble) {
					ql[qlIdx] = (byte) ((ql[qlIdx] & 0xF0) | (unsigned & 0x0F));
				} else {
					ql[qlIdx] = (byte) ((ql[qlIdx] & 0x0F) | ((unsigned & 0x0F) << 4));
				}
				int shift = quad * 2; // 0,2,4,6 for quad 0..3
				qh[qhBase + l] = (byte) ((qh[qhBase + l] & ~(0x3 << shift))
						| (((unsigned >> 4) & 0x3) << shift));
			}

			buf.put(ql);
			buf.put(qh);
			buf.put(sc);
			buf.putShort(f32ToF16(d));
		}
		return buf.array();
	}

	// ── Q2_K ─────────────────────────────────────────────────────────────────
	// Superblock of 256: [scales:16 bytes][qs:64 bytes][d:f16][dmin:f16] = 84 bytes
	// 16 sub-blocks of 16 elements. 4-bit subscale (low nibble) + 4-bit submin (high nibble).
	// 2-bit quant [0..3].

	private static byte[] quantizeQ2_K(float[] data, int n) {
		int QK_K    = 256;
		int nBlocks = n / QK_K;
		ByteBuffer buf = ByteBuffer.allocate(nBlocks * 84).order(ByteOrder.LITTLE_ENDIAN);

		for (int b = 0; b < nBlocks; b++) {
			int base = b * QK_K;

			// 16 sub-blocks of 16 elements each, but treated as 2 halves of 8
			float[] subMins   = new float[16];
			float[] subRanges = new float[16];
			for (int s = 0; s < 16; s++) {
				float min = Float.MAX_VALUE, max = -Float.MAX_VALUE;
				for (int i = 0; i < 16; i++) {
					float v = data[base + s * 16 + i];
					if (v < min) min = v;
					if (v > max) max = v;
				}
				if (min > max) { min = 0f; max = 0f; }
				subMins[s]   = min;
				subRanges[s] = max - min;
			}

			float maxRange = 0f, maxAbsMin = 0f;
			for (int s = 0; s < 16; s++) {
				if (subRanges[s] > maxRange) maxRange   = subRanges[s];
				float am = -subMins[s];
				if (am > maxAbsMin)          maxAbsMin  = am;
			}
			// d * 3 * 15 = maxRange  (q2 max = 3, subscale max = 15)
			float d    = maxRange  / (3f * 15f);
			float dmin = maxAbsMin / 15f;

			// Per-subblock 4-bit subscale and submin
			// layout in 16 sc bytes: sc[s] low nibble = subscale, high nibble = submin
			// But Q2_K's actual layout in the file groups them across 2 halves of 8:
			// half 0 (sub-blocks 0..7): sc[0..7] stored sequentially
			// half 1 (sub-blocks 8..15): sc[8..15]
			// Each sc byte: bits 0..3 = subscale[s], bits 4..7 = submin[s]
			byte[] sc = new byte[16];
			int[]  ls  = new int[16], lm = new int[16];
			for (int s = 0; s < 16; s++) {
				ls[s] = d    > 0f ? clamp4(Math.round(subRanges[s] / (d * 3f))) : 0;
				lm[s] = dmin > 0f ? clamp4(Math.round(-subMins[s]  / dmin))     : 0;
				sc[s] = (byte) ((ls[s] & 0xF) | ((lm[s] & 0xF) << 4));
			}

			// Pack 2-bit quants into qs[64].
			// Q2_K layout (from dequant): two halves of 128 elements each.
			// Within each half, 8 sub-blocks of 16 elements.
			// Sub-block k (0..7 within half): elements at half*128 + k*16 + l (l=0..15).
			// qs encoding (from dequant loop):
			//   k=0: qs[qBase+l] bits 0..1
			//   k=1: qs[qBase+l+16] bits 0..1
			//   k=2: qs[qBase+l] bits 2..3
			//   k=3: qs[qBase+l+16] bits 2..3
			//   k=4: qs[qBase+l] bits 4..5
			//   k=5: qs[qBase+l+16] bits 4..5
			//   k=6: qs[qBase+l] bits 6..7
			//   k=7: qs[qBase+l+16] bits 6..7
			byte[] qs = new byte[64];
			for (int half = 0; half < 2; half++) {
				int qBase   = half * 32;
				int scBase  = half * 8;
				for (int k = 0; k < 8; k++) {
					int s     = scBase + k;         // global sub-block index (0..15)
					float sc_f = d * ls[s];
					float mn_f = dmin * lm[s];
					int   qs_off = (k % 2 == 0) ? 0 : 16; // even k → qs[qBase+l], odd → qs[qBase+l+16]
					int   shift  = (k / 2) * 2;

					for (int l = 0; l < 16; l++) {
						float x  = data[base + half * 128 + k * 16 + l];
						int   q2 = sc_f > 0f ? clamp(Math.round((x + mn_f) / sc_f), 0, 3) : 0;
						int   qi = qBase + l + qs_off;
						qs[qi]   = (byte) ((qs[qi] & ~(0x3 << shift)) | (q2 << shift));
					}
				}
			}

			buf.put(sc);
			buf.put(qs);
			buf.putShort(f32ToF16(d));
			buf.putShort(f32ToF16(dmin));
		}
		return buf.array();
	}

	// ── Q3_K ─────────────────────────────────────────────────────────────────
	// Superblock of 256: [hmask:32 bytes][qs:64 bytes][scales:12 bytes][d:f16] = 110 bytes
	// 16 sub-blocks of 16 elements. 3-bit signed quant [-4..3].
	// 6-bit signed scale per sub-block (biased +32 in storage, so stored as [0..63]).

	private static byte[] quantizeQ3_K(float[] data, int n) {
		int QK_K    = 256;
		int nBlocks = n / QK_K;
		ByteBuffer buf = ByteBuffer.allocate(nBlocks * 110).order(ByteOrder.LITTLE_ENDIAN);

		for (int b = 0; b < nBlocks; b++) {
			int base = b * QK_K;

			// 16 sub-blocks of 16 elements
			float[] subMax = new float[16];
			int[]   subSc  = new int[16];  // signed [-32..31] stored as [0..63] (+32 bias)
			for (int s = 0; s < 16; s++) {
				float m = 0f;
				for (int i = 0; i < 16; i++) m = Math.max(m, Math.abs(data[base + s * 16 + i]));
				subMax[s] = m;
			}

			float globalMax = 0f;
			for (float m : subMax) if (m > globalMax) globalMax = m;
			// d * maxScale * 4 = globalMax  (q3 range [-4..3], max abs = 4)
			// maxScale ≤ 31 (6-bit signed: -32..31)
			float d    = globalMax / (4f * 31f);
			float invD = d > 0f ? 1f / d : 0f;

			for (int s = 0; s < 16; s++) {
				int sv = d > 0f ? clamp(Math.round(subMax[s] * invD / 4f), -31, 31) : 0;
				subSc[s] = sv; // signed [-31..31]
			}

			// Pack 16 × 6-bit scales into 12 bytes using the same
			// aux0/aux1/aux2 → utmp packing that the dequant reverses.
			// The stored value is subSc[s] + 32 (bias), so [0..63] fits in 6 bits.
			int[] stored = new int[16];
			for (int s = 0; s < 16; s++) stored[s] = subSc[s] + 32; // [0..63]

			// Reverse of the utmp unpack in the dequant:
			// utmp[0] = aux0 & 0x0f0f0f0f | (aux2       & 0x03030303) << 4  → sc[0,4,8,12]
			// utmp[1] = aux1 & 0x0f0f0f0f | (aux2>>2    & 0x03030303) << 4  → sc[1,5,9,13]
			// utmp[2] = (aux0>>4) & 0x0f0f0f0f | (aux2>>4 & 0x03030303) << 4 → sc[2,6,10,14]
			// utmp[3] = (aux1>>4) & 0x0f0f0f0f | (aux2>>6 & 0x03030303) << 4 → sc[3,7,11,15]
			// sc[k*4+r] = (utmp[k] >> (r*8)) & 0xFF (after subtracting 32)
			// So stored[k*4+r] = (utmp[k] >> (r*8)) & 0xFF
			// utmp[k] = stored[k*4+0] | stored[k*4+1]<<8 | stored[k*4+2]<<16 | stored[k*4+3]<<24
			int[] utmp = new int[4];
			for (int k = 0; k < 4; k++) {
				utmp[k] = (stored[k*4]     & 0xFF)
						| ((stored[k*4+1]  & 0xFF) <<  8)
						| ((stored[k*4+2]  & 0xFF) << 16)
						| ((stored[k*4+3]  & 0xFF) << 24);
			}
			// Now reverse: extract aux0, aux1, aux2 from utmp.
			// utmp[k] low nibble per byte → aux[k] low nibble per byte
			// utmp[k] high 2 bits of low byte (bits 4..5) → aux2 2-bit chunk
			// aux0 = utmp[0] & 0x0f0f0f0f  (low nibbles)
			// aux1 = utmp[1] & 0x0f0f0f0f
			// aux2 high bits reconstructed from utmp[][5:4] for utmp[0..3]
			int aux0 = utmp[0] & 0x0f0f0f0f;
			int aux1 = utmp[1] & 0x0f0f0f0f;
			// aux2: bits 0..1 from utmp[0] bits 4..5, bits 2..3 from utmp[1], etc.
			// Per dequant: utmp[0] = aux0 & 0x0f | (aux2 & 0x03)<<4 per byte
			// → (aux2 & 0x03) per byte = (utmp[0] >> 4) & 0x03
			// aux2 byte 0: bits 0..1 = (utmp[0]>>4)&3, bits 2..3=(utmp[1]>>4)&3,
			//              bits 4..5 = (utmp[2]>>4)&3, bits 6..7=(utmp[3]>>4)&3
			int aux2 = 0;
			for (int byteIdx = 0; byteIdx < 4; byteIdx++) {
				int shift = byteIdx * 8;
				int b0 = (utmp[0] >> (shift + 4)) & 0x3;
				int b1 = (utmp[1] >> (shift + 4)) & 0x3;
				int b2 = (utmp[2] >> (shift + 4)) & 0x3;
				int b3 = (utmp[3] >> (shift + 4)) & 0x3;
				aux2 |= (b0 | (b1 << 2) | (b2 << 4) | (b3 << 6)) << shift;
				aux0 |= ((utmp[0] >> (shift + 4)) & 0x3) << (shift + 4);
				aux1 |= ((utmp[1] >> (shift + 4)) & 0x3) << (shift + 4);
			}
			byte[] scRaw = new byte[12];
			scRaw[ 0] = (byte)  aux0;
			scRaw[ 1] = (byte) (aux0 >>> 8);
			scRaw[ 2] = (byte) (aux0 >>> 16);
			scRaw[ 3] = (byte) (aux0 >>> 24);
			scRaw[ 4] = (byte)  aux1;
			scRaw[ 5] = (byte) (aux1 >>> 8);
			scRaw[ 6] = (byte) (aux1 >>> 16);
			scRaw[ 7] = (byte) (aux1 >>> 24);
			scRaw[ 8] = (byte)  aux2;
			scRaw[ 9] = (byte) (aux2 >>> 8);
			scRaw[10] = (byte) (aux2 >>> 16);
			scRaw[11] = (byte) (aux2 >>> 24);

			// Pack 3-bit quants into hmask[32] + qs[64].
			// Q3_K layout (mirroring dequant):
			// 2 halves of 128 elements. Within each half, 4 shift iterations (0,2,4,6),
			// each producing 32 elements split as 2×16 (l=0..15 in two qs rows).
			// hmask bitmask m advances 1→2→4→8→16→32→64→128 across both halves.
			//   q3 = (low2 from qs) | (hmask_bit<<2)) - 4   range [-4..3]
			byte[] hmask = new byte[32];
			byte[] qs    = new byte[64];
			int m = 1;
			int is_idx = 0; // sub-block index across the loops
			for (int half = 0; half < 2; half++) {
				int qBase  = half * 32;
				int scBase = half * 8;
				int shift  = 0;
				for (int j = 0; j < 4; j++) {
					// First 16 elements of this (half, j) group
					int s0 = is_idx++;
					float dl0 = d * subSc[s0];
					for (int l = 0; l < 16; l++) {
						float x   = data[base + half * 128 + j * 32 + l];
						int   q3  = dl0 != 0f ? clamp(Math.round(x / dl0), -4, 3) : 0;
						// unsigned q3 in [0..7]: high bit → hmask, low 2 bits → qs
						int uq3   = q3 + 4;
						int low2  = uq3 & 0x3;
						int hi1   = (uq3 >> 2) & 1;
						qs[qBase + l] = (byte) ((qs[qBase + l] & ~(0x3 << shift)) | (low2 << shift));
						if (hi1 != 0) hmask[l] |= (byte) m;
					}
					// Second 16 elements
					int s1 = is_idx++;
					float dl1 = d * subSc[s1];
					for (int l = 0; l < 16; l++) {
						float x   = data[base + half * 128 + j * 32 + 16 + l];
						int   q3  = dl1 != 0f ? clamp(Math.round(x / dl1), -4, 3) : 0;
						int uq3   = q3 + 4;
						int low2  = uq3 & 0x3;
						int hi1   = (uq3 >> 2) & 1;
						qs[qBase + l + 16] = (byte) ((qs[qBase + l + 16] & ~(0x3 << shift)) | (low2 << shift));
						if (hi1 != 0) hmask[l + 16] |= (byte) m;
					}
					shift += 2;
					m <<= 1;
				}
			}

			buf.put(hmask);
			buf.put(qs);
			buf.put(scRaw);
			buf.putShort(f32ToF16(d));
		}
		return buf.array();
	}

	// ── Float / half conversion helpers ──────────────────────────────────────

	/**
	 * IEEE-754 float32 → float16 conversion (round-to-nearest).
	 * Handles subnormals, infinities, and NaN.
	 */
	static short f32ToF16(float value) {
		int bits = Float.floatToRawIntBits(value);
		int sign = (bits >>> 31) & 1;
		int exp  = ((bits >>> 23) & 0xFF) - 127 + 15; // rebias to f16
		int mant = bits & 0x7FFFFF;

		if (exp <= 0) {
			if (exp < -10) return (short) (sign << 15); // underflow → ±0
			// Subnormal: set implicit leading bit and shift
			mant = (mant | 0x800000) >> (1 - exp);
			// Round-to-nearest-even on the low 13 bits
			mant = (mant + 0x0FFF + ((mant >> 13) & 1)) >> 13;
			return (short) ((sign << 15) | mant);
		}
		if (exp >= 31) return (short) ((sign << 15) | 0x7C00); // overflow → ±inf

		// Round-to-nearest-even
		int round = mant & 0x1FFF;
		mant >>= 13;
		if (round > 0x1000 || (round == 0x1000 && (mant & 1) == 1)) {
			mant++;
			if (mant == 0x400) { mant = 0; exp++; }
		}
		if (exp >= 31) return (short) ((sign << 15) | 0x7C00);
		return (short) ((sign << 15) | (exp << 10) | mant);
	}

	// ── Utility helpers ───────────────────────────────────────────────────────

	private static int clamp(int v, int lo, int hi) {
		return v < lo ? lo : (v > hi ? hi : v);
	}

	private static int clamp4(int v) { return clamp(v, 0, 15); }
	private static int clamp6(int v) { return clamp(v, 0, 63); }
}