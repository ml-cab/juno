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
import java.nio.file.Path;
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
 *   W_merged = W + (alpha / rank) x B x A
 * </pre>
 * The merged weights are re-quantised back to the tensor's original format
 * (Q4_K, Q6_K, Q8_0, F16, etc.) so the output GGUF is byte-for-byte compatible
 * with the original: same file size, same tensor layout, only the data bytes of
 * the adapted projection weights differ.
 *
 * <h3>Supported quantisation types</h3>
 * F32, F16, BF16, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, Q2_K, Q3_K.
 *
 * <h3>Projection name mapping (LoRA key to GGUF tensor name)</h3>
 * <pre>
 *   "L:wq" to blk.L.attn_q.weight
 *   "L:wk" to blk.L.attn_k.weight
 *   "L:wv" to blk.L.attn_v.weight
 *   "L:wo" to blk.L.attn_output.weight
 * </pre>
 *
 * <h3>Strategy</h3>
 * The source file (GGUF or llamafile) is copied verbatim to {@code outputPath}.
 * Only the raw bytes for the adapted tensors are overwritten in-place; all
 * metadata, tokeniser data, and non-adapted tensors are untouched.
 */
public final class LoraMerge {

	private static final int ALIGNMENT = 32; // GGUF data-section alignment (bytes)

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

	private static final Map<String, String> PROJ_SUFFIX = Map.of(
			"wq", "attn_q.weight",
			"wk", "attn_k.weight",
			"wv", "attn_v.weight",
			"wo", "attn_output.weight"
	);

	private LoraMerge() {}

	public record Result(int adaptersApplied, List<String> tensorsPatched, List<String> skipped) {}

	/**
	 * Merge loraPath into modelPath and write the result to outputPath.
	 *
	 * <p>Writes a new, valid GGUF file where:
	 * <ul>
	 *   <li>The LoRA-patched projection tensors (wq/wv) are stored as F32,
	 *       preserving W_merged = W + (alpha/rank) x B x A with full precision.
	 *   <li>Every other tensor keeps its original quantisation (Q4_K, Q6_K, etc.)
	 *       and its raw bytes are copied verbatim.
	 * </ul>
	 *
	 * <p>Why F32 and not re-quantise to Q4_K?  The LoRA delta is typically
	 * ~6e-4 per element, while Q4_K quantisation noise is ~3e-3: five times
	 * larger.  Re-quantising destroys the delta entirely.  F32 precision (~1e-7)
	 * preserves it with SNR ~6000x.  The merged file is larger (~1 GB for
	 * TinyLlama 1.1B vs 667 MB Q4_K original) but recalls training correctly.
	 *
	 * <p>The output is always a plain GGUF v3 file, even when the source is a
	 * llamafile ZIP polyglot.
	 */
	public static Result merge(Path modelPath, Path loraPath, Path outputPath) throws IOException {
		LoraAdapterSet adapters = LoraAdapterSet.load(loraPath);

		// Build tensor-name -> LoraAdapter lookup; collect unknown-projection skips
		Map<String, LoraAdapter> adapterByTensor = new java.util.LinkedHashMap<>();
		List<String> skipped = new ArrayList<>();
		for (Map.Entry<String, LoraAdapter> entry : adapters.asMap().entrySet()) {
			String key    = entry.getKey();
			String suffix = PROJ_SUFFIX.get(LoraAdapterSet.keyProj(key));
			if (suffix == null) { skipped.add(key + " (unknown projection)"); continue; }
			adapterByTensor.put("blk." + LoraAdapterSet.keyLayer(key) + "." + suffix, entry.getValue());
		}

		List<String> patched = new ArrayList<>();

		try (GgufReader reader = GgufReader.open(modelPath);
			 FileChannel srcCh = FileChannel.open(modelPath, StandardOpenOption.READ);
			 FileChannel outCh = FileChannel.open(outputPath,
					 StandardOpenOption.WRITE, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {

			// Remove adapters whose tensor does not exist in this model
			for (String tName : new ArrayList<>(adapterByTensor.keySet()))
				if (!reader.hasTensor(tName)) {
					skipped.add(tName + " (tensor not in model)");
					adapterByTensor.remove(tName);
				}

			List<String> tensorOrder = reader.tensorOrder();

			// ── 1. Compute new data-section offsets ───────────────────────────────
			// Patched tensors: F32 (4 bytes/element).  Others: original raw size.
			long[] newDataOffsets = new long[tensorOrder.size()];
			long   cursor         = 0L;
			for (int i = 0; i < tensorOrder.size(); i++) {
				String name = tensorOrder.get(i);
				newDataOffsets[i] = cursor;
				cursor += adapterByTensor.containsKey(name)
						? reader.tensorNelems(name) * 4L
						: GgufReader.rawByteCount(reader.tensorType(name), reader.tensorNelems(name));
			}

			// ── 2. Copy header + KV section verbatim ─────────────────────────────
			// [ggufFileOffset, metadataSectionEnd) = magic + version + counts + KV pairs
			long headerStart = reader.ggufFileOffset();
			long headerEnd   = reader.metadataSectionEnd();
			srcCh.transferTo(headerStart, headerEnd - headerStart, outCh);

			// ── 3. Write new tensor-info section ─────────────────────────────────
			for (int i = 0; i < tensorOrder.size(); i++) {
				String name = tensorOrder.get(i);
				int    type = adapterByTensor.containsKey(name) ? TYPE_F32 : reader.tensorType(name);
				writeTensorInfoEntry(outCh, name, reader.tensorDims(name), type, newDataOffsets[i]);
			}

			// ── 4. Alignment padding ──────────────────────────────────────────────
			// Output is always a plain GGUF starting at file position 0, so we
			// align the current output position to ALIGNMENT (32) bytes.
			long pos     = outCh.position();
			long aligned = ((pos + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
			if (aligned > pos)
				outCh.write(ByteBuffer.allocate((int) (aligned - pos)));

			// ── 5. Data section ───────────────────────────────────────────────────
			for (String name : tensorOrder) {
				if (adapterByTensor.containsKey(name)) {
					// Dequantise, apply LoRA delta, store as F32 (full precision)
					float[] w    = reader.tensor(name);
					long[]  dims = reader.tensorDims(name);
					applyDelta(w, adapterByTensor.get(name), (int) dims[1], (int) dims[0]);
					ByteBuffer f32 = ByteBuffer.allocate(w.length * 4).order(ByteOrder.LITTLE_ENDIAN);
					for (float f : w) f32.putFloat(f);
					f32.flip();
					outCh.write(f32);
					patched.add(name);
				} else {
					// Copy original quantised bytes verbatim - no precision loss
					outCh.write(ByteBuffer.wrap(reader.tensorRaw(name).data()));
				}
			}
		}
		return new Result(patched.size(), List.copyOf(patched), List.copyOf(skipped));
	}

	/** Write one tensor-info entry in GGUF little-endian binary format. */
	private static void writeTensorInfoEntry(FileChannel ch, String name, long[] dims, int type, long dataOffset)
			throws IOException {
		byte[] nb = name.getBytes(java.nio.charset.StandardCharsets.UTF_8);
		ByteBuffer buf = ByteBuffer.allocate(8 + nb.length + 4 + dims.length * 8 + 4 + 8)
				.order(ByteOrder.LITTLE_ENDIAN);
		buf.putLong(nb.length);
		buf.put(nb);
		buf.putInt(dims.length);
		for (long d : dims) buf.putLong(d);
		buf.putInt(type);
		buf.putLong(dataOffset);
		buf.flip();
		ch.write(buf);
	}

	// ── LoRA delta ────────────────────────────────────────────────────────────

	static void applyDelta(float[] w, LoraAdapter lora, int outDim, int inDim) {
		float[] a = lora.a(), b = lora.b();
		float scale = lora.scale;
		int rank = lora.rank;
		for (int r = 0; r < outDim; r++) {
			int wBase = r * inDim, bBase = r * rank;
			for (int k = 0; k < rank; k++) {
				float bs = b[bBase + k] * scale;
				if (bs == 0f) continue;
				int aBase = k * inDim;
				for (int c = 0; c < inDim; c++) w[wBase + c] += bs * a[aBase + c];
			}
		}
	}

	// ── Dispatcher ────────────────────────────────────────────────────────────

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
			default -> throw new UnsupportedOperationException("Re-quantisation not implemented for GGML type " + type);
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

	// ── Q8_0: block=32, [d:f16][32×int8] = 34 bytes ──────────────────────────

	private static byte[] quantizeQ8_0(float[] data, int n) {
		int nBlocks = n / 32;
		ByteBuffer buf = ByteBuffer.allocate(nBlocks * 34).order(ByteOrder.LITTLE_ENDIAN);
		for (int b = 0; b < nBlocks; b++) {
			int base = b * 32;
			float absMax = 0f;
			for (int i = 0; i < 32; i++) absMax = Math.max(absMax, Math.abs(data[base + i]));
			float d = absMax / 127f, invD = d > 0f ? 1f / d : 0f;
			buf.putShort(f32ToF16(d));
			for (int i = 0; i < 32; i++) buf.put((byte) clamp(Math.round(data[base + i] * invD), -127, 127));
		}
		return buf.array();
	}

	// ── Q4_0: block=32, [d:f16][16 packed nibbles] = 18 bytes ───────────────

	private static byte[] quantizeQ4_0(float[] data, int n) {
		int nBlocks = n / 32;
		ByteBuffer buf = ByteBuffer.allocate(nBlocks * 18).order(ByteOrder.LITTLE_ENDIAN);
		for (int b = 0; b < nBlocks; b++) {
			int base = b * 32;
			float absMax = 0f;
			for (int i = 0; i < 32; i++) absMax = Math.max(absMax, Math.abs(data[base + i]));
			float d = absMax / 8f, invD = d > 0f ? 1f / d : 0f;
			buf.putShort(f32ToF16(d));
			byte[] qs = new byte[16];
			for (int i = 0; i < 16; i++) {
				int lo = clamp(Math.round(data[base + i]      * invD) + 8, 0, 15);
				int hi = clamp(Math.round(data[base + 16 + i] * invD) + 8, 0, 15);
				qs[i]  = (byte) (lo | (hi << 4));
			}
			buf.put(qs);
		}
		return buf.array();
	}

	// ── Q4_K: superblock=256, [d:f16][dmin:f16][scales:12][qs:128] = 144 bytes
	//
	// FIX: d = maxRange / (63 * 15)  — NOT maxRange/63
	//
	// The 6-bit subscale ls[s] encodes the sub-block range relative to d:
	//   d * ls[s] * 15 ≈ ranges[s]     (effective range for 4-bit quant)
	//   ls[s] = round(63 * ranges[s] / maxRange)  in [0..63]
	// Storing d = maxRange/63 (the previous bug) made d*ls[s] ≈ ranges[s],
	// so (x - min) / (d*ls[s]) ∈ [0..1] and all quants collapsed to {0,1}.

	private static byte[] quantizeQ4_K(float[] data, int n) {
		int QK_K = 256, nBlocks = n / QK_K;
		ByteBuffer buf = ByteBuffer.allocate(nBlocks * 144).order(ByteOrder.LITTLE_ENDIAN);

		for (int b = 0; b < nBlocks; b++) {
			int base = b * QK_K;
			float[] mins = new float[8], ranges = new float[8];
			for (int s = 0; s < 8; s++) {
				float mn = Float.MAX_VALUE, mx = -Float.MAX_VALUE;
				for (int i = 0; i < 32; i++) { float v = data[base + s*32 + i]; if (v < mn) mn = v; if (v > mx) mx = v; }
				if (mn > mx) { mn = 0f; mx = 0f; }
				mins[s] = mn; ranges[s] = mx - mn;
			}
			float maxRange = 0f, maxAbsMin = 0f;
			for (int s = 0; s < 8; s++) { if (ranges[s] > maxRange) maxRange = ranges[s]; float am = -mins[s]; if (am > maxAbsMin) maxAbsMin = am; }

			// CORRECT: d * ls_max * 15 = maxRange  =>  d = maxRange / (63 * 15)
			float d    = maxRange  > 0f ? maxRange  / (63f * 15f) : 0f;
			float dmin = maxAbsMin > 0f ? maxAbsMin / 63f          : 0f;

			int[] ls = new int[8], lm = new int[8];
			for (int s = 0; s < 8; s++) {
				ls[s] = maxRange  > 0f ? clamp6(Math.round(ranges[s] * 63f / maxRange))  : 0;
				lm[s] = maxAbsMin > 0f ? clamp6(Math.round(-mins[s]  * 63f / maxAbsMin)) : 0;
			}

			buf.putShort(f32ToF16(d));
			buf.putShort(f32ToF16(dmin));

			// 12-byte subscale/submin pack (getScale4K / getMin4K round-trip)
			byte[] sc = new byte[12];
			for (int j = 0; j < 4; j++) {
				sc[j]     = (byte) ((ls[j]     & 0x3F) | ((ls[j+4] & 0x30) << 2));
				sc[j + 4] = (byte) ((lm[j]     & 0x3F) | ((lm[j+4] & 0x30) << 2));
				sc[j + 8] = (byte) ((ls[j+4]   & 0x0F) | ((lm[j+4] & 0x0F) << 4));
			}
			buf.put(sc);

			// 4-bit nibbles: 4 groups, each = 2 sub-blocks of 32 elements
			// low nibble = first sub-block (s0), high nibble = second (s1)
			byte[] qs = new byte[128];
			for (int g = 0; g < 4; g++) {
				int s0 = g*2, s1 = s0+1;
				float sc0 = d * ls[s0], mn0 = dmin * lm[s0];
				float sc1 = d * ls[s1], mn1 = dmin * lm[s1];
				int qi = g * 32;
				for (int i = 0; i < 32; i++) {
					int q0 = sc0 > 0f ? clamp(Math.round((data[base + s0*32 + i] + mn0) / sc0), 0, 15) : 0;
					int q1 = sc1 > 0f ? clamp(Math.round((data[base + s1*32 + i] + mn1) / sc1), 0, 15) : 0;
					qs[qi + i] = (byte) (q0 | (q1 << 4));
				}
			}
			buf.put(qs);
		}
		return buf.array();
	}

	// ── Q5_K: superblock=256, [d:f16][dmin:f16][scales:12][qh:32][qs:128] = 176 bytes
	//
	// FIX: d = maxRange / (63 * 31)  — NOT maxRange/63
	//
	// Same subscale structure as Q4_K; quant is 5-bit [0..31].
	//   d * ls[s] * 31 ≈ ranges[s]
	// 5-bit q split: low 4 bits into qs nibble, bit 4 into qh.

	private static byte[] quantizeQ5_K(float[] data, int n) {
		int QK_K = 256, nBlocks = n / QK_K;
		ByteBuffer buf = ByteBuffer.allocate(nBlocks * 176).order(ByteOrder.LITTLE_ENDIAN);

		for (int b = 0; b < nBlocks; b++) {
			int base = b * QK_K;
			float[] mins = new float[8], ranges = new float[8];
			for (int s = 0; s < 8; s++) {
				float mn = Float.MAX_VALUE, mx = -Float.MAX_VALUE;
				for (int i = 0; i < 32; i++) { float v = data[base + s*32 + i]; if (v < mn) mn = v; if (v > mx) mx = v; }
				if (mn > mx) { mn = 0f; mx = 0f; }
				mins[s] = mn; ranges[s] = mx - mn;
			}
			float maxRange = 0f, maxAbsMin = 0f;
			for (int s = 0; s < 8; s++) { if (ranges[s] > maxRange) maxRange = ranges[s]; float am = -mins[s]; if (am > maxAbsMin) maxAbsMin = am; }

			// CORRECT: d * ls_max * 31 = maxRange  =>  d = maxRange / (63 * 31)
			float d    = maxRange  > 0f ? maxRange  / (63f * 31f) : 0f;
			float dmin = maxAbsMin > 0f ? maxAbsMin / 63f          : 0f;

			int[] ls = new int[8], lm = new int[8];
			for (int s = 0; s < 8; s++) {
				ls[s] = maxRange  > 0f ? clamp6(Math.round(ranges[s] * 63f / maxRange))  : 0;
				lm[s] = maxAbsMin > 0f ? clamp6(Math.round(-mins[s]  * 63f / maxAbsMin)) : 0;
			}

			buf.putShort(f32ToF16(d));
			buf.putShort(f32ToF16(dmin));

			byte[] sc = new byte[12];
			for (int j = 0; j < 4; j++) {
				sc[j]     = (byte) ((ls[j]   & 0x3F) | ((ls[j+4] & 0x30) << 2));
				sc[j + 4] = (byte) ((lm[j]   & 0x3F) | ((lm[j+4] & 0x30) << 2));
				sc[j + 8] = (byte) ((ls[j+4] & 0x0F) | ((lm[j+4] & 0x0F) << 4));
			}
			buf.put(sc);

			byte[] qh = new byte[32], qs = new byte[128];
			for (int g = 0; g < 4; g++) {
				int s0 = g*2, s1 = s0+1;
				float sc0 = d * ls[s0], mn0 = dmin * lm[s0];
				float sc1 = d * ls[s1], mn1 = dmin * lm[s1];
				int qi = g * 32, hiBit0 = g*2, hiBit1 = g*2+1;
				for (int l = 0; l < 32; l++) {
					int q0 = sc0 > 0f ? clamp(Math.round((data[base + s0*32 + l] + mn0) / sc0), 0, 31) : 0;
					int q1 = sc1 > 0f ? clamp(Math.round((data[base + s1*32 + l] + mn1) / sc1), 0, 31) : 0;
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

	// ── Q6_K: superblock=256, [ql:128][qh:64][scales:16][d:f16] = 210 bytes
	// 16 sub-blocks of 16 elements; 6-bit signed quant [-32..31]; int8 per-subblock scale.
	//   d * sc8[s] * 32 ≈ subMax[s]   =>   d = globalMax / (127 * 32)

	private static byte[] quantizeQ6_K(float[] data, int n) {
		int QK_K = 256, nBlocks = n / QK_K;
		ByteBuffer buf = ByteBuffer.allocate(nBlocks * 210).order(ByteOrder.LITTLE_ENDIAN);
		for (int b = 0; b < nBlocks; b++) {
			int base = b * QK_K;
			float[] subMax = new float[16];
			for (int s = 0; s < 16; s++) { float m = 0f; for (int i = 0; i < 16; i++) m = Math.max(m, Math.abs(data[base + s*16 + i])); subMax[s] = m; }
			float globalMax = 0f;
			for (float m : subMax) if (m > globalMax) globalMax = m;

			float d    = globalMax > 0f ? globalMax / (127f * 32f) : 0f;
			float invD = d > 0f ? 1f / d : 0f;
			byte[] sc = new byte[16];
			for (int s = 0; s < 16; s++) sc[s] = (byte) (d > 0f ? clamp(Math.round(subMax[s] * invD / 32f), -127, 127) : 0);

			byte[] ql = new byte[128], qh = new byte[64];
			for (int p = 0; p < QK_K; p++) {
				int half = p/128, lp = p%128, quad = lp/32, l = lp%32;
				int sub = half*8 + (l/16) + quad*2;
				float effScale = d * sc[sub];
				int q6 = effScale != 0f ? clamp(Math.round(data[base + p] / effScale), -32, 31) : 0;
				int unsigned = q6 + 32;
				int qlBase = half*64, qhBase = half*32;
				int qlIdx = qlBase + l + ((quad%2 == 1) ? 32 : 0);
				if (quad < 2) ql[qlIdx] = (byte) ((ql[qlIdx] & 0xF0) | (unsigned & 0x0F));
				else          ql[qlIdx] = (byte) ((ql[qlIdx] & 0x0F) | ((unsigned & 0x0F) << 4));
				int shift = quad * 2;
				qh[qhBase + l] = (byte) ((qh[qhBase + l] & ~(0x3 << shift)) | (((unsigned >> 4) & 0x3) << shift));
			}
			buf.put(ql); buf.put(qh); buf.put(sc); buf.putShort(f32ToF16(d));
		}
		return buf.array();
	}

	// ── Q2_K: superblock=256, [scales:16][qs:64][d:f16][dmin:f16] = 84 bytes
	// 16 sub-blocks of 16 elements; 4-bit subscale [0..15]; 2-bit quant [0..3].
	//   d = maxRange / (15 * 3);  dmin = maxAbsMin / 15

	private static byte[] quantizeQ2_K(float[] data, int n) {
		int QK_K = 256, nBlocks = n / QK_K;
		ByteBuffer buf = ByteBuffer.allocate(nBlocks * 84).order(ByteOrder.LITTLE_ENDIAN);
		for (int b = 0; b < nBlocks; b++) {
			int base = b * QK_K;
			float[] subMins = new float[16], subRanges = new float[16];
			for (int s = 0; s < 16; s++) {
				float mn = Float.MAX_VALUE, mx = -Float.MAX_VALUE;
				for (int i = 0; i < 16; i++) { float v = data[base + s*16 + i]; if (v < mn) mn = v; if (v > mx) mx = v; }
				if (mn > mx) { mn = 0f; mx = 0f; }
				subMins[s] = mn; subRanges[s] = mx - mn;
			}
			float maxRange = 0f, maxAbsMin = 0f;
			for (int s = 0; s < 16; s++) { if (subRanges[s] > maxRange) maxRange = subRanges[s]; float am = -subMins[s]; if (am > maxAbsMin) maxAbsMin = am; }

			float d    = maxRange  > 0f ? maxRange  / (15f * 3f) : 0f;
			float dmin = maxAbsMin > 0f ? maxAbsMin / 15f          : 0f;

			byte[] sc = new byte[16];
			int[] ls = new int[16], lm = new int[16];
			for (int s = 0; s < 16; s++) {
				ls[s] = maxRange  > 0f ? clamp4(Math.round(subRanges[s] * 15f / maxRange))  : 0;
				lm[s] = maxAbsMin > 0f ? clamp4(Math.round(-subMins[s]  * 15f / maxAbsMin)) : 0;
				sc[s] = (byte) ((ls[s] & 0xF) | ((lm[s] & 0xF) << 4));
			}

			byte[] qs = new byte[64];
			for (int half = 0; half < 2; half++) {
				int qBase = half*32, scBase = half*8;
				for (int k = 0; k < 8; k++) {
					int s = scBase + k;
					float scF = d * ls[s], mnF = dmin * lm[s];
					int qsOff = (k%2 == 0) ? 0 : 16, shift = (k/2)*2;
					for (int l = 0; l < 16; l++) {
						float x = data[base + half*128 + k*16 + l];
						int q = scF > 0f ? clamp(Math.round((x + mnF) / scF), 0, 3) : 0;
						int qi = qBase + l + qsOff;
						qs[qi] = (byte) ((qs[qi] & ~(0x3 << shift)) | (q << shift));
					}
				}
			}
			buf.put(sc); buf.put(qs); buf.putShort(f32ToF16(d)); buf.putShort(f32ToF16(dmin));
		}
		return buf.array();
	}

	// ── Q3_K: superblock=256, [hmask:32][qs:64][scales:12][d:f16] = 110 bytes
	// 16 sub-blocks of 16 elements; 3-bit signed quant [-4..3].
	// Per-subblock 6-bit signed scale stored biased +32 in [0..63].
	//   d = globalMax / (31 * 4)
	//
	// scRaw packing (clean inverse of GgufReader.loadQ3_K utmp decode):
	//   aux0 byte r = (stored[0*4+r] & 0xF) | ((stored[2*4+r] & 0xF) << 4)
	//   aux1 byte r = (stored[1*4+r] & 0xF) | ((stored[3*4+r] & 0xF) << 4)
	//   aux2 byte r = (stored[0*4+r]>>4)&3 | ((stored[1*4+r]>>4)&3)<<2
	//               | ((stored[2*4+r]>>4)&3)<<4 | ((stored[3*4+r]>>4)&3)<<6

	private static byte[] quantizeQ3_K(float[] data, int n) {
		int QK_K = 256, nBlocks = n / QK_K;
		ByteBuffer buf = ByteBuffer.allocate(nBlocks * 110).order(ByteOrder.LITTLE_ENDIAN);
		for (int b = 0; b < nBlocks; b++) {
			int base = b * QK_K;
			float[] subMax = new float[16];
			for (int s = 0; s < 16; s++) { float m = 0f; for (int i = 0; i < 16; i++) m = Math.max(m, Math.abs(data[base + s*16 + i])); subMax[s] = m; }
			float globalMax = 0f;
			for (float m : subMax) if (m > globalMax) globalMax = m;

			float d    = globalMax > 0f ? globalMax / (31f * 4f) : 0f;
			float invD = d > 0f ? 1f / d : 0f;
			int[] subSc  = new int[16];
			int[] stored = new int[16];
			for (int s = 0; s < 16; s++) { subSc[s] = d > 0f ? clamp(Math.round(subMax[s] * invD / 4f), -31, 31) : 0; stored[s] = subSc[s] + 32; }

			// Pack 16 x 6-bit stored[] into 12 bytes via aux0/aux1/aux2
			int aux0 = 0, aux1 = 0, aux2 = 0;
			for (int r = 0; r < 4; r++) {
				int s0 = stored[r], s1 = stored[4+r], s2 = stored[8+r], s3 = stored[12+r];
				aux0 |= ((s0 & 0xF) | ((s2 & 0xF) << 4)) << (r*8);
				aux1 |= ((s1 & 0xF) | ((s3 & 0xF) << 4)) << (r*8);
				aux2 |= (((s0>>4)&3) | (((s1>>4)&3)<<2) | (((s2>>4)&3)<<4) | (((s3>>4)&3)<<6)) << (r*8);
			}
			byte[] scRaw = new byte[12];
			for (int i = 0; i < 4; i++) { scRaw[i]   = (byte)(aux0>>(i*8)); scRaw[i+4] = (byte)(aux1>>(i*8)); scRaw[i+8] = (byte)(aux2>>(i*8)); }

			// Pack 3-bit quants: 2 halves x 4 shift iters x 2 groups of 16
			byte[] hmask = new byte[32], qs = new byte[64];
			int is_idx = 0, m = 1;
			for (int half = 0; half < 2; half++) {
				int qBase = half*32, shift = 0;
				for (int j = 0; j < 4; j++) {
					float dl0 = d * subSc[is_idx++];
					for (int l = 0; l < 16; l++) {
						int q3 = dl0 != 0f ? clamp(Math.round(data[base + half*128 + j*32 + l] / dl0), -4, 3) : 0;
						int uq3 = q3 + 4;
						qs[qBase + l] = (byte) ((qs[qBase + l] & ~(0x3<<shift)) | ((uq3 & 3) << shift));
						if (((uq3>>2)&1) != 0) hmask[l] |= (byte) m;
					}
					float dl1 = d * subSc[is_idx++];
					for (int l = 0; l < 16; l++) {
						int q3 = dl1 != 0f ? clamp(Math.round(data[base + half*128 + j*32 + 16 + l] / dl1), -4, 3) : 0;
						int uq3 = q3 + 4;
						qs[qBase + l + 16] = (byte) ((qs[qBase + l + 16] & ~(0x3<<shift)) | ((uq3 & 3) << shift));
						if (((uq3>>2)&1) != 0) hmask[l+16] |= (byte) m;
					}
					shift += 2; m <<= 1;
				}
			}
			buf.put(hmask); buf.put(qs); buf.put(scRaw); buf.putShort(f32ToF16(d));
		}
		return buf.array();
	}

	// ── f32 -> f16 (round-to-nearest-even) ───────────────────────────────────

	static short f32ToF16(float value) {
		int bits = Float.floatToRawIntBits(value);
		int sign = (bits >>> 31) & 1;
		int exp  = ((bits >>> 23) & 0xFF) - 127 + 15;
		int mant = bits & 0x7FFFFF;
		if (exp <= 0) {
			if (exp < -10) return (short)(sign << 15);
			mant = (mant | 0x800000) >> (1 - exp);
			mant = (mant + 0x0FFF + ((mant >> 13) & 1)) >> 13;
			return (short)((sign << 15) | mant);
		}
		if (exp >= 31) return (short)((sign << 15) | 0x7C00);
		int round = mant & 0x1FFF; mant >>= 13;
		if (round > 0x1000 || (round == 0x1000 && (mant & 1) == 1)) { mant++; if (mant == 0x400) { mant = 0; exp++; } }
		if (exp >= 31) return (short)((sign << 15) | 0x7C00);
		return (short)((sign << 15) | (exp << 10) | mant);
	}

	private static int clamp(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
	private static int clamp4(int v) { return clamp(v, 0, 15); }
	private static int clamp6(int v) { return clamp(v, 0, 63); }
}