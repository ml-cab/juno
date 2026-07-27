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

import cab.ml.juno.lora.DoraMagnitude;
import cab.ml.juno.lora.DoraProjection;
import cab.ml.juno.lora.LoraAdapter;
import cab.ml.juno.lora.LoraAdapterSet;
import cab.ml.juno.lora.LoraMode;
import cab.ml.juno.lora.MergeCapability;
import cab.ml.juno.lora.QaLoraAdapter;
import cab.ml.juno.lora.QaLoraEntryMeta;

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
 *   "L:wq"    to blk.L.attn_q.weight
 *   "L:wk"    to blk.L.attn_k.weight
 *   "L:wv"    to blk.L.attn_v.weight
 *   "L:wo"    to blk.L.attn_output.weight
 *   "L:wgate" to blk.L.ffn_gate.weight
 *   "L:wup"   to blk.L.ffn_up.weight
 *   "L:wdown" to blk.L.ffn_down.weight
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

	private LoraMerge() {}

	public record Result(int adaptersApplied, List<String> tensorsPatched, List<String> skipped,
			List<TensorMergeReport> reports) {}

	/**
	 * Per-tensor report for projected / F32 merge paths.
	 *
	 * @param requantization true when {@link MergeCapability#SOURCE_TYPE_PROJECTED}
	 */
	public record TensorMergeReport(String tensorName, int sourceType, int destType, boolean requantization,
			double targetDeltaNorm, double deltaRetention, double rmse, double maxAbsError, long changedBlocks) {}

	/**
	 * Merge with default {@link MergeCapability#F32_PRESERVE} (or per-adapter QA meta).
	 */
	public static Result merge(Path modelPath, Path loraPath, Path outputPath) throws IOException {
		return merge(modelPath, loraPath, outputPath, null);
	}

	/**
	 * @param capabilityOverride when non-null, forces this policy for all adapters;
	 *                           otherwise dense LoRA uses F32_PRESERVE and QA uses
	 *                           each entry's stored {@link MergeCapability}
	 */
	public static Result merge(Path modelPath, Path loraPath, Path outputPath, MergeCapability capabilityOverride)
			throws IOException {
		long t0 = System.currentTimeMillis();
		LoraAdapterSet adapters = LoraAdapterSet.load(loraPath);
		LoraMetricsIdentity identity = LoraMetricsIdentity.fromAdapterSet(adapters, "", "cpu");
		try {
			Result result = mergeLoaded(modelPath, adapters, outputPath, capabilityOverride);
			commitMergeEvent(identity, capabilityOverride, result, outputPath, System.currentTimeMillis() - t0, true,
					"");
			return result;
		} catch (IOException | RuntimeException ex) {
			commitMergeEvent(identity, capabilityOverride, null, outputPath, System.currentTimeMillis() - t0, false,
					shortError(ex));
			throw ex;
		}
	}

	private static String shortError(Throwable ex) {
		String msg = ex.getMessage();
		if (msg == null || msg.isBlank())
			msg = ex.getClass().getSimpleName();
		return msg.length() > 120 ? msg.substring(0, 117) + "..." : msg;
	}

	private static void commitMergeEvent(LoraMetricsIdentity identity, MergeCapability override, Result result,
			Path outputPath, long durationMs, boolean success, String error) {
		LoraMergeEvent ev = new LoraMergeEvent();
		ev.begin();
		if (identity != null) {
			identity.apply(ev);
			if (override != null)
				ev.mergeCapability = LoraMetricsIdentity.mergeCapabilityLabel(override);
		}
		ev.durationMs = durationMs;
		ev.success = success;
		ev.error = error != null ? error : "";
		if (result != null) {
			ev.tensorsPatched = result.tensorsPatched().size();
			try {
				if (outputPath != null && java.nio.file.Files.isRegularFile(outputPath))
					ev.bytesWritten = java.nio.file.Files.size(outputPath);
			} catch (IOException ignored) {
			}
			aggregateProjectedMetrics(ev, result.reports());
		}
		ev.commit();
	}

	private static void aggregateProjectedMetrics(LoraMergeEvent ev, List<TensorMergeReport> reports) {
		if (reports == null || reports.isEmpty())
			return;
		double sumRmse = 0;
		double maxAbs = 0;
		double sumRetention = 0;
		long changed = 0;
		int n = 0;
		int projected = 0;
		for (TensorMergeReport r : reports) {
			if (!r.requantization())
				continue;
			projected++;
			sumRmse += r.rmse();
			if (r.maxAbsError() > maxAbs)
				maxAbs = r.maxAbsError();
			sumRetention += r.deltaRetention();
			changed += r.changedBlocks();
			n++;
		}
		if (n == 0)
			return;
		ev.rmse = (float) (sumRmse / n);
		ev.maxAbsError = (float) maxAbs;
		ev.deltaRetention = (float) (sumRetention / n);
		ev.changedBlocks = changed;
		ev.totalBlocks = projected;
	}

	private static Result mergeLoaded(Path modelPath, LoraAdapterSet adapters, Path outputPath,
			MergeCapability capabilityOverride) throws IOException {
		Map<String, String> keyByTensor = new java.util.LinkedHashMap<>();
		List<String> skipped = new ArrayList<>();
		for (Map.Entry<String, LoraAdapter> entry : adapters.asMap().entrySet()) {
			String key = entry.getKey();
			int layer = LoraAdapterSet.keyLayer(key);
			LoraProjection proj = LoraProjection.fromKey(LoraAdapterSet.keyProj(key));
			keyByTensor.put(proj.ggufTensorName(layer), key);
		}
		for (Map.Entry<String, QaLoraAdapter> entry : adapters.asQaMap().entrySet()) {
			String key = entry.getKey();
			int layer = LoraAdapterSet.keyLayer(key);
			LoraProjection proj = LoraProjection.fromKey(LoraAdapterSet.keyProj(key));
			keyByTensor.put(proj.ggufTensorName(layer), key);
		}

		List<String> patched = new ArrayList<>();
		List<TensorMergeReport> reports = new ArrayList<>();

		try (GgufReader reader = GgufReader.open(modelPath);
			 FileChannel srcCh = FileChannel.open(modelPath, StandardOpenOption.READ);
			 FileChannel outCh = FileChannel.open(outputPath,
					 StandardOpenOption.WRITE, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {

			DoraInitializer.verifyFingerprints(reader, adapters);
			QaLoraInitializer.verifyFingerprints(reader, adapters);

			for (String tName : new ArrayList<>(keyByTensor.keySet()))
				if (!reader.hasTensor(tName)) {
					skipped.add(tName + " (tensor not in model)");
					keyByTensor.remove(tName);
				}

			// Resolve per-tensor merge policy up front
			Map<String, MergeCapability> policyByTensor = new java.util.LinkedHashMap<>();
			for (var e : keyByTensor.entrySet()) {
				MergeCapability pol = resolvePolicy(adapters, e.getValue(), capabilityOverride);
				if (pol == MergeCapability.SIDECAR_ONLY)
					throw new IllegalArgumentException(
							"SIDECAR_ONLY forbids merge for " + e.getKey() + "; use overlay playback");
				if (pol == MergeCapability.EXACT_AFFINE)
					throw new IllegalArgumentException(
							"EXACT_AFFINE is unavailable for GGUF K-quants (" + e.getKey() + ")");
				if (pol == MergeCapability.UNSUPPORTED)
					throw new IllegalArgumentException("UNSUPPORTED merge for " + e.getKey());
				policyByTensor.put(e.getKey(), pol);
			}

			List<String> tensorOrder = reader.tensorOrder();

			long[] newDataOffsets = new long[tensorOrder.size()];
			long cursor = 0L;
			for (int i = 0; i < tensorOrder.size(); i++) {
				String name = tensorOrder.get(i);
				newDataOffsets[i] = cursor;
				if (!keyByTensor.containsKey(name)) {
					cursor += GgufReader.rawByteCount(reader.tensorType(name), reader.tensorNelems(name));
				} else if (policyByTensor.get(name) == MergeCapability.SOURCE_TYPE_PROJECTED) {
					cursor += GgufReader.rawByteCount(reader.tensorType(name), reader.tensorNelems(name));
				} else {
					cursor += reader.tensorNelems(name) * 4L;
				}
			}

			long headerStart = reader.ggufFileOffset();
			long headerEnd = reader.metadataSectionEnd();
			srcCh.transferTo(headerStart, headerEnd - headerStart, outCh);

			for (int i = 0; i < tensorOrder.size(); i++) {
				String name = tensorOrder.get(i);
				int type;
				if (!keyByTensor.containsKey(name))
					type = reader.tensorType(name);
				else if (policyByTensor.get(name) == MergeCapability.SOURCE_TYPE_PROJECTED)
					type = reader.tensorType(name);
				else
					type = TYPE_F32;
				writeTensorInfoEntry(outCh, name, reader.tensorDims(name), type, newDataOffsets[i]);
			}

			long pos = outCh.position();
			long aligned = ((pos + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
			if (aligned > pos)
				outCh.write(ByteBuffer.allocate((int) (aligned - pos)));

			for (String name : tensorOrder) {
				if (!keyByTensor.containsKey(name)) {
					outCh.write(ByteBuffer.wrap(reader.tensorRaw(name).data()));
					continue;
				}
				String key = keyByTensor.get(name);
				MergeCapability pol = policyByTensor.get(name);
				int sourceType = reader.tensorType(name);
				byte[] rawOriginal = reader.tensorRaw(name).data();
				float[] w = reader.tensor(name);
				long[] dims = reader.tensorDims(name);
				int outDim = (int) dims[1];
				int inDim = (int) dims[0];
				int layer = LoraAdapterSet.keyLayer(key);
				String proj = LoraAdapterSet.keyProj(key);

				float[] before = w.clone();
				LoraAdapter dense = adapters.asMap().get(key);
				QaLoraAdapter qa = adapters.asQaMap().get(key);
				if (dense != null)
					applyAdapter(w, dense, adapters.getMagnitude(layer, proj), outDim, inDim);
				else if (qa != null)
					applyQaDelta(w, qa, outDim, inDim);
				else
					throw new IllegalStateException("missing adapter for " + key);

				float[] targetDelta = new float[w.length];
				double targetNormSq = 0;
				for (int i = 0; i < w.length; i++) {
					targetDelta[i] = w[i] - before[i];
					targetNormSq += (double) targetDelta[i] * targetDelta[i];
				}
				double targetDeltaNorm = Math.sqrt(targetNormSq);

				if (pol == MergeCapability.F32_PRESERVE) {
					ByteBuffer f32 = ByteBuffer.allocate(w.length * 4).order(ByteOrder.LITTLE_ENDIAN);
					for (float f : w) f32.putFloat(f);
					f32.flip();
					outCh.write(f32);
					reports.add(new TensorMergeReport(name, sourceType, TYPE_F32, false, targetDeltaNorm, 1.0, 0, 0, 0));
				} else if (pol == MergeCapability.SOURCE_TYPE_PROJECTED) {
					if (QuantizationLayout.forType(sourceType) == null)
						throw new IllegalArgumentException(
								"SOURCE_TYPE_PROJECTED requires Q4_K/Q5_K/Q6_K for " + name + ", got type "
										+ sourceType);
					// Zero conceptual delta → byte-identical copy (no decode/re-encode).
					if (targetDeltaNorm == 0.0) {
						outCh.write(ByteBuffer.wrap(GgufQuantCodec.copyRawUnchanged(rawOriginal)));
						reports.add(new TensorMergeReport(name, sourceType, sourceType, true, 0, 1.0, 0, 0, 0));
					} else {
						byte[] encoded = GgufQuantCodec.encode(w, sourceType);
						float[] roundTrip = GgufQuantCodec.decode(encoded, sourceType);
						QuantizedMergeMetrics recon = QuantizedMergeMetrics.ofReconstruction(w, roundTrip);
						float[] retainedDelta = new float[w.length];
						for (int i = 0; i < w.length; i++)
							retainedDelta[i] = roundTrip[i] - before[i];
						double retention = QuantizedMergeMetrics.deltaRetention(targetDelta, retainedDelta);
						long changed = 0;
						QuantizationLayout layout = QuantizationLayout.require(sourceType);
						int bb = layout.blockBytes();
						for (int off = 0; off < encoded.length; off += bb) {
							boolean diff = false;
							for (int j = 0; j < bb; j++) {
								if (encoded[off + j] != rawOriginal[off + j]) {
									diff = true;
									break;
								}
							}
							if (diff) changed++;
						}
						outCh.write(ByteBuffer.wrap(encoded));
						reports.add(new TensorMergeReport(name, sourceType, sourceType, true, targetDeltaNorm,
								retention, recon.rmse(), recon.maxAbsError(), changed));
					}
				} else {
					throw new IllegalStateException("unhandled merge policy " + pol);
				}
				patched.add(name);
			}
		}
		return new Result(patched.size(), List.copyOf(patched), List.copyOf(skipped), List.copyOf(reports));
	}

	private static MergeCapability resolvePolicy(LoraAdapterSet adapters, String key, MergeCapability override) {
		if (override != null)
			return override;
		QaLoraEntryMeta meta = adapters.qaMeta().get(key);
		if (meta != null)
			return meta.mergeCapability();
		return MergeCapability.F32_PRESERVE;
	}

	static void applyQaDelta(float[] w, QaLoraAdapter qa, int outDim, int inDim) {
		if (qa.outDim != outDim || qa.inDim != inDim)
			throw new IllegalArgumentException("QA adapter/tensor dimension mismatch");
		float[] deltaW = qa.expandDenseDelta();
		for (int i = 0; i < w.length; i++)
			w[i] += deltaW[i];
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

	// ── LoRA / rsLoRA / DoRA merge formulas ───────────────────────────────────

	/**
	 * Apply adapter to dense {@code w} (row-major out×in) in place.
	 * <ul>
	 * <li>LoRA/rsLoRA: {@code W += scale·B·A}
	 * <li>DoRA: {@code W ← (magnitude/‖direction‖) ⊙ direction} with
	 * {@code direction = W + scale·B·A}
	 * </ul>
	 */
	static void applyAdapter(float[] w, LoraAdapter lora, DoraMagnitude magnitude, int outDim, int inDim) {
		if (lora.outDim != outDim || lora.inDim != inDim)
			throw new IllegalArgumentException("adapter/tensor dimension mismatch");
		applyDelta(w, lora, outDim, inDim);
		if (lora.mode != LoraMode.DORA)
			return;
		if (magnitude == null)
			throw new IllegalArgumentException("DoRA merge requires magnitude");
		if (magnitude.length() != outDim)
			throw new IllegalArgumentException("DoRA magnitude length mismatch");
		float[] mag = magnitude.values();
		for (int r = 0; r < outDim; r++) {
			int base = r * inDim;
			double sumSq = 0;
			for (int c = 0; c < inDim; c++) {
				float v = w[base + c];
				sumSq += (double) v * v;
			}
			float norm = (float) Math.sqrt(sumSq);
			float coeff = mag[r] / Math.max(norm, DoraProjection.EPS);
			if (!Float.isFinite(coeff))
				throw new IllegalArgumentException("non-finite DoRA merge coefficient at row " + r);
			for (int c = 0; c < inDim; c++)
				w[base + c] *= coeff;
		}
	}

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
			case TYPE_Q4_K -> GgufKQuantCodec.encodeQ4K(data);
			case TYPE_Q5_K -> GgufKQuantCodec.encodeQ5K(data);
			case TYPE_Q6_K -> GgufKQuantCodec.encodeQ6K(data);
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

	// Q4_K / Q5_K / Q6_K encode: see GgufKQuantCodec (encoder id juno-kquant-v1).

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
}