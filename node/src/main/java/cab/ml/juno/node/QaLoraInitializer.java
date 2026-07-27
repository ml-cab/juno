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
import java.util.Collection;
import java.util.List;
import java.util.Random;

import cab.ml.juno.lora.LoraAdapterConfig;
import cab.ml.juno.lora.LoraAdapterSet;
import cab.ml.juno.lora.LoraMode;
import cab.ml.juno.lora.MergeCapability;
import cab.ml.juno.lora.QaLoraAdapter;
import cab.ml.juno.lora.QaLoraEntryMeta;

/**
 * Model-aware QA-LoRA initializer: group width from each tensor's actual GGML
 * type (not the GGUF filename suffix).
 */
public final class QaLoraInitializer {

	private QaLoraInitializer() {
	}

	/**
	 * @param groupWidthOverride {@code <= 0} → auto from {@link QuantizationLayout};
	 *                           otherwise must divide every target {@code inDim}
	 */
	public static LoraAdapterSet create(GgufReader reader, LlamaConfig cfg, Collection<LoraProjection> targets,
			LoraAdapterConfig config, Random rng, int groupWidthOverride, MergeCapability mergeCapability)
			throws IOException {
		if (config.mode() != LoraMode.QA_LORA)
			throw new IllegalArgumentException("QaLoraInitializer requires LoraMode.QA_LORA");
		List<LoraProjection> ordered = LoraProjection.sortedUnique(targets);
		if (ordered.isEmpty())
			throw new IllegalArgumentException("targets must not be empty");
		MergeCapability merge = mergeCapability != null ? mergeCapability : MergeCapability.F32_PRESERVE;
		if (merge == MergeCapability.EXACT_AFFINE)
			throw new IllegalArgumentException("EXACT_AFFINE unavailable for GGUF K-quants");

		LoraAdapterSet set = new LoraAdapterSet();
		for (int li = 0; li < cfg.numLayers(); li++) {
			for (LoraProjection proj : ordered) {
				String name = proj.ggufTensorName(li);
				int ggmlType = reader.tensorType(name);
				int groupWidth = resolveGroupWidth(ggmlType, groupWidthOverride);
				int inDim = proj.inDim(cfg);
				int outDim = proj.outDim(cfg);
				if (inDim % groupWidth != 0)
					throw new IllegalArgumentException("tensor " + name + " inDim=" + inDim
							+ " not divisible by groupWidth=" + groupWidth + " (ggml type " + ggmlType + ")");

				QaLoraAdapter adapter = new QaLoraAdapter(config, inDim, outDim, groupWidth, rng);
				QaLoraEntryMeta meta = QaLoraEntryMeta.of(groupWidth, adapter.groupCount, ggmlType,
						GgufKQuantCodec.ENCODER_ID, merge);
				set.addQa(li, proj.key(), adapter, meta);
				set.putFingerprint(li, proj.key(), DoraInitializer.fingerprint(reader, name));
			}
		}
		return set;
	}

	public static void verifyFingerprints(GgufReader reader, LoraAdapterSet set) throws IOException {
		for (var entry : set.asQaMap().entrySet()) {
			String key = entry.getKey();
			int layer = LoraAdapterSet.keyLayer(key);
			String projKey = LoraAdapterSet.keyProj(key);
			LoraProjection proj = LoraProjection.fromKey(projKey);
			LoraAdapterSet.BaseTensorFingerprint expected = set.getFingerprint(layer, projKey);
			if (expected == null)
				throw new IllegalArgumentException("QA-LoRA adapter missing base fingerprint: " + key);
			LoraAdapterSet.BaseTensorFingerprint actual = DoraInitializer.fingerprint(reader,
					proj.ggufTensorName(layer));
			if (!expected.equals(actual))
				throw new IllegalArgumentException("QA-LoRA base fingerprint mismatch for " + key);

			QaLoraEntryMeta meta = set.getQaMeta(layer, projKey);
			if (meta == null)
				throw new IllegalArgumentException("QA-LoRA missing entry meta: " + key);
			int ggmlType = reader.tensorType(proj.ggufTensorName(layer));
			if (meta.ggmlType() != ggmlType)
				throw new IllegalArgumentException("QA-LoRA ggml type mismatch for " + key + ": checkpoint "
						+ meta.ggmlType() + " vs model " + ggmlType);
			QaLoraAdapter a = entry.getValue();
			if (meta.groupWidth() != a.groupWidth || meta.groupCount() != a.groupCount)
				throw new IllegalArgumentException("QA-LoRA grouping mismatch for " + key);
		}
	}

	static int resolveGroupWidth(int ggmlType, int override) {
		if (override > 0)
			return override;
		QuantizationLayout layout = QuantizationLayout.forType(ggmlType);
		if (layout == null)
			throw new IllegalArgumentException(
					"QA-LoRA requires Q4_K/Q5_K/Q6_K tensor; unsupported GGML type " + ggmlType);
		return layout.subBlockWidth();
	}
}
