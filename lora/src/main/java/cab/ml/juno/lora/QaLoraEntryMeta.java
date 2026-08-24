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
package cab.ml.juno.lora;

import java.util.Objects;

/**
 * Per-adapter Tier-5 metadata for QA-LoRA checkpoints (layout, encoder, merge).
 */
public record QaLoraEntryMeta(
		int groupWidth,
		int groupCount,
		int ggmlType,
		String encoderId,
		MergeCapability mergeCapability,
		QaLoraAdapter.PoolingOp pooling) {

	public QaLoraEntryMeta {
		if (groupWidth < 1)
			throw new IllegalArgumentException("groupWidth must be >= 1");
		if (groupCount < 1)
			throw new IllegalArgumentException("groupCount must be >= 1");
		Objects.requireNonNull(encoderId, "encoderId");
		Objects.requireNonNull(mergeCapability, "mergeCapability");
		Objects.requireNonNull(pooling, "pooling");
		if (mergeCapability == MergeCapability.EXACT_AFFINE)
			throw new IllegalArgumentException(
					"EXACT_AFFINE is unavailable for GGUF K-quants without a representability proof");
	}

	public static QaLoraEntryMeta of(int groupWidth, int groupCount, int ggmlType, String encoderId,
			MergeCapability mergeCapability) {
		return new QaLoraEntryMeta(groupWidth, groupCount, ggmlType, encoderId, mergeCapability,
				QaLoraAdapter.PoolingOp.SUM);
	}
}
