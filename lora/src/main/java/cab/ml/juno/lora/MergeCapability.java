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

/**
 * Explicit merge output policy for LoRA / QA-LoRA checkpoints.
 *
 * <p>Never silently fall back from {@link #EXACT_AFFINE} to
 * {@link #SOURCE_TYPE_PROJECTED}. {@link #F32_PRESERVE} is the product default.
 * {@link #SOURCE_TYPE_PROJECTED} is approximate requantization, not exact
 * QA-LoRA zero-point merge.
 */
public enum MergeCapability {

	SIDECAR_ONLY,
	F32_PRESERVE,
	SOURCE_TYPE_PROJECTED,
	EXACT_AFFINE,
	UNSUPPORTED;

	public static MergeCapability fromId(int id) {
		MergeCapability[] values = values();
		if (id < 0 || id >= values.length)
			throw new IllegalArgumentException("unknown MergeCapability id: " + id);
		return values[id];
	}
}
