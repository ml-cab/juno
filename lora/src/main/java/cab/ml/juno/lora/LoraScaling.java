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
 * LoRA delta scaling convention.
 *
 * <ul>
 * <li>{@link #STANDARD} — {@code alpha / rank} (classic LoRA).
 * <li>{@link #RANK_STABILIZED} — {@code alpha / sqrt(rank)} (rsLoRA).
 * </ul>
 */
public enum LoraScaling {

	STANDARD, RANK_STABILIZED;

	/** Effective multiplier applied to {@code B × A}. */
	public float effectiveScale(float alpha, int rank) {
		if (rank < 1)
			throw new IllegalArgumentException("rank must be >= 1");
		return switch (this) {
		case STANDARD -> alpha / rank;
		case RANK_STABILIZED -> alpha / (float) Math.sqrt(rank);
		};
	}

	static LoraScaling fromId(int id) {
		LoraScaling[] values = values();
		if (id < 0 || id >= values.length)
			throw new IllegalArgumentException("unknown LoraScaling id: " + id);
		return values[id];
	}
}
