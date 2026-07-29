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
 * Immutable adapter identity: rank, declared alpha, scaling, initialization, and
 * mode. Separated from training policy ({@code LoraTrainingConfig}).
 *
 * <p>
 * Declared alpha is stored separately from the effective scale derived by
 * {@link LoraScaling#effectiveScale(float, int)}.
 */
public final class LoraAdapterConfig {

	private final int rank;
	private final float alpha;
	private final LoraScaling scaling;
	private final LoraInitialization initialization;
	private final LoraMode mode;

	private LoraAdapterConfig(int rank, float alpha, LoraScaling scaling, LoraInitialization initialization,
			LoraMode mode) {
		if (rank < 1)
			throw new IllegalArgumentException("rank must be >= 1");
		if (!Float.isFinite(alpha))
			throw new IllegalArgumentException("alpha must be finite");
		if (scaling == null)
			throw new IllegalArgumentException("scaling must be non-null");
		if (initialization == null)
			throw new IllegalArgumentException("initialization must be non-null");
		if (mode == null)
			throw new IllegalArgumentException("mode must be non-null");
		this.rank = rank;
		this.alpha = alpha;
		this.scaling = scaling;
		this.initialization = initialization;
		this.mode = mode;
	}

	/**
	 * Compatibility defaults: standard scaling, Kaiming-uniform A, plain LoRA.
	 */
	public static LoraAdapterConfig of(int rank, float alpha) {
		return of(rank, alpha, LoraScaling.STANDARD, LoraInitialization.KAIMING_UNIFORM, LoraMode.LORA);
	}

	public static LoraAdapterConfig of(int rank, float alpha, LoraScaling scaling, LoraInitialization initialization,
			LoraMode mode) {
		return new LoraAdapterConfig(rank, alpha, scaling, initialization, mode);
	}

	/** Classic LoRA with legacy {@code N(0, 0.01)} A init (pre-Tier-3 default). */
	public static LoraAdapterConfig legacy(int rank, float alpha) {
		return of(rank, alpha, LoraScaling.STANDARD, LoraInitialization.LEGACY_NORMAL, LoraMode.LORA);
	}

	public int rank() {
		return rank;
	}

	/** Declared alpha (not the effective scale). */
	public float alpha() {
		return alpha;
	}

	public LoraScaling scaling() {
		return scaling;
	}

	public LoraInitialization initialization() {
		return initialization;
	}

	public LoraMode mode() {
		return mode;
	}

	public float effectiveScale() {
		return scaling.effectiveScale(alpha, rank);
	}
}
