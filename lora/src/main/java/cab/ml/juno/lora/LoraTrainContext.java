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
 * Per-chunk training context for deterministic LoRA dropout.
 *
 * <p>
 * Inference and validation leave this unset (or use {@link #disabled()}). Only
 * compact seed/rate indices are stored — masks are regenerated on demand.
 */
public final class LoraTrainContext {

	private static final LoraTrainContext DISABLED = new LoraTrainContext(0L, 0f, 0, 0);

	private final long rootSeed;
	private final float dropoutRate;
	private final int optimizerUpdate;
	private final int chunkOrdinal;

	public LoraTrainContext(long rootSeed, float dropoutRate, int optimizerUpdate, int chunkOrdinal) {
		LoraDropout.validateRate(dropoutRate);
		if (optimizerUpdate < 0)
			throw new IllegalArgumentException("optimizerUpdate must be >= 0");
		if (chunkOrdinal < 0)
			throw new IllegalArgumentException("chunkOrdinal must be >= 0");
		this.rootSeed = rootSeed;
		this.dropoutRate = dropoutRate;
		this.optimizerUpdate = optimizerUpdate;
		this.chunkOrdinal = chunkOrdinal;
	}

	public static LoraTrainContext disabled() {
		return DISABLED;
	}

	public boolean dropoutEnabled() {
		return dropoutRate > 0f;
	}

	public long rootSeed() {
		return rootSeed;
	}

	public float dropoutRate() {
		return dropoutRate;
	}

	public int optimizerUpdate() {
		return optimizerUpdate;
	}

	public int chunkOrdinal() {
		return chunkOrdinal;
	}
}
