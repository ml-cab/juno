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
 * Deterministic inverted-dropout mask generation for LoRA training.
 *
 * <p>
 * Masks are regenerated from a stateless hash of
 * {@code (rootSeed, optimizerUpdate, chunkOrdinal, tokenPosition, absoluteLayer,
 * projectionOrdinal, inputIndex)} so forward and backward agree without storing
 * large masks. Inference and validation must not use dropout.
 */
public final class LoraDropout {

	private LoraDropout() {
	}

	public static void validateRate(float rate) {
		if (!Float.isFinite(rate) || rate < 0f || rate >= 1f)
			throw new IllegalArgumentException("dropout rate must be in [0, 1)");
	}

	/** Inverted-dropout keep probability scale {@code 1 / (1 - rate)}. */
	public static float invertedScale(float rate) {
		validateRate(rate);
		if (rate == 0f)
			return 1f;
		return 1f / (1f - rate);
	}

	/**
	 * Whether input coordinate {@code inputIndex} is kept under inverted dropout.
	 */
	public static boolean keep(long rootSeed, int optimizerUpdate, int chunkOrdinal, int tokenPosition,
			int absoluteLayer, int projectionOrdinal, int inputIndex, float rate) {
		validateRate(rate);
		if (rate == 0f)
			return true;
		long h = mix64(rootSeed);
		h = mix64(h ^ (optimizerUpdate * 0x9E3779B97F4A7C15L));
		h = mix64(h ^ (chunkOrdinal * 0xBF58476D1CE4E5B9L));
		h = mix64(h ^ (tokenPosition * 0x94D049BB133111EBL));
		h = mix64(h ^ (absoluteLayer * 0xD6E8FEB86659FD93L));
		h = mix64(h ^ (projectionOrdinal * 0xA24BAED4963EE407L));
		h = mix64(h ^ (inputIndex * 0x9FB21C651E98DF25L));
		// Map top 24 bits to [0, 1).
		float u = ((h >>> 40) & 0xFFFFFFL) * (1.0f / 16777216.0f);
		return u >= rate;
	}

	private static long mix64(long z) {
		z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
		z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
		return z ^ (z >>> 31);
	}
}
