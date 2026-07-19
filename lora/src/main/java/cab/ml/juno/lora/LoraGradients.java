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
 * Global gradient preparation for LoRA adapters: prediction-count
 * normalization, non-finite rejection, and optional L2 clipping.
 *
 * <p>
 * Call after accumulating unnormalized summed gradients across one or more
 * chunks and before {@link LoraAdamOptimizer#step}. Clipping (when enabled)
 * uses the joint L2 norm of every A and B gradient after normalization.
 */
public final class LoraGradients {

	private LoraGradients() {
	}

	/**
	 * @param globalNorm L2 norm of all adapter gradients after dividing by
	 *                   {@code predictionCount} (before clipping)
	 * @param scale      multiplicative factor actually applied to every gradient
	 *                   element (includes 1/{@code predictionCount} and any clip)
	 * @param clipped    whether the clip factor was applied ({@code maxNorm > 0}
	 *                   and normalized norm exceeded {@code maxNorm})
	 */
	public record PrepResult(double globalNorm, float scale, boolean clipped) {
	}

	/**
	 * Normalize adapter gradients by {@code predictionCount}, optionally clip by
	 * global L2 norm, and reject non-finite values.
	 *
	 * <p>
	 * {@code maxNorm == 0} disables clipping but still divides by prediction count.
	 *
	 * @param adapters         adapter set whose {@code gradA}/{@code gradB} are
	 *                         mutated in place
	 * @param predictionCount  total number of prediction tokens in the
	 *                         accumulation group (≥ 1)
	 * @param maxNorm          maximum allowed L2 norm after normalization; 0 =
	 *                         clipping disabled
	 * @return norm before clipping, applied scale, and clipped flag
	 * @throws IllegalArgumentException if {@code predictionCount < 1} or
	 *                                  {@code maxNorm < 0}
	 * @throws IllegalStateException    if any gradient element is NaN or infinite
	 */
	public static PrepResult prepare(LoraAdapterSet adapters, int predictionCount, float maxNorm) {
		if (predictionCount < 1)
			throw new IllegalArgumentException("predictionCount must be >= 1");
		if (maxNorm < 0f)
			throw new IllegalArgumentException("maxNorm must be >= 0");

		double sumSq = 0.0;
		for (LoraAdapter adapter : adapters.all()) {
			sumSq = accumulateSq(adapter.gradA(), sumSq);
			sumSq = accumulateSq(adapter.gradB(), sumSq);
		}

		double rawNorm = Math.sqrt(sumSq);
		double globalNorm = rawNorm / predictionCount;

		float scale = 1f / predictionCount;
		boolean clipped = false;
		if (maxNorm > 0f && globalNorm > maxNorm) {
			scale *= (float) (maxNorm / globalNorm);
			clipped = true;
		}

		if (scale != 1f) {
			for (LoraAdapter adapter : adapters.all()) {
				scaleInPlace(adapter.gradA(), scale);
				scaleInPlace(adapter.gradB(), scale);
			}
		}

		return new PrepResult(globalNorm, scale, clipped);
	}

	private static double accumulateSq(float[] g, double sumSq) {
		for (float v : g) {
			if (!Float.isFinite(v))
				throw new IllegalStateException("non-finite LoRA gradient value: " + v);
			sumSq += (double) v * (double) v;
		}
		return sumSq;
	}

	private static void scaleInPlace(float[] g, float scale) {
		for (int i = 0; i < g.length; i++)
			g[i] *= scale;
	}
}
