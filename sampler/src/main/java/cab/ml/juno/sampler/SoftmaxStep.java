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
package cab.ml.juno.sampler;

/**
 * Step 3: Softmax normalization.
 *
 * Converts logits to a probability distribution. Uses the numerically stable
 * formulation: subtract max before exp to avoid float overflow when logits are
 * large.
 *
 * After this step logits[] contains probabilities summing to 1.0. Always
 * applied — required before sampling.
 */
public final class SoftmaxStep implements SamplingStep {

	public static final SoftmaxStep INSTANCE = new SoftmaxStep();

	private SoftmaxStep() {
	}

	@Override
	public float[] apply(float[] logits, SamplingParams params, int[] generatedTokens) {
		// Find max for numerical stability
		float max = Float.NEGATIVE_INFINITY;
		for (float v : logits) {
			if (v > max)
				max = v;
		}

		// Compute exp(logit - max) and accumulate sum
		float sum = 0.0f;
		for (int i = 0; i < logits.length; i++) {
			logits[i] = (float) Math.exp(logits[i] - max);
			sum += logits[i];
		}

		// Normalize
		if (sum > 0.0f) {
			for (int i = 0; i < logits.length; i++) {
				logits[i] /= sum;
			}
		}

		return logits;
	}
}
