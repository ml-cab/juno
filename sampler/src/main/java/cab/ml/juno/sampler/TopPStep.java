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

import java.util.Arrays;

/**
 * Step 4: Top-P (nucleus) sampling.
 *
 * Keeps the smallest set of tokens whose cumulative probability exceeds topP.
 * Must be applied AFTER softmax (operates on probabilities, not logits).
 *
 * Example: topP=0.9 keeps the most probable tokens that together account for
 * 90% of the probability mass.
 *
 * Skipped if params.topP() >= 1.0 (disabled) or greedy=true.
 */
public final class TopPStep implements SamplingStep {

	public static final TopPStep INSTANCE = new TopPStep();

	private TopPStep() {
	}

	@Override
	public float[] apply(float[] logits, SamplingParams params, int[] generatedTokens) {
		if (params.greedy())
			return logits;

		float p = params.topP();
		if (p >= 1.0f)
			return logits;

		// Build index array sorted by probability descending
		Integer[] indices = new Integer[logits.length];
		for (int i = 0; i < indices.length; i++)
			indices[i] = i;
		Arrays.sort(indices, (a, b) -> Float.compare(logits[b], logits[a]));

		// Walk sorted tokens, accumulate probability, zero out the rest
		float cumulative = 0.0f;
		boolean thresholdCrossed = false;
		for (int idx : indices) {
			if (thresholdCrossed) {
				logits[idx] = 0.0f;
			} else {
				cumulative += logits[idx];
				if (cumulative >= p) {
					thresholdCrossed = true;
				}
			}
		}

		// Re-normalize so probabilities sum to 1.0
		float sum = 0.0f;
		for (float v : logits)
			sum += v;
		if (sum > 0.0f) {
			for (int i = 0; i < logits.length; i++)
				logits[i] /= sum;
		}

		return logits;
	}
}
