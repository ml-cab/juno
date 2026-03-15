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
 * Step 2: Top-K filtering.
 *
 * Keeps only the K tokens with the highest logit scores. All others are set to
 * -Infinity so softmax will assign them probability 0.
 *
 * Skipped if params.topK() == 0 (disabled) or greedy=true.
 */
public final class TopKStep implements SamplingStep {

	public static final TopKStep INSTANCE = new TopKStep();

	private static final float NEG_INF = Float.NEGATIVE_INFINITY;

	private TopKStep() {
	}

	@Override
	public float[] apply(float[] logits, SamplingParams params, int[] generatedTokens) {
		if (params.greedy())
			return logits;

		int k = params.topK();
		if (k <= 0 || k >= logits.length)
			return logits;

		// Find the k-th largest value (pivot)
		float[] sorted = logits.clone();
		Arrays.sort(sorted);
		float threshold = sorted[sorted.length - k]; // k-th from the top

		// Zero out everything below threshold
		for (int i = 0; i < logits.length; i++) {
			if (logits[i] < threshold) {
				logits[i] = NEG_INF;
			}
		}
		return logits;
	}
}
