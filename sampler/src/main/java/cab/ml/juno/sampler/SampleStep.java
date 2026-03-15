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

import java.util.concurrent.ThreadLocalRandom;

/**
 * Step 6: Final token selection.
 *
 * Two modes: greedy — argmax: always picks the highest probability token sample
 * — weighted random draw over the probability distribution
 *
 * Returns the selected token ID via a single-element array. The logits array is
 * treated as probabilities at this stage (post-softmax).
 *
 * Thread-safe: uses ThreadLocalRandom, no shared mutable state.
 */
public final class SampleStep {

	public static final SampleStep INSTANCE = new SampleStep();

	private SampleStep() {
	}

	/**
	 * Select the next token from the probability distribution.
	 *
	 * @param probs  probability distribution (output of softmax + filters)
	 * @param params sampling configuration
	 * @return selected token ID
	 */
	public int sample(float[] probs, SamplingParams params) {
		return params.greedy() ? greedy(probs) : weightedSample(probs);
	}

	// ── Greedy: argmax ────────────────────────────────────────────────────────

	private int greedy(float[] probs) {
		int best = 0;
		for (int i = 1; i < probs.length; i++) {
			if (probs[i] > probs[best])
				best = i;
		}
		return best;
	}

	// ── Weighted random draw ──────────────────────────────────────────────────

	private int weightedSample(float[] probs) {
		double r = ThreadLocalRandom.current().nextDouble();
		double cumulative = 0.0;
		for (int i = 0; i < probs.length; i++) {
			cumulative += probs[i];
			if (r < cumulative)
				return i;
		}
		// Fallback — floating point rounding edge case: return last non-zero token
		for (int i = probs.length - 1; i >= 0; i--) {
			if (probs[i] > 0.0f)
				return i;
		}
		return 0;
	}
}
