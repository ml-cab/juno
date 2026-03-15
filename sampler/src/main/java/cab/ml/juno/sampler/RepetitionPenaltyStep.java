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

import java.util.HashSet;
import java.util.Set;

/**
 * Step 5: Repetition penalty.
 *
 * Reduces the probability of tokens that have already appeared in the generated
 * sequence. Applied after softmax, before final sampling.
 *
 * For each token already seen: prob[token] /= repetitionPenalty
 *
 * penalty=1.0 → no effect (disabled) penalty=1.1 → mild suppression (default)
 * penalty=1.5 → strong suppression
 *
 * Skipped if penalty == 1.0 or generatedTokens is empty.
 */
public final class RepetitionPenaltyStep implements SamplingStep {

	public static final RepetitionPenaltyStep INSTANCE = new RepetitionPenaltyStep();

	private RepetitionPenaltyStep() {
	}

	@Override
	public float[] apply(float[] logits, SamplingParams params, int[] generatedTokens) {
		float penalty = params.repetitionPenalty();
		if (penalty == 1.0f || generatedTokens == null || generatedTokens.length == 0) {
			return logits;
		}

		// Build a set of already-seen token IDs for O(1) lookup
		Set<Integer> seen = new HashSet<>(generatedTokens.length * 2);
		for (int token : generatedTokens)
			seen.add(token);

		// Apply penalty in logit space (before softmax).
		// Positive logits are divided (suppressed), negative logits are multiplied
		// (pushed further negative) — both reduce the token's probability.
		for (int tokenId : seen) {
			if (tokenId >= 0 && tokenId < logits.length) {
				if (logits[tokenId] > 0) {
					logits[tokenId] /= penalty;
				} else {
					logits[tokenId] *= penalty;
				}
			}
		}

		return logits;
	}
}