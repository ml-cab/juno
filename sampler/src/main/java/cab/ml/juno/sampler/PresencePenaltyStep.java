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
 * OpenAI-style presence penalty on raw logits.
 *
 * <p>For each token that has appeared at least once in the generated history:
 * {@code logits[token] -= presencePenalty}. Penalty {@code 0} is a no-op.
 */
public final class PresencePenaltyStep implements SamplingStep {

	public static final PresencePenaltyStep INSTANCE = new PresencePenaltyStep();

	private PresencePenaltyStep() {
	}

	@Override
	public float[] apply(float[] logits, SamplingParams params, int[] generatedTokens) {
		float penalty = params.presencePenalty();
		if (penalty == 0.0f || generatedTokens == null || generatedTokens.length == 0)
			return logits;

		Set<Integer> seen = new HashSet<>(generatedTokens.length * 2);
		for (int token : generatedTokens)
			seen.add(token);

		for (int tokenId : seen) {
			if (tokenId >= 0 && tokenId < logits.length)
				logits[tokenId] -= penalty;
		}
		return logits;
	}
}
