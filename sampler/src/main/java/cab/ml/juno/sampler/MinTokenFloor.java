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
 * Holds a sequence open until it has produced a minimum number of tokens, by
 * making the end-of-sequence token unsamplable below that count.
 *
 * <p>Suppressing the token is deliberate, rather than ignoring it once sampled.
 * An ignored end-of-sequence token still has text, so every generation path would
 * have to decide separately not to emit it, and the model would keep proposing it
 * at every step. Masking instead lets the model fall through to its next-best
 * continuation, which is what a caller asking for more tokens wants.
 *
 * <p>The floor is stateless: the number of tokens generated so far is already
 * known at every sampling site, so carrying a second copy of it here would only
 * create a counter that can disagree with the sequence. That also makes the floor
 * safe on the speculative path, where one step samples several positions.
 *
 * <p>It yields in one case. A grammar can reduce the legal set to end-of-sequence
 * alone; masking it then would leave every logit at negative infinity, and a
 * softmax over that is not a distribution. A request for more tokens must not make
 * sampling impossible, so where nothing else survives the token is left alone and
 * the sequence is allowed to end below its minimum.
 *
 * @author Yevhen Soldatov
 */
public final class MinTokenFloor {

	private static final float NEG_INF = Float.NEGATIVE_INFINITY;

	private final int eosTokenId;
	private final int minTokens;

	/**
	 * @param eosTokenId the end-of-sequence token to hold back, ignored when it
	 *                   falls outside the vocabulary
	 * @param minTokens  tokens this sequence must produce before it may end; zero
	 *                   or less disables the floor
	 */
	public MinTokenFloor(int eosTokenId, int minTokens) {
		this.eosTokenId = eosTokenId;
		this.minTokens = minTokens;
	}

	/** Whether the sequence is still being held open at {@code generatedCount}. */
	public boolean holdsOpen(int generatedCount) {
		return minTokens > 0 && generatedCount < minTokens;
	}

	/**
	 * Masks end-of-sequence in place when the sequence is still below its minimum.
	 *
	 * @param logits         logit scores, modified in place; null or empty is a
	 *                       no-op
	 * @param generatedCount tokens generated so far in this sequence
	 */
	public void mask(float[] logits, int generatedCount) {
		if (logits == null || logits.length == 0)
			return;
		if (!holdsOpen(generatedCount))
			return;
		if (eosTokenId < 0 || eosTokenId >= logits.length)
			return;
		if (!anyOtherCandidate(logits))
			return;
		logits[eosTokenId] = NEG_INF;
	}

	/**
	 * Whether some token other than end-of-sequence could still be sampled. Scans
	 * until the first survivor, so a normal distribution costs one comparison.
	 */
	private boolean anyOtherCandidate(float[] logits) {
		for (int i = 0; i < logits.length; i++) {
			if (i == eosTokenId)
				continue;
			float v = logits[i];
			if (v > NEG_INF && !Float.isNaN(v))
				return true;
		}
		return false;
	}
}
