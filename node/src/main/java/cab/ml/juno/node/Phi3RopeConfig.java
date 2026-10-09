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

package cab.ml.juno.node;

import java.io.IOException;

/**
 * Phi-3 extended RoPE parameters from GGUF — mirrors the reference implementation's {@code rope_ext}
 * inputs for linear scaling with short/long frequency factor tensors.
 */
record Phi3RopeConfig(
		float freqBase,
		float freqScale,
		float attnFactor,
		int originalContextLength,
		int contextLength,
		float[] ropeFactorsShort,
		float[] ropeFactorsLong) {

	static Phi3RopeConfig from(GgufReader r, LlamaConfig cfg) throws IOException {
		String p = cfg.architecture() + ".";
		float freqBase = r.metaFloat(p + "rope.freq_base", cfg.ropeTheta());
		float ropeScale = r.metaFloat(p + "rope.scaling.factor", 0f);
		if (ropeScale == 0f)
			ropeScale = r.metaFloat(p + "rope.scale_linear", 0f);
		float freqScale = ropeScale == 0f ? 1.0f : 1.0f / ropeScale;
		float attnFactor = r.metaFloat(p + "rope.scaling.attn_factor", 1.0f);
		int origCtx = r.metaInt(p + "rope.scaling.original_context_length", 4096);
		int contextLen = r.metaInt(p + "context_length", origCtx);
		float[] shortF = r.hasTensor("rope_factors_short.weight") ? r.tensor("rope_factors_short.weight") : null;
		float[] longF = r.hasTensor("rope_factors_long.weight") ? r.tensor("rope_factors_long.weight") : null;
		return new Phi3RopeConfig(freqBase, freqScale, attnFactor, origCtx, contextLen, shortF, longF);
	}

	/**
	 * Frequency factors for RoPE. A model that carries both sets rotates with the
	 * short ones: the long ones are for sequences configured beyond
	 * {@link #originalContextLength}, and rotating a short sequence with them
	 * measurably damages it (Phi-3.5-mini's end-of-turn probability falls from
	 * about 0.99 to 0.50). Juno has no per-session context setting, so a sequence
	 * never starts on the long factors; {@link #requirePosition} refuses the
	 * positions that would need them. The trained {@link #contextLength} is not
	 * the selector: for Phi-3.5 it is 131072, which would pick the long factors
	 * for every request.
	 */
	float[] selectFactors() {
		return ropeFactorsShort != null ? ropeFactorsShort : ropeFactorsLong;
	}

	/**
	 * Fails closed when {@code pos} is at or beyond the original training context
	 * of a model whose long-context factors {@link #selectFactors} holds back.
	 * Rotating those positions with the short factors would be silently wrong,
	 * and switching factors mid-sequence would leave the cached keys rotated
	 * with the other set.
	 *
	 * @throws IllegalStateException when the position needs the long factors
	 */
	void requirePosition(int pos) {
		if (pos >= originalContextLength && holdsBackLongFactors())
			throw new IllegalStateException("Phi-3 position " + pos + " reaches the model's original context length of "
					+ originalContextLength + " tokens. Positions beyond it need the long-context RoPE factors, which "
					+ "Juno does not use; keep the prompt and generated tokens within " + originalContextLength + ".");
	}

	/**
	 * Positions a sequence may occupy, {@code [0, positionLimit())}: the original
	 * training context when the long factors are held back (see
	 * {@link #requirePosition}), otherwise unbounded here.
	 */
	int positionLimit() {
		return holdsBackLongFactors() ? originalContextLength : Integer.MAX_VALUE;
	}

	private boolean holdsBackLongFactors() {
		return ropeFactorsShort != null && ropeFactorsLong != null && contextLength > originalContextLength;
	}
}
