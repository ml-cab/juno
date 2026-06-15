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
 * Phi-3 extended RoPE parameters from GGUF — mirrors llama.cpp {@code ggml_rope_ext}
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
	 * Frequency factors for RoPE — same rule as {@code llama_model::get_rope_factors}:
	 * uses configured model context length vs original training context, not the
	 * live sequence length.
	 */
	float[] selectFactors() {
		if (contextLength > originalContextLength && ropeFactorsLong != null)
			return ropeFactorsLong;
		if (ropeFactorsShort != null)
			return ropeFactorsShort;
		return ropeFactorsLong;
	}
}
