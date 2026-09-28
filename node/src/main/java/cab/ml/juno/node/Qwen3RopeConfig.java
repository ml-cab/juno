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
 * Qwen3 extended RoPE parameters from GGUF — YaRN ({@code rope.scaling.type=yarn})
 * or standard RoPE when scaling is absent — and the Q/K pair layout the rotation
 * uses (see {@link RopePairing}).
 */
record Qwen3RopeConfig(
		float freqBase,
		float freqScale,
		float attnFactor,
		int originalContextLength,
		int contextLength,
		boolean yarn,
		RopePairing pairing) {

	/**
	 * The pair layout Qwen3-family files are served with: rotate-half, because
	 * their conversion does not permute Q/K rows (confirmed by perplexity on a
	 * real file; the adjacent layout is two orders of magnitude worse).
	 */
	static final RopePairing PAIRING = RopePairing.SPLIT_HALF;

	/** Same parameters with a different pair layout (test-only override). */
	Qwen3RopeConfig withPairing(RopePairing p) {
		return new Qwen3RopeConfig(freqBase, freqScale, attnFactor, originalContextLength, contextLength, yarn, p);
	}

	static Qwen3RopeConfig from(GgufReader r, LlamaConfig cfg) throws IOException {
		String p = cfg.architecture() + ".";
		float freqBase = r.metaFloat(p + "rope.freq_base", cfg.ropeTheta());
		String scalingType = r.metaString(p + "rope.scaling.type");
		boolean yarn = scalingType != null && "yarn".equalsIgnoreCase(scalingType.strip());
		float ropeFactor = r.metaFloat(p + "rope.scaling.factor", 0f);
		float freqScale = (yarn && ropeFactor > 0f) ? (1.0f / ropeFactor) : 1.0f;
		float attnFactor = r.metaFloat(p + "rope.scaling.attn_factor", 1.0f);
		int origCtx = r.metaInt(p + "rope.scaling.original_context_length", 32768);
		int contextLen = r.metaInt(p + "context_length", origCtx);
		return new Qwen3RopeConfig(freqBase, freqScale, attnFactor, origCtx, contextLen, yarn, PAIRING);
	}

	static Qwen3RopeConfig standard(LlamaConfig cfg) {
		return new Qwen3RopeConfig(cfg.ropeTheta(), 1.0f, 1.0f, 32768, 32768, false, PAIRING);
	}
}
