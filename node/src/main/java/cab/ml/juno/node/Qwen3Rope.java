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

/**
 * Qwen3 rotary embeddings — LLaMA adjacent-pair RoPE with optional YaRN scaling
 * ({@code rope.scaling.type=yarn}), port of llama.cpp {@code ggml_rope_ext}.
 */
final class Qwen3Rope {

	private Qwen3Rope() {
	}

	static void apply(float[] x, int pos, int nHeads, int headDim, Qwen3RopeConfig cfg) {
		if (cfg.yarn()) {
			ropeYarn(x, pos, nHeads, headDim, cfg);
		} else {
			LlamaTransformerHandler.rope(x, pos, nHeads, headDim, cfg.freqBase());
		}
	}

	private static void ropeYarn(float[] x, int pos, int nHeads, int headDim, Qwen3RopeConfig cfg) {
		float thetaScale = (float) Math.pow(cfg.freqBase(), -2.0 / headDim);
		float[] corrDims = new float[2];
		ropeYarnCorrDims(headDim, cfg.originalContextLength(), cfg.freqBase(), 1.0f, 1.0f, corrDims);
		float extFactor = 1.0f;
		float[] cache = new float[headDim];
		ropeCacheInit(pos, cfg.freqScale(), null, corrDims, headDim, extFactor, cfg.attnFactor(), 1.0f, thetaScale,
				cache);
		applyAdjacentRotations(x, nHeads, headDim, cache);
	}

	/** LLaMA-style adjacent-pair rotation within each head. */
	private static void applyAdjacentRotations(float[] x, int nHeads, int headDim, float[] cache) {
		for (int h = 0; h < nHeads; h++) {
			int base = h * headDim;
			for (int i = 0; i < headDim / 2; i++) {
				float cosA = cache[2 * i];
				float sinA = cache[2 * i + 1];
				float x0 = x[base + 2 * i];
				float x1 = x[base + 2 * i + 1];
				x[base + 2 * i] = x0 * cosA - x1 * sinA;
				x[base + 2 * i + 1] = x0 * sinA + x1 * cosA;
			}
		}
	}

	private static void ropeCacheInit(int position, float freqScale, float[] freqFactors, float[] corrDims, int ne0,
			float extFactor, float attnFactor, float sinSign, float thetaScale, float[] cache) {
		float theta = position;
		for (int i0 = 0; i0 < ne0; i0 += 2) {
			float ff = (freqFactors != null && i0 / 2 < freqFactors.length) ? freqFactors[i0 / 2] : 1.0f;
			ropeYarn(theta / ff, freqScale, corrDims, i0, extFactor, attnFactor, cache, i0, sinSign);
			theta *= thetaScale;
		}
	}

	private static void ropeYarn(float thetaExtrap, float freqScale, float[] corrDims, int i0, float extFactor,
			float mscale, float[] cache, int cacheOffset, float sinSign) {
		float thetaInterp = freqScale * thetaExtrap;
		float theta = thetaInterp;
		if (extFactor != 0.0f) {
			float rampMix = ropeYarnRamp(corrDims[0], corrDims[1], i0) * extFactor;
			theta = thetaInterp * (1 - rampMix) + thetaExtrap * rampMix;
			mscale *= 1.0f + 0.1f * (float) Math.log(1.0f / freqScale);
		}
		cache[cacheOffset] = (float) Math.cos(theta) * mscale;
		cache[cacheOffset + 1] = (float) Math.sin(theta) * mscale * sinSign;
	}

	private static float ropeYarnRamp(float low, float high, int i0) {
		float y = (i0 / 2.0f - low) / Math.max(0.001f, high - low);
		return 1 - Math.min(1, Math.max(0, y));
	}

	private static void ropeYarnCorrDims(int nDims, int nCtxOrig, float freqBase, float betaFast, float betaSlow,
			float[] dims) {
		float start = (float) Math.floor(ropeYarnCorrDim(nDims, nCtxOrig, betaFast, freqBase));
		float end = (float) Math.ceil(ropeYarnCorrDim(nDims, nCtxOrig, betaSlow, freqBase));
		dims[0] = Math.max(0, start);
		dims[1] = Math.min(nDims - 1, end);
	}

	private static float ropeYarnCorrDim(int nDims, int nCtxOrig, float nRot, float base) {
		return (float) (nDims * Math.log(nCtxOrig / (nRot * 2 * Math.PI)) / (2 * Math.log(base)));
	}
}
