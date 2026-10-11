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
 * Buffers for a handler's single-token CPU decode path, one set per decoding
 * thread and handler, so a decode step allocates no per-layer arrays. The
 * residual stream flows through {@link #resid}: a layer reads its input once
 * (into the first residual sum) and leaves its output there for the next layer,
 * and the handler copies it out before returning, because the next token
 * overwrites it.
 */
final class CpuDecodeScratch {

	final float[] xNorm;
	final float[] q;
	final float[] k;
	final float[] v;
	final float[] attnOut;
	final float[] attnProj;
	final float[] resid;
	final float[] gate;
	final float[] up;
	final float[] hidden;
	final float[] ffnOut;
	private float[] scores = new float[0];

	CpuDecodeScratch(LlamaConfig cfg) {
		int H = cfg.hiddenDim();
		int qDim = cfg.numHeads() * cfg.headDim();
		int I = cfg.intermediateSize();
		xNorm = new float[H];
		q = new float[qDim];
		k = new float[cfg.kvDim()];
		v = new float[cfg.kvDim()];
		attnOut = new float[qDim];
		attnProj = new float[H];
		resid = new float[H];
		gate = new float[I];
		up = new float[I];
		hidden = new float[I];
		ffnOut = new float[H];
	}

	/** Attention scores for {@code n} positions; grows, never shrinks. */
	float[] scores(int n) {
		if (scores.length < n)
			scores = new float[Math.max(n, 2 * scores.length)];
		return scores;
	}
}
