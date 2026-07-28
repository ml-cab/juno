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
 * Shared LoRA training math helpers used by architecture-specific handlers.
 *
 * <p>Transpose matVec implementations live here so LLaMA, Qwen2, Phi-3, and
 * Qwen3 handlers share one quantized adjoint path. {@link LoraTrainableHandler}
 * retains thin static wrappers for existing unit tests.
 */
public final class LoraTrainingMath {

	private LoraTrainingMath() {
	}

	/**
	 * RMSNorm backward.
	 *
	 * <pre>
	 *   y_i = w_i * x_i * scale,  scale = 1/sqrt(mean(x^2) + eps)
	 *   dL/dx_j = w_j * scale * gradOut_j
	 *           - x_j * (scale^3 / n) * sum_i(gradOut_i * w_i * x_i)
	 * </pre>
	 */
	public static float[] rmsNormBackward(float[] x, float[] w, float[] gradOut, float eps) {
		int n = x.length;
		float ss = 0f;
		for (float v : x)
			ss += v * v;
		float normSq = ss / n + eps;
		float scale = (float) (1.0 / Math.sqrt(normSq));

		float dot = 0f;
		for (int i = 0; i < n; i++)
			dot += gradOut[i] * w[i] * x[i];

		float s3OverN = (scale * scale * scale) / n;
		float[] gradX = new float[n];
		for (int i = 0; i < n; i++)
			gradX[i] = w[i] * scale * gradOut[i] - x[i] * s3OverN * dot;
		return gradX;
	}

	/**
	 * Inverse LLaMA adjacent-pair RoPE: R(-angle) applied in-place to gradients.
	 */
	public static void ropeBackward(float[] g, int pos, int nHeads, int headDim, float ropeTheta) {
		for (int h = 0; h < nHeads; h++) {
			int base = h * headDim;
			for (int i = 0; i < headDim / 2; i++) {
				double freq = 1.0 / Math.pow(ropeTheta, (2.0 * i) / headDim);
				double angle = pos * freq;
				float cosA = (float) Math.cos(angle);
				float sinA = (float) Math.sin(angle);
				float g0 = g[base + 2 * i];
				float g1 = g[base + 2 * i + 1];
				g[base + 2 * i] = g0 * cosA + g1 * sinA;
				g[base + 2 * i + 1] = -g0 * sinA + g1 * cosA;
			}
		}
	}

	/**
	 * Phi-3 / NeoX-style RoPE adjoint with optional {@code attnFactor} scale on
	 * cos/sin. In-place on {@code g} laid out as contiguous heads × headDim.
	 *
	 * <pre>
	 *   dx0 = cos*g0 + sin*g1
	 *   dx1 = -sin*g0 + cos*g1
	 * </pre>
	 * where cos/sin already include {@code attnFactor}.
	 */
	public static void phi3RopeBackward(float[] g, int pos, int nHeads, int headDim, float ropeTheta,
			float attnFactor) {
		for (int h = 0; h < nHeads; h++) {
			int base = h * headDim;
			for (int i = 0; i < headDim / 2; i++) {
				double freq = 1.0 / Math.pow(ropeTheta, (2.0 * i) / headDim);
				double angle = pos * freq;
				float cosA = (float) (Math.cos(angle) * attnFactor);
				float sinA = (float) (Math.sin(angle) * attnFactor);
				float g0 = g[base + i];
				float g1 = g[base + i + headDim / 2];
				g[base + i] = cosA * g0 + sinA * g1;
				g[base + i + headDim / 2] = -sinA * g0 + cosA * g1;
			}
		}
	}

	/**
	 * Per-head RMSNorm backward (Qwen3 Q/K norms). {@code x} and {@code gradOut}
	 * are {@code nHeads * headDim}; {@code w} has length {@code headDim} and is
	 * shared across heads.
	 */
	public static float[] perHeadRmsNormBackward(float[] x, float[] w, float[] gradOut, int nHeads, int headDim,
			float eps) {
		if (x.length != nHeads * headDim || gradOut.length != x.length)
			throw new IllegalArgumentException("x/gradOut length mismatch");
		if (w.length != headDim)
			throw new IllegalArgumentException("w length must equal headDim");
		float[] gradX = new float[x.length];
		for (int h = 0; h < nHeads; h++) {
			int base = h * headDim;
			float[] xh = new float[headDim];
			float[] gh = new float[headDim];
			System.arraycopy(x, base, xh, 0, headDim);
			System.arraycopy(gradOut, base, gh, 0, headDim);
			float[] dx = rmsNormBackward(xh, w, gh, eps);
			System.arraycopy(dx, 0, gradX, base, headDim);
		}
		return gradX;
	}

	/** Transpose matrix–vector multiply: y[cols] = A^T × v. */
	public static float[] transposedMatVec(GgufReader.QuantizedTensor A, float[] v, int rows, int cols) {
		return LoraTrainableHandler.transposedMatVec(A, v, rows, cols);
	}

	/**
	 * Adjoint of {@code Qwen3Rope.apply} — inverse adjacent-pair rotation for
	 * gradient backpropagation. Standard (non-YaRN) delegates to
	 * {@link #ropeBackward}; YaRN rebuilds the same cos/sin cache used in the
	 * forward pass and applies the transpose rotation.
	 *
	 * <p>
	 * Forward rotation: {@code y[2i] = x[2i]*cos - x[2i+1]*sin},
	 * {@code y[2i+1] = x[2i]*sin + x[2i+1]*cos}
	 * <br>
	 * Adjoint (transpose Jacobian): {@code dx[2i] = g[2i]*cos + g[2i+1]*sin},
	 * {@code dx[2i+1] = -g[2i]*sin + g[2i+1]*cos}
	 */
	public static void qwen3RopeBackward(float[] g, int pos, int nHeads, int headDim, Qwen3RopeConfig cfg) {
		if (!cfg.yarn()) {
			ropeBackward(g, pos, nHeads, headDim, cfg.freqBase());
			return;
		}
		float thetaScale = (float) Math.pow(cfg.freqBase(), -2.0 / headDim);
		float[] corrDims = new float[2];
		yarnCorrDims(headDim, cfg.originalContextLength(), cfg.freqBase(), 1.0f, 1.0f, corrDims);
		float[] cache = new float[headDim];
		yarnCacheInit(pos, cfg.freqScale(), corrDims, headDim, 1.0f, cfg.attnFactor(), thetaScale, cache);
		for (int h = 0; h < nHeads; h++) {
			int base = h * headDim;
			for (int i = 0; i < headDim / 2; i++) {
				float cosA = cache[2 * i];
				float sinA = cache[2 * i + 1];
				float g0 = g[base + 2 * i];
				float g1 = g[base + 2 * i + 1];
				g[base + 2 * i] = g0 * cosA + g1 * sinA;
				g[base + 2 * i + 1] = -g0 * sinA + g1 * cosA;
			}
		}
	}

	private static void yarnCorrDims(int nDims, int nCtxOrig, float freqBase, float betaFast, float betaSlow,
			float[] dims) {
		float start = (float) Math.floor(yarnCorrDim(nDims, nCtxOrig, betaFast, freqBase));
		float end = (float) Math.ceil(yarnCorrDim(nDims, nCtxOrig, betaSlow, freqBase));
		dims[0] = Math.max(0, start);
		dims[1] = Math.min(nDims - 1, end);
	}

	private static float yarnCorrDim(int nDims, int nCtxOrig, float nRot, float base) {
		return (float) (nDims * Math.log(nCtxOrig / (nRot * 2 * Math.PI)) / (2 * Math.log(base)));
	}

	/**
	 * Builds the YaRN cos/sin cache in {@code cache[0..ne0-1]} (interleaved
	 * cos/sin per pair). Mirrors {@code Qwen3Rope.ropeCacheInit} with
	 * {@code freqFactors=null}, {@code extFactor=1}, {@code sinSign=1}.
	 */
	private static void yarnCacheInit(int position, float freqScale, float[] corrDims, int ne0, float extFactor,
			float attnFactor, float thetaScale, float[] cache) {
		float theta = position;
		for (int i0 = 0; i0 < ne0; i0 += 2) {
			float thetaExtrap = theta;
			float thetaInterp = freqScale * thetaExtrap;
			float t = thetaInterp;
			float mscale = attnFactor;
			if (extFactor != 0.0f) {
				float rampMix = yarnRamp(corrDims[0], corrDims[1], i0) * extFactor;
				t = thetaInterp * (1 - rampMix) + thetaExtrap * rampMix;
				mscale = attnFactor * (1.0f + 0.1f * (float) Math.log(1.0f / freqScale));
			}
			cache[i0] = (float) Math.cos(t) * mscale;
			cache[i0 + 1] = (float) Math.sin(t) * mscale;
			theta *= thetaScale;
		}
	}

	private static float yarnRamp(float low, float high, int i0) {
		float y = (i0 / 2.0f - low) / Math.max(0.001f, high - low);
		return 1 - Math.min(1, Math.max(0, y));
	}
}
