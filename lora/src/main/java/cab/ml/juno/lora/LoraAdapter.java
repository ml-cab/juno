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
package cab.ml.juno.lora;

import java.util.Arrays;
import java.util.Random;

/**
 * Low-Rank Adapter (LoRA) for a single weight projection.
 *
 * <p>
 * For a frozen weight matrix W (shape: outDim × inDim), LoRA inserts a
 * trainable low-rank perturbation:
 *
 * <pre>
 *   W_effective = W + ΔW,   where ΔW = (alpha / rank) × B × A
 * </pre>
 *
 * A is (rank × inDim), B is (outDim × rank). A is initialised ~N(0, 0.01), B is
 * zero-initialised, so ΔW = 0 at the start of training.
 */
public final class LoraAdapter {

	private static final float INIT_STD = 0.01f;

	public final int rank;
	public final int inDim;
	public final int outDim;
	public final float scale;

	final float[] a;
	final float[] b;

	final float[] gradA;
	final float[] gradB;

	public LoraAdapter(int rank, int inDim, int outDim, float alpha, Random rng) {
		if (rank < 1)
			throw new IllegalArgumentException("rank must be >= 1");
		if (inDim < 1)
			throw new IllegalArgumentException("inDim must be >= 1");
		if (outDim < 1)
			throw new IllegalArgumentException("outDim must be >= 1");

		this.rank = rank;
		this.inDim = inDim;
		this.outDim = outDim;
		this.scale = alpha / rank;

		this.a = new float[rank * inDim];
		this.b = new float[outDim * rank];
		this.gradA = new float[rank * inDim];
		this.gradB = new float[outDim * rank];

		for (int i = 0; i < a.length; i++)
			a[i] = (float) (rng.nextGaussian() * INIT_STD);
	}

	public static LoraAdapter fromWeights(int rank, int inDim, int outDim, float alpha, float[] a, float[] b) {
		LoraAdapter lora = new LoraAdapter(rank, inDim, outDim, alpha, new Random());
		System.arraycopy(a, 0, lora.a, 0, a.length);
		System.arraycopy(b, 0, lora.b, 0, b.length);
		return lora;
	}

	public float[] forward(float[] x) {
		return forwardFromInput(x);
	}

	/**
	 * Training forward with deterministic inverted dropout on the LoRA branch
	 * input. Rate {@code 0} is a bitwise-compatible fast path through
	 * {@link #forward(float[])}.
	 */
	public float[] forwardTrain(float[] x, float dropoutRate, long rootSeed, int optimizerUpdate, int chunkOrdinal,
			int tokenPosition, int absoluteLayer, int projectionOrdinal) {
		if (dropoutRate == 0f)
			return forward(x);
		return forwardFromInput(
				maskedInput(x, dropoutRate, rootSeed, optimizerUpdate, chunkOrdinal, tokenPosition, absoluteLayer,
						projectionOrdinal));
	}

	public float[] backward(float[] gradDelta, float[] x) {
		return backwardFromInput(gradDelta, x, null, 1f);
	}

	/**
	 * Training backward matching {@link #forwardTrain}. Regenerates the same mask
	 * from the hash indices; dropped coordinates contribute zero input gradient.
	 */
	public float[] backwardTrain(float[] gradDelta, float[] x, float dropoutRate, long rootSeed, int optimizerUpdate,
			int chunkOrdinal, int tokenPosition, int absoluteLayer, int projectionOrdinal) {
		if (dropoutRate == 0f)
			return backward(gradDelta, x);
		float invScale = LoraDropout.invertedScale(dropoutRate);
		boolean[] keep = new boolean[inDim];
		float[] xd = new float[inDim];
		for (int j = 0; j < inDim; j++) {
			keep[j] = LoraDropout.keep(rootSeed, optimizerUpdate, chunkOrdinal, tokenPosition, absoluteLayer,
					projectionOrdinal, j, dropoutRate);
			if (keep[j])
				xd[j] = x[j] * invScale;
		}
		return backwardFromInput(gradDelta, xd, keep, invScale);
	}

	private float[] maskedInput(float[] x, float dropoutRate, long rootSeed, int optimizerUpdate, int chunkOrdinal,
			int tokenPosition, int absoluteLayer, int projectionOrdinal) {
		float invScale = LoraDropout.invertedScale(dropoutRate);
		float[] xd = new float[inDim];
		for (int j = 0; j < inDim; j++) {
			if (LoraDropout.keep(rootSeed, optimizerUpdate, chunkOrdinal, tokenPosition, absoluteLayer,
					projectionOrdinal, j, dropoutRate))
				xd[j] = x[j] * invScale;
		}
		return xd;
	}

	private float[] forwardFromInput(float[] x) {
		float[] h = new float[rank];
		for (int r = 0; r < rank; r++) {
			float acc = 0f;
			int base = r * inDim;
			for (int c = 0; c < inDim; c++)
				acc += a[base + c] * x[c];
			h[r] = acc;
		}

		float[] delta = new float[outDim];
		for (int r = 0; r < outDim; r++) {
			float acc = 0f;
			int base = r * rank;
			for (int c = 0; c < rank; c++)
				acc += b[base + c] * h[c];
			delta[r] = acc * scale;
		}
		return delta;
	}

	/**
	 * @param keepMask when non-null, {@code gradX[j]} is zero when {@code !keepMask[j]},
	 *                 otherwise scaled by {@code gradXScale}
	 */
	private float[] backwardFromInput(float[] gradDelta, float[] x, boolean[] keepMask, float gradXScale) {
		float[] h = new float[rank];
		for (int r = 0; r < rank; r++) {
			int base = r * inDim;
			for (int c = 0; c < inDim; c++)
				h[r] += a[base + c] * x[c];
		}

		float[] gradH = new float[rank];
		for (int c = 0; c < rank; c++) {
			float acc = 0f;
			for (int r = 0; r < outDim; r++)
				acc += b[r * rank + c] * gradDelta[r];
			gradH[c] = acc * scale;
		}

		for (int r = 0; r < outDim; r++) {
			int base = r * rank;
			float gScale = gradDelta[r] * scale;
			for (int c = 0; c < rank; c++)
				gradB[base + c] += gScale * h[c];
		}

		for (int r = 0; r < rank; r++) {
			int base = r * inDim;
			float gH = gradH[r];
			for (int j = 0; j < inDim; j++)
				gradA[base + j] += gH * x[j];
		}

		float[] gradX = new float[inDim];
		for (int j = 0; j < inDim; j++) {
			if (keepMask != null && !keepMask[j])
				continue;
			float acc = 0f;
			for (int r = 0; r < rank; r++)
				acc += a[r * inDim + j] * gradH[r];
			gradX[j] = keepMask == null ? acc : acc * gradXScale;
		}
		return gradX;
	}

	public void zeroGrad() {
		Arrays.fill(gradA, 0f);
		Arrays.fill(gradB, 0f);
	}

	/**
	 * Restore post-construction init: A ~ N(0, 0.01), B = 0, grads cleared.
	 * Makes {@code ΔW = 0} again so inference matches the base model.
	 */
	public void reinitialize(Random rng) {
		for (int i = 0; i < a.length; i++)
			a[i] = (float) (rng.nextGaussian() * INIT_STD);
		Arrays.fill(b, 0f);
		zeroGrad();
	}

	/** Copy A/B weights from {@code src} (same shape required). */
	public void copyWeightsFrom(LoraAdapter src) {
		if (src.rank != rank || src.inDim != inDim || src.outDim != outDim)
			throw new IllegalArgumentException("adapter shape mismatch");
		System.arraycopy(src.a, 0, a, 0, a.length);
		System.arraycopy(src.b, 0, b, 0, b.length);
		zeroGrad();
	}

	public float[] a() {
		return a;
	}

	public float[] b() {
		return b;
	}

	public float[] gradA() {
		return gradA;
	}

	public float[] gradB() {
		return gradB;
	}
}
