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

	public float[] backward(float[] gradDelta, float[] x) {
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
			float acc = 0f;
			for (int r = 0; r < rank; r++)
				acc += a[r * inDim + j] * gradH[r];
			gradX[j] = acc;
		}
		return gradX;
	}

	public void zeroGrad() {
		Arrays.fill(gradA, 0f);
		Arrays.fill(gradB, 0f);
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
