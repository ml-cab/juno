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
import java.util.Objects;
import java.util.Random;

/**
 * Low-Rank Adapter (LoRA) for a single weight projection.
 *
 * <p>
 * For a frozen weight matrix W (shape: outDim × inDim), LoRA inserts a
 * trainable low-rank perturbation:
 *
 * <pre>
 *   ΔW = scale × B × A
 * </pre>
 *
 * where {@code scale} is {@code alpha/rank} (standard) or {@code alpha/√rank}
 * (rsLoRA). A is (rank × inDim), B is (outDim × rank). B is always
 * zero-initialised so ΔW = 0 at the start of training.
 */
public final class LoraAdapter {

	private static final float LEGACY_INIT_STD = 0.01f;

	public final int rank;
	public final int inDim;
	public final int outDim;
	/** Declared alpha (checkpoint identity); not necessarily {@code scale * rank}. */
	public final float alpha;
	/** Effective multiplier applied to {@code B × A}. */
	public final float scale;
	public final LoraScaling scaling;
	public final LoraInitialization initialization;
	public final LoraMode mode;

	final float[] a;
	final float[] b;

	final float[] gradA;
	final float[] gradB;

	/**
	 * Compatibility constructor: standard scaling, legacy-normal A init, plain
	 * LoRA. Prefer {@link #LoraAdapter(LoraAdapterConfig, int, int, Random)} for
	 * new adapters (Kaiming by default via {@link LoraAdapterConfig#of}).
	 */
	public LoraAdapter(int rank, int inDim, int outDim, float alpha, Random rng) {
		this(LoraAdapterConfig.legacy(rank, alpha), inDim, outDim, rng);
	}

	public LoraAdapter(LoraAdapterConfig config, int inDim, int outDim, Random rng) {
		this(config, inDim, outDim, null, null, Objects.requireNonNull(rng, "rng"));
	}

	/**
	 * Load A/B without random initialization. Uses legacy-normal provenance for
	 * the compatibility overload that only supplies alpha.
	 */
	public static LoraAdapter fromWeights(int rank, int inDim, int outDim, float alpha, float[] a, float[] b) {
		return fromWeights(LoraAdapterConfig.legacy(rank, alpha), inDim, outDim, a, b);
	}

	public static LoraAdapter fromWeights(LoraAdapterConfig config, int inDim, int outDim, float[] a, float[] b) {
		return new LoraAdapter(config, inDim, outDim, a, b, null);
	}

	private LoraAdapter(LoraAdapterConfig config, int inDim, int outDim, float[] aSrc, float[] bSrc, Random rng) {
		Objects.requireNonNull(config, "config");
		if (inDim < 1)
			throw new IllegalArgumentException("inDim must be >= 1");
		if (outDim < 1)
			throw new IllegalArgumentException("outDim must be >= 1");

		this.rank = config.rank();
		this.inDim = inDim;
		this.outDim = outDim;
		this.alpha = config.alpha();
		this.scaling = config.scaling();
		this.initialization = config.initialization();
		this.mode = config.mode();
		this.scale = config.effectiveScale();

		this.a = new float[rank * inDim];
		this.b = new float[outDim * rank];
		this.gradA = new float[rank * inDim];
		this.gradB = new float[outDim * rank];

		if (aSrc != null) {
			if (aSrc.length != a.length)
				throw new IllegalArgumentException("A length mismatch: " + aSrc.length + " != " + a.length);
			if (bSrc == null || bSrc.length != b.length)
				throw new IllegalArgumentException("B length mismatch");
			System.arraycopy(aSrc, 0, a, 0, a.length);
			System.arraycopy(bSrc, 0, b, 0, b.length);
		} else {
			initializeA(rng);
			// B stays zero
		}
	}

	private void initializeA(Random rng) {
		switch (initialization) {
		case LEGACY_NORMAL -> {
			for (int i = 0; i < a.length; i++)
				a[i] = (float) (rng.nextGaussian() * LEGACY_INIT_STD);
		}
		case KAIMING_UNIFORM -> {
			float bound = 1f / (float) Math.sqrt(inDim);
			for (int i = 0; i < a.length; i++)
				a[i] = (rng.nextFloat() * 2f - 1f) * bound;
		}
		}
	}

	public LoraAdapterConfig config() {
		return LoraAdapterConfig.of(rank, alpha, scaling, initialization, mode);
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
	 * Restore post-construction init for this adapter's initialization mode: A
	 * re-sampled, B = 0, grads cleared. Makes {@code ΔW = 0} again.
	 */
	public void reinitialize(Random rng) {
		initializeA(Objects.requireNonNull(rng, "rng"));
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
