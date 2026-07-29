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

import java.util.Objects;
import java.util.Random;

/**
 * Grouped QA-LoRA adapter: sum-pool contiguous input groups, then apply low-rank
 * {@code A} ({@code rank × groupCount}) and {@code B} ({@code outDim × rank}).
 *
 * <pre>
 *   pooled[g] = sum(x[g*W .. (g+1)*W))
 *   Δ = scale × B × A × pooled
 * </pre>
 *
 * <p>This is not QLoRA. Exact absorption of the grouped delta into GGUF K-quant
 * zero-points is not assumed.
 */
public final class QaLoraAdapter {

	private static final float LEGACY_INIT_STD = 0.01f;

	public final int rank;
	public final int inDim;
	public final int outDim;
	public final int groupWidth;
	public final int groupCount;
	public final float alpha;
	public final float scale;
	public final LoraScaling scaling;
	public final LoraInitialization initialization;
	/** Always {@link LoraMode#QA_LORA}. */
	public final LoraMode mode = LoraMode.QA_LORA;
	public final PoolingOp pooling = PoolingOp.SUM;

	final float[] a; // rank × groupCount
	final float[] b; // outDim × rank
	final float[] gradA;
	final float[] gradB;

	public enum PoolingOp {
		SUM;

		static PoolingOp fromId(int id) {
			PoolingOp[] values = values();
			if (id < 0 || id >= values.length)
				throw new IllegalArgumentException("unknown PoolingOp id: " + id);
			return values[id];
		}
	}

	public QaLoraAdapter(LoraAdapterConfig config, int inDim, int outDim, int groupWidth, Random rng) {
		this(config, inDim, outDim, groupWidth, null, null, Objects.requireNonNull(rng, "rng"));
	}

	public static QaLoraAdapter fromWeights(LoraAdapterConfig config, int inDim, int outDim, int groupWidth,
			float[] a, float[] b) {
		return new QaLoraAdapter(config, inDim, outDim, groupWidth, a, b, null);
	}

	private QaLoraAdapter(LoraAdapterConfig config, int inDim, int outDim, int groupWidth, float[] aSrc, float[] bSrc,
			Random rng) {
		Objects.requireNonNull(config, "config");
		if (config.mode() != LoraMode.QA_LORA)
			throw new IllegalArgumentException("QaLoraAdapter requires LoraMode.QA_LORA");
		if (inDim < 1 || outDim < 1)
			throw new IllegalArgumentException("inDim/outDim must be >= 1");
		if (groupWidth < 1)
			throw new IllegalArgumentException("groupWidth must be >= 1");
		if (inDim % groupWidth != 0)
			throw new IllegalArgumentException(
					"inDim=" + inDim + " not divisible by groupWidth=" + groupWidth);

		this.rank = config.rank();
		this.inDim = inDim;
		this.outDim = outDim;
		this.groupWidth = groupWidth;
		this.groupCount = inDim / groupWidth;
		this.alpha = config.alpha();
		this.scaling = config.scaling();
		this.initialization = config.initialization();
		this.scale = config.effectiveScale();

		this.a = new float[rank * groupCount];
		this.b = new float[outDim * rank];
		this.gradA = new float[a.length];
		this.gradB = new float[b.length];

		if (aSrc != null) {
			if (aSrc.length != a.length)
				throw new IllegalArgumentException("A length mismatch: " + aSrc.length + " != " + a.length);
			if (bSrc == null || bSrc.length != b.length)
				throw new IllegalArgumentException("B length mismatch");
			System.arraycopy(aSrc, 0, a, 0, a.length);
			System.arraycopy(bSrc, 0, b, 0, b.length);
		} else {
			initializeA(rng);
		}
	}

	private void initializeA(Random rng) {
		switch (initialization) {
		case LEGACY_NORMAL -> {
			for (int i = 0; i < a.length; i++)
				a[i] = (float) (rng.nextGaussian() * LEGACY_INIT_STD);
		}
		case KAIMING_UNIFORM -> {
			// Fan-in of A is groupCount (pooled features), not inDim.
			float bound = 1f / (float) Math.sqrt(groupCount);
			for (int i = 0; i < a.length; i++)
				a[i] = (rng.nextFloat() * 2f - 1f) * bound;
		}
		}
	}

	public LoraAdapterConfig config() {
		return LoraAdapterConfig.of(rank, alpha, scaling, initialization, LoraMode.QA_LORA);
	}

	/** Sum-pool {@code x} into {@code groupCount} features. */
	public float[] pool(float[] x) {
		if (x.length != inDim)
			throw new IllegalArgumentException("x length " + x.length + " != inDim " + inDim);
		float[] pooled = new float[groupCount];
		for (int g = 0; g < groupCount; g++) {
			float sum = 0f;
			int base = g * groupWidth;
			for (int i = 0; i < groupWidth; i++)
				sum += x[base + i];
			pooled[g] = sum;
		}
		return pooled;
	}

	/**
	 * Densely expand the effective ΔW ({@code outDim × inDim}) where every input
	 * column in a group shares the same low-rank column
	 * {@code scale * B * A[:, g]}.
	 */
	public float[] expandDenseDelta() {
		float[] deltaW = new float[outDim * inDim];
		for (int o = 0; o < outDim; o++) {
			for (int g = 0; g < groupCount; g++) {
				float col = 0f;
				for (int r = 0; r < rank; r++)
					col += b[o * rank + r] * a[r * groupCount + g];
				col *= scale;
				int rowBase = o * inDim + g * groupWidth;
				for (int i = 0; i < groupWidth; i++)
					deltaW[rowBase + i] = col;
			}
		}
		return deltaW;
	}

	public float[] forward(float[] x) {
		return forwardFromPooled(pool(x));
	}

	public float[] forwardTrain(float[] x, float dropoutRate, long rootSeed, int optimizerUpdate, int chunkOrdinal,
			int tokenPosition, int absoluteLayer, int projectionOrdinal) {
		if (dropoutRate == 0f)
			return forward(x);
		return forwardFromPooled(pool(maskedInput(x, dropoutRate, rootSeed, optimizerUpdate, chunkOrdinal,
				tokenPosition, absoluteLayer, projectionOrdinal)));
	}

	public float[] backward(float[] gradDelta, float[] x) {
		return backwardFromInput(gradDelta, x, null, 1f);
	}

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

	private float[] forwardFromPooled(float[] pooled) {
		float[] h = new float[rank];
		for (int r = 0; r < rank; r++) {
			float acc = 0f;
			int base = r * groupCount;
			for (int g = 0; g < groupCount; g++)
				acc += a[base + g] * pooled[g];
			h[r] = acc;
		}
		float[] delta = new float[outDim];
		for (int o = 0; o < outDim; o++) {
			float acc = 0f;
			int base = o * rank;
			for (int r = 0; r < rank; r++)
				acc += b[base + r] * h[r];
			delta[o] = acc * scale;
		}
		return delta;
	}

	private float[] backwardFromInput(float[] gradDelta, float[] x, boolean[] keepMask, float gradXScale) {
		float[] pooled = pool(x);

		float[] h = new float[rank];
		for (int r = 0; r < rank; r++) {
			int base = r * groupCount;
			for (int g = 0; g < groupCount; g++)
				h[r] += a[base + g] * pooled[g];
		}

		float[] gradH = new float[rank];
		for (int r = 0; r < rank; r++) {
			float acc = 0f;
			for (int o = 0; o < outDim; o++)
				acc += b[o * rank + r] * gradDelta[o];
			gradH[r] = acc * scale;
		}

		for (int o = 0; o < outDim; o++) {
			int base = o * rank;
			float gScale = gradDelta[o] * scale;
			for (int r = 0; r < rank; r++)
				gradB[base + r] += gScale * h[r];
		}

		float[] gradPooled = new float[groupCount];
		for (int r = 0; r < rank; r++) {
			int base = r * groupCount;
			float gH = gradH[r];
			for (int g = 0; g < groupCount; g++) {
				gradA[base + g] += gH * pooled[g];
				gradPooled[g] += a[base + g] * gH;
			}
		}

		// Sum-pool adjoint: each element in the group receives the full pooled gradient.
		float[] gradX = new float[inDim];
		for (int g = 0; g < groupCount; g++) {
			float gp = gradPooled[g];
			int base = g * groupWidth;
			for (int i = 0; i < groupWidth; i++) {
				int j = base + i;
				if (keepMask != null && !keepMask[j])
					continue;
				gradX[j] = gp * gradXScale;
			}
		}
		return gradX;
	}

	public void zeroGrad() {
		java.util.Arrays.fill(gradA, 0f);
		java.util.Arrays.fill(gradB, 0f);
	}

	public void reinitialize(Random rng) {
		java.util.Arrays.fill(a, 0f);
		java.util.Arrays.fill(b, 0f);
		initializeA(Objects.requireNonNull(rng, "rng"));
		zeroGrad();
	}

	public void copyWeightsFrom(QaLoraAdapter src) {
		Objects.requireNonNull(src, "src");
		if (src.rank != rank || src.inDim != inDim || src.outDim != outDim || src.groupWidth != groupWidth)
			throw new IllegalArgumentException("QaLoraAdapter shape mismatch on copy");
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
