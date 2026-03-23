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

import java.util.Arrays;
import java.util.Random;

/**
 * Low-Rank Adapter (LoRA) for a single weight projection.
 *
 * <p>For a frozen weight matrix W (shape: outDim × inDim), LoRA inserts a
 * trainable low-rank perturbation:
 * <pre>
 *   W_effective = W + ΔW,   where ΔW = (alpha / rank) × B × A
 * </pre>
 * A is (rank × inDim), B is (outDim × rank). A is initialised ~N(0, 0.01),
 * B is zero-initialised, so ΔW = 0 at the start of training.
 *
 * <h3>Forward</h3>
 * <pre>
 *   h = A × x                  [rank]
 *   delta = scale × B × h      [outDim]    (scale = alpha / rank)
 * </pre>
 * The caller adds this delta to the frozen projection output before continuing.
 *
 * <h3>Backward</h3>
 * Given the upstream gradient dL/dDelta (shape: [outDim]):
 * <pre>
 *   dL/dB_rc   += scale × gradDelta[r] × h[c]     — accumulate into gradB
 *   dL/dA_rj   += scale × (B^T × gradDelta)[r] × x[j]  — accumulate into gradA
 *   dL/dx[j]    = scale × (A^T × B^T × gradDelta)[j]   — returned for further backprop
 * </pre>
 * Gradients accumulate over multiple calls. Call {@link #zeroGrad()} before
 * each training step.
 *
 * <h3>Thread safety</h3>
 * NOT thread-safe. Each training request must use its own adapter (or the
 * caller must synchronise). Inference ({@link #forward(float[])}) is read-only
 * on A and B and is safe to call concurrently.
 */
public final class LoraAdapter {

    /** Standard deviation for A initialisation. */
    private static final float INIT_STD = 0.01f;

    // ── Dimensions ────────────────────────────────────────────────────────────
    public final int    rank;
    public final int    inDim;
    public final int    outDim;
    /** alpha / rank — pre-computed scaling factor applied in both forward and backward. */
    public final float  scale;

    // ── Parameters ────────────────────────────────────────────────────────────
    /** A matrix, row-major [rank × inDim]. */
    final float[] a;
    /** B matrix, row-major [outDim × rank]. */
    final float[] b;

    // ── Gradient accumulators ─────────────────────────────────────────────────
    /** Accumulated dL/dA, same shape as a. */
    final float[] gradA;
    /** Accumulated dL/dB, same shape as b. */
    final float[] gradB;

    // ── Factory ───────────────────────────────────────────────────────────────

    /**
     * Create a new LoRA adapter with random A and zero B.
     *
     * @param rank   low-rank bottleneck dimension (4, 8, or 16 are common values)
     * @param inDim  input dimension of the projection being adapted (= cols of W)
     * @param outDim output dimension of the projection being adapted (= rows of W)
     * @param alpha  scaling factor; typical value equals rank (giving scale = 1.0)
     * @param rng    random source for A initialisation
     */
    public LoraAdapter(int rank, int inDim, int outDim, float alpha, Random rng) {
        if (rank < 1)  throw new IllegalArgumentException("rank must be >= 1");
        if (inDim < 1) throw new IllegalArgumentException("inDim must be >= 1");
        if (outDim < 1) throw new IllegalArgumentException("outDim must be >= 1");

        this.rank   = rank;
        this.inDim  = inDim;
        this.outDim = outDim;
        this.scale  = alpha / rank;

        this.a     = new float[rank  * inDim];
        this.b     = new float[outDim * rank];
        this.gradA = new float[rank  * inDim];
        this.gradB = new float[outDim * rank];

        // A: small random init; B: zero (ensures ΔW = 0 at start)
        for (int i = 0; i < a.length; i++)
            a[i] = (float) (rng.nextGaussian() * INIT_STD);
        // b and grads already zero from new float[]
    }

    /**
     * Restore an adapter from saved parameters (checkpoint loading).
     * gradA and gradB start at zero.
     */
    public static LoraAdapter fromWeights(int rank, int inDim, int outDim,
                                          float alpha, float[] a, float[] b) {
        LoraAdapter lora = new LoraAdapter(rank, inDim, outDim, alpha, new Random());
        System.arraycopy(a, 0, lora.a, 0, a.length);
        System.arraycopy(b, 0, lora.b, 0, b.length);
        return lora;
    }

    // ── Forward ───────────────────────────────────────────────────────────────

    /**
     * Compute the LoRA delta: {@code scale × B × (A × x)}.
     *
     * <p>This is the additive perturbation to overlay on top of the frozen
     * weight's output. The caller computes {@code frozenOut + lora.forward(x)}.
     *
     * @param x input vector, length {@link #inDim}
     * @return delta vector of length {@link #outDim}
     */
    public float[] forward(float[] x) {
        // h = A × x  [rank]
        float[] h = new float[rank];
        for (int r = 0; r < rank; r++) {
            float acc = 0f;
            int base = r * inDim;
            for (int c = 0; c < inDim; c++)
                acc += a[base + c] * x[c];
            h[r] = acc;
        }

        // delta = scale × B × h  [outDim]
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

    // ── Backward ──────────────────────────────────────────────────────────────

    /**
     * Accumulate parameter gradients and return the input gradient.
     *
     * <p>Chain rule through {@code delta = scale × B × (A × x)}:
     * <ul>
     *   <li>h = A × x
     *   <li>gradH = scale × B^T × gradDelta           — gradient w.r.t. h
     *   <li>gradB += scale × gradDelta ⊗ h^T          — outer product
     *   <li>gradA += gradH ⊗ x^T                      — outer product
     *   <li>gradX  = A^T × gradH                      — returned
     * </ul>
     *
     * @param gradDelta upstream gradient, length {@link #outDim}
     * @param x         the same input that was passed to {@link #forward(float[])}
     *                  during the corresponding forward pass
     * @return gradient w.r.t. x, length {@link #inDim}
     */
    public float[] backward(float[] gradDelta, float[] x) {
        // Recompute h = A × x  (cheaper than storing per-call)
        float[] h = new float[rank];
        for (int r = 0; r < rank; r++) {
            int base = r * inDim;
            for (int c = 0; c < inDim; c++)
                h[r] += a[base + c] * x[c];
        }

        // gradH = scale × B^T × gradDelta   [rank]
        float[] gradH = new float[rank];
        for (int c = 0; c < rank; c++) {
            float acc = 0f;
            for (int r = 0; r < outDim; r++)
                acc += b[r * rank + c] * gradDelta[r];
            gradH[c] = acc * scale;
        }

        // Accumulate dL/dB: B[r,c] += scale × gradDelta[r] × h[c]
        for (int r = 0; r < outDim; r++) {
            int base = r * rank;
            float gScale = gradDelta[r] * scale;
            for (int c = 0; c < rank; c++)
                gradB[base + c] += gScale * h[c];
        }

        // Accumulate dL/dA: A[r,j] += gradH[r] × x[j]
        for (int r = 0; r < rank; r++) {
            int base = r * inDim;
            float gH = gradH[r];
            for (int j = 0; j < inDim; j++)
                gradA[base + j] += gH * x[j];
        }

        // gradX = A^T × gradH   [inDim]
        float[] gradX = new float[inDim];
        for (int j = 0; j < inDim; j++) {
            float acc = 0f;
            for (int r = 0; r < rank; r++)
                acc += a[r * inDim + j] * gradH[r];
            gradX[j] = acc;
        }
        return gradX;
    }

    // ── Utilities ─────────────────────────────────────────────────────────────

    /** Zero all gradient accumulators. Call once before each training step. */
    public void zeroGrad() {
        Arrays.fill(gradA, 0f);
        Arrays.fill(gradB, 0f);
    }

    // Exposed for LoraAdamOptimizer and LoraAdapterSet
    public float[] a()     { return a; }
    public float[] b()     { return b; }
    public float[] gradA() { return gradA; }
    public float[] gradB() { return gradB; }
}