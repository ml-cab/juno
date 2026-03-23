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

import java.util.IdentityHashMap;
import java.util.Map;

/**
 * Adam optimiser for {@link LoraAdapterSet} parameters.
 *
 * <p>
 * Each adapter's A and B matrices get their own independent first-moment (m)
 * and second-moment (v) buffers, allocated lazily on first use and keyed by
 * adapter identity (not name) so re-loading from a checkpoint with different
 * objects resets the momentum state cleanly.
 *
 * <h3>Update rule (standard Adam with optional weight decay on A only)</h3>
 * 
 * <pre>
 *   m ← β₁ × m + (1−β₁) × g
 *   v ← β₂ × v + (1−β₂) × g²
 *   m̂ = m / (1 − β₁ᵗ)
 *   v̂ = v / (1 − β₂ᵗ)
 *   param ← param − lr × m̂ / (√v̂ + ε)
 * </pre>
 *
 * <p>
 * Note: weight decay is intentionally NOT applied to B because B starts at zero
 * and weight decay would keep pulling it back toward zero, making the adapter
 * learn very slowly. Weight decay on A provides a light regulariser without
 * this pathology.
 *
 * <h3>Usage</h3>
 * 
 * <pre>
 * LoraAdamOptimizer opt = new LoraAdamOptimizer(1e-4, 0.9, 0.999, 1e-8, 0.01);
 * for (int step = 0; step < 1000; step++) {
 * 	adapters.zeroAllGrads();
 * 	float loss = handler.trainStep(tokens);
 * 	opt.step(adapters); // update parameters, increment step counter
 * }
 * </pre>
 */
public final class LoraAdamOptimizer {

	// ── Hyper-parameters ──────────────────────────────────────────────────────

	private final double lr;
	private final double beta1;
	private final double beta2;
	private final double eps;
	private final double weightDecay; // applied to A only

	// ── State ─────────────────────────────────────────────────────────────────

	/** Step counter (1-indexed, incremented in {@link #step(LoraAdapterSet)}). */
	private int t = 0;

	/**
	 * Per-adapter moment buffers. Value is float[4][]: [0] = mA, [1] = vA, [2] =
	 * mB, [3] = vB
	 */
	private final Map<LoraAdapter, float[][]> state = new IdentityHashMap<>();

	// ── Factory ───────────────────────────────────────────────────────────────

	/**
	 * Construct an optimiser with fully-specified hyper-parameters.
	 *
	 * @param lr          learning rate (typical: 1e-4 for rank 8)
	 * @param beta1       first-moment decay (0.9 is standard)
	 * @param beta2       second-moment decay (0.999 is standard)
	 * @param eps         denominator stabiliser (1e-8 is standard)
	 * @param weightDecay L2 regularisation on A matrix (0.01 is light; 0 to
	 *                    disable)
	 */
	public LoraAdamOptimizer(double lr, double beta1, double beta2, double eps, double weightDecay) {
		if (lr <= 0)
			throw new IllegalArgumentException("lr must be > 0");
		if (beta1 <= 0 || beta1 >= 1)
			throw new IllegalArgumentException("beta1 must be in (0,1)");
		if (beta2 <= 0 || beta2 >= 1)
			throw new IllegalArgumentException("beta2 must be in (0,1)");
		this.lr = lr;
		this.beta1 = beta1;
		this.beta2 = beta2;
		this.eps = eps;
		this.weightDecay = weightDecay;
	}

	/**
	 * Sensible defaults: lr=1e-4, β₁=0.9, β₂=0.999, ε=1e-8, weightDecay=0.01.
	 */
	public static LoraAdamOptimizer defaults(double lr) {
		return new LoraAdamOptimizer(lr, 0.9, 0.999, 1e-8, 0.01);
	}

	// ── Optimisation step ─────────────────────────────────────────────────────

	/**
	 * Apply one Adam update to every adapter in the set.
	 *
	 * <p>
	 * Reads gradients from each adapter's {@code gradA}/{@code gradB} accumulators
	 * and updates the corresponding {@code a}/{@code b} arrays in-place. The caller
	 * is responsible for zeroing gradients before the next backward pass.
	 *
	 * @param adapters the adapter set whose parameters should be updated
	 */
	public void step(LoraAdapterSet adapters) {
		t++;
		double bc1 = 1.0 - Math.pow(beta1, t); // bias correction factors
		double bc2 = 1.0 - Math.pow(beta2, t);

		for (LoraAdapter adapter : adapters.all()) {
			float[][] buf = state.computeIfAbsent(adapter, a -> new float[][] { new float[a.a.length], // mA
					new float[a.a.length], // vA
					new float[a.b.length], // mB
					new float[a.b.length] // vB
			});
			updateParams(adapter.a(), adapter.gradA(), buf[0], buf[1], bc1, bc2, true);
			updateParams(adapter.b(), adapter.gradB(), buf[2], buf[3], bc1, bc2, false);
		}
	}

	/** One-parameter Adam update in-place. */
	private void updateParams(float[] param, float[] grad, float[] m, float[] v, double bc1, double bc2,
			boolean applyWeightDecay) {
		double lrCorrected = lr * Math.sqrt(bc2) / bc1;
		for (int i = 0; i < param.length; i++) {
			double g = grad[i];
			if (applyWeightDecay)
				g += weightDecay * param[i]; // L2 reg
			m[i] = (float) (beta1 * m[i] + (1 - beta1) * g);
			v[i] = (float) (beta2 * v[i] + (1 - beta2) * g * g);
			param[i] -= (float) (lrCorrected * m[i] / (Math.sqrt(v[i]) + eps));
		}
	}

	/** Current step count (for learning-rate scheduling). */
	public int step() {
		return t;
	}

	/**
	 * Reset step counter and all moment buffers (e.g. after loading a checkpoint).
	 */
	public void reset() {
		t = 0;
		state.clear();
	}
}