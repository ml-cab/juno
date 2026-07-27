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

import java.util.IdentityHashMap;
import java.util.Map;

/**
 * Decoupled AdamW optimiser for {@link LoraAdapterSet} parameters with optional
 * LoRA+ A/B learning-rate groups.
 *
 * <p>
 * Raw gradients feed Adam moments. Decoupled weight decay is applied to A only,
 * using the scheduled (uncorrected) A learning rate and the pre-update
 * parameter value. B is never decayed. LoRA+ scales B's learning rate by
 * {@code loraPlusRatio}; ratio {@code 1.0} matches ordinary non-LoRA+ updates.
 *
 * <p>
 * Gradient clipping must happen before {@link #step}; decay is never included in
 * the global gradient norm.
 */
public final class LoraAdamOptimizer {

	private final double baseLr;
	private final double beta1;
	private final double beta2;
	private final double eps;
	private final double weightDecay;
	private final double loraPlusRatio;

	private int t = 0;
	private double lastLrA = Double.NaN;
	private double lastLrB = Double.NaN;

	private final Map<LoraAdapter, float[][]> state = new IdentityHashMap<>();
	private final Map<QaLoraAdapter, float[][]> qaState = new IdentityHashMap<>();
	private final Map<DoraMagnitude, float[][]> magState = new IdentityHashMap<>();

	public LoraAdamOptimizer(double lr, double beta1, double beta2, double eps, double weightDecay) {
		this(lr, beta1, beta2, eps, weightDecay, 1.0);
	}

	public LoraAdamOptimizer(double lr, double beta1, double beta2, double eps, double weightDecay,
			double loraPlusRatio) {
		validateBaseLr(lr);
		if (beta1 <= 0 || beta1 >= 1)
			throw new IllegalArgumentException("beta1 must be in (0,1)");
		if (beta2 <= 0 || beta2 >= 1)
			throw new IllegalArgumentException("beta2 must be in (0,1)");
		if (!Double.isFinite(eps) || eps <= 0)
			throw new IllegalArgumentException("eps must be finite and > 0");
		if (!Double.isFinite(weightDecay) || weightDecay < 0)
			throw new IllegalArgumentException("weightDecay must be finite and >= 0");
		if (!Double.isFinite(loraPlusRatio) || loraPlusRatio <= 0)
			throw new IllegalArgumentException("loraPlusRatio must be finite and > 0");
		this.baseLr = lr;
		this.beta1 = beta1;
		this.beta2 = beta2;
		this.eps = eps;
		this.weightDecay = weightDecay;
		this.loraPlusRatio = loraPlusRatio;
	}

	public static LoraAdamOptimizer defaults(double lr) {
		return new LoraAdamOptimizer(lr, 0.9, 0.999, 1e-8, 0.01, 1.0);
	}

	/** Step using the configured base learning rate. */
	public void step(LoraAdapterSet adapters) {
		step(adapters, baseLr);
	}

	/**
	 * One AdamW update with a scheduled base learning rate. A uses
	 * {@code learningRate}; B uses {@code learningRate * loraPlusRatio}. DoRA
	 * magnitude vectors (when present) use the A learning rate with weight decay
	 * off, then {@link LoraAdapterSet#invalidateDoraCaches()} is called.
	 */
	public void step(LoraAdapterSet adapters, double learningRate) {
		validateStepLr(learningRate);
		t++;
		double lrA = learningRate;
		double lrB = learningRate * loraPlusRatio;
		lastLrA = lrA;
		lastLrB = lrB;
		double bc1 = 1.0 - Math.pow(beta1, t);
		double bc2 = 1.0 - Math.pow(beta2, t);

		boolean touchedDora = false;
		for (LoraAdapter adapter : adapters.all()) {
			float[][] buf = state.computeIfAbsent(adapter, a -> new float[][] { new float[a.a.length],
					new float[a.a.length], new float[a.b.length], new float[a.b.length] });
			updateParams(adapter.a(), adapter.gradA(), buf[0], buf[1], bc1, bc2, lrA, true);
			updateParams(adapter.b(), adapter.gradB(), buf[2], buf[3], bc1, bc2, lrB, false);
			if (adapter.mode == LoraMode.DORA)
				touchedDora = true;
		}
		for (QaLoraAdapter adapter : adapters.allQa()) {
			float[][] buf = qaState.computeIfAbsent(adapter, a -> new float[][] { new float[a.a.length],
					new float[a.a.length], new float[a.b.length], new float[a.b.length] });
			updateParams(adapter.a(), adapter.gradA(), buf[0], buf[1], bc1, bc2, lrA, true);
			updateParams(adapter.b(), adapter.gradB(), buf[2], buf[3], bc1, bc2, lrB, false);
		}
		for (DoraMagnitude mag : adapters.magnitudes().values()) {
			float[][] buf = magState.computeIfAbsent(mag,
					m -> new float[][] { new float[m.length()], new float[m.length()] });
			updateParams(mag.values(), mag.grad(), buf[0], buf[1], bc1, bc2, lrA, false);
			touchedDora = true;
		}
		if (touchedDora)
			adapters.invalidateDoraCaches();
	}

	private void updateParams(float[] param, float[] grad, float[] m, float[] v, double bc1, double bc2, double lr,
			boolean applyWeightDecay) {
		double lrCorrected = lr * Math.sqrt(bc2) / bc1;
		for (int i = 0; i < param.length; i++) {
			double g = grad[i];
			m[i] = (float) (beta1 * m[i] + (1 - beta1) * g);
			v[i] = (float) (beta2 * v[i] + (1 - beta2) * g * g);
			double p = param[i];
			if (applyWeightDecay && weightDecay != 0.0)
				p -= lr * weightDecay * p;
			param[i] = (float) (p - lrCorrected * m[i] / (Math.sqrt(v[i]) + eps));
		}
	}

	public int step() {
		return t;
	}

	public double baseLearningRate() {
		return baseLr;
	}

	public double weightDecay() {
		return weightDecay;
	}

	public double loraPlusRatio() {
		return loraPlusRatio;
	}

	/** Last applied A learning rate, or NaN before the first step. */
	public double lastLearningRateA() {
		return lastLrA;
	}

	/** Last applied B learning rate, or NaN before the first step. */
	public double lastLearningRateB() {
		return lastLrB;
	}

	public void reset() {
		t = 0;
		lastLrA = Double.NaN;
		lastLrB = Double.NaN;
		state.clear();
		qaState.clear();
		magState.clear();
	}

	private static void validateBaseLr(double lr) {
		if (!Double.isFinite(lr) || lr <= 0)
			throw new IllegalArgumentException("lr must be finite and > 0");
	}

	private static void validateStepLr(double lr) {
		if (!Double.isFinite(lr) || lr < 0)
			throw new IllegalArgumentException("step learningRate must be finite and >= 0");
	}
}
