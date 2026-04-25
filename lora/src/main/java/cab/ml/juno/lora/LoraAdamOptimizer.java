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
 * Adam optimiser for {@link LoraAdapterSet} parameters.
 */
public final class LoraAdamOptimizer {

	private final double lr;
	private final double beta1;
	private final double beta2;
	private final double eps;
	private final double weightDecay;

	private int t = 0;

	private final Map<LoraAdapter, float[][]> state = new IdentityHashMap<>();

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

	public static LoraAdamOptimizer defaults(double lr) {
		return new LoraAdamOptimizer(lr, 0.9, 0.999, 1e-8, 0.01);
	}

	public void step(LoraAdapterSet adapters) {
		t++;
		double bc1 = 1.0 - Math.pow(beta1, t);
		double bc2 = 1.0 - Math.pow(beta2, t);

		for (LoraAdapter adapter : adapters.all()) {
			float[][] buf = state.computeIfAbsent(adapter, a -> new float[][] { new float[a.a.length],
					new float[a.a.length], new float[a.b.length], new float[a.b.length] });
			updateParams(adapter.a(), adapter.gradA(), buf[0], buf[1], bc1, bc2, true);
			updateParams(adapter.b(), adapter.gradB(), buf[2], buf[3], bc1, bc2, false);
		}
	}

	private void updateParams(float[] param, float[] grad, float[] m, float[] v, double bc1, double bc2,
			boolean applyWeightDecay) {
		double lrCorrected = lr * Math.sqrt(bc2) / bc1;
		for (int i = 0; i < param.length; i++) {
			double g = grad[i];
			if (applyWeightDecay)
				g += weightDecay * param[i];
			m[i] = (float) (beta1 * m[i] + (1 - beta1) * g);
			v[i] = (float) (beta2 * v[i] + (1 - beta2) * g * g);
			param[i] -= (float) (lrCorrected * m[i] / (Math.sqrt(v[i]) + eps));
		}
	}

	public int step() {
		return t;
	}

	public void reset() {
		t = 0;
		state.clear();
	}
}
