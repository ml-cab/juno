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
package cab.ml.juno.sampler;

/**
 * Step 1: Temperature scaling.
 *
 * Divides each logit by the temperature value. temperature → 0.0 : approaches
 * one-hot (fully deterministic) temperature = 1.0 : no change temperature → 2.0
 * : flattens distribution (more random)
 *
 * Skipped if greedy=true (temperature is irrelevant for argmax).
 */
public final class TemperatureStep implements SamplingStep {

	public static final TemperatureStep INSTANCE = new TemperatureStep();

	private TemperatureStep() {
	}

	@Override
	public float[] apply(float[] logits, SamplingParams params, int[] generatedTokens) {
		if (params.greedy())
			return logits;

		float temperature = params.temperature();
		// Treat near-zero as greedy to avoid division instability
		if (temperature < 1e-6f)
			return logits;

		for (int i = 0; i < logits.length; i++) {
			logits[i] /= temperature;
		}
		return logits;
	}
}
