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
 * Result of a single forward/backward gradient computation over one token
 * sequence (no optimizer step).
 *
 * @param lossSum          sum of per-prediction cross-entropy (nats), not mean
 * @param predictionCount  number of prediction positions ({@code tokens.length - 1})
 * @param forwardMs        wall time of the forward pass
 * @param backwardMs       wall time of the backward pass
 * @param timing           Tier-4/9 subset timings (never null)
 */
public record LoraGradientResult(float lossSum, int predictionCount, long forwardMs, long backwardMs,
		LoraStepTiming timing) {

	public LoraGradientResult {
		if (timing == null)
			timing = LoraStepTiming.zero();
	}

	/** Compatibility constructor: zero subset timings. */
	public LoraGradientResult(float lossSum, int predictionCount, long forwardMs, long backwardMs) {
		this(lossSum, predictionCount, forwardMs, backwardMs, LoraStepTiming.zero());
	}

	/** Token-weighted mean loss over the predictions; {@link Float#NaN} when empty. */
	public float meanLoss() {
		if (predictionCount == 0)
			return Float.NaN;
		return lossSum / predictionCount;
	}
}
