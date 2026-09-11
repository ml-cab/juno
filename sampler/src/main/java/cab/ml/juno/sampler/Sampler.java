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

import java.util.Random;

/**
 * Main sampling pipeline.
 *
 * Chains all steps in order: presencePenalty → repetitionPenalty → temperature
 * → topK → softmax → topP → sample
 *
 * Stateless and thread-safe when no seeded {@link Random} is supplied. One
 * instance may be shared across requests; seeded RNGs stay request-local.
 *
 * Usage: Sampler sampler = Sampler.create(); int nextToken =
 * sampler.sample(logits, SamplingParams.defaults(), generatedSoFar);
 */
public final class Sampler {

	private final TemperatureStep temperatureStep;
	private final TopKStep topKStep;
	private final SoftmaxStep softmaxStep;
	private final TopPStep topPStep;
	private final RepetitionPenaltyStep repetitionPenaltyStep;
	private final PresencePenaltyStep presencePenaltyStep;
	private final SampleStep sampleStep;

	private Sampler(TemperatureStep temperatureStep, TopKStep topKStep, SoftmaxStep softmaxStep, TopPStep topPStep,
			RepetitionPenaltyStep repetitionPenaltyStep, PresencePenaltyStep presencePenaltyStep,
			SampleStep sampleStep) {
		this.temperatureStep = temperatureStep;
		this.topKStep = topKStep;
		this.softmaxStep = softmaxStep;
		this.topPStep = topPStep;
		this.repetitionPenaltyStep = repetitionPenaltyStep;
		this.presencePenaltyStep = presencePenaltyStep;
		this.sampleStep = sampleStep;
	}

	/**
	 * Create a default Sampler with all standard pipeline steps.
	 */
	public static Sampler create() {
		return new Sampler(TemperatureStep.INSTANCE, TopKStep.INSTANCE, SoftmaxStep.INSTANCE, TopPStep.INSTANCE,
				RepetitionPenaltyStep.INSTANCE, PresencePenaltyStep.INSTANCE, SampleStep.INSTANCE);
	}

	/**
	 * Run the full sampling pipeline and return the next token ID.
	 *
	 * @param rawLogits       float[vocabSize] from the last inference node — WILL
	 *                        BE MUTATED
	 * @param params          sampling configuration for this request
	 * @param generatedTokens token IDs generated so far in this sequence (for
	 *                        repetition / presence penalty)
	 * @return next token ID
	 */
	public int sample(float[] rawLogits, SamplingParams params, int[] generatedTokens) {
		return sample(rawLogits, params, generatedTokens, null);
	}

	/**
	 * @param rng optional seeded RNG for stochastic sampling; {@code null} uses
	 *            thread-local randomness
	 */
	public int sample(float[] rawLogits, SamplingParams params, int[] generatedTokens, Random rng) {
		if (rawLogits == null || rawLogits.length == 0)
			throw new IllegalArgumentException("logits must not be null or empty");
		if (params == null)
			throw new IllegalArgumentException("params must not be null");

		float[] logits = rawLogits.clone();

		logits = presencePenaltyStep.apply(logits, params, generatedTokens);
		logits = repetitionPenaltyStep.apply(logits, params, generatedTokens);
		logits = temperatureStep.apply(logits, params, generatedTokens);
		logits = topKStep.apply(logits, params, generatedTokens);
		logits = softmaxStep.apply(logits, params, generatedTokens);
		logits = topPStep.apply(logits, params, generatedTokens);

		return sampleStep.sample(logits, params, rng);
	}

	/**
	 * Convenience overload — no previous tokens (start of sequence).
	 */
	public int sample(float[] rawLogits, SamplingParams params) {
		return sample(rawLogits, params, new int[0], null);
	}

	/**
	 * Check whether a token is a stop condition.
	 */
	public boolean isStopToken(int tokenId, SamplingParams params) {
		for (int stop : params.stopTokenIds()) {
			if (stop == tokenId)
				return true;
		}
		return false;
	}
}
