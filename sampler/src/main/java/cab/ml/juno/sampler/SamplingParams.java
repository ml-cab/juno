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

import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable sampling configuration for a single inference request. Use the
 * static factory methods for preset profiles.
 */
public record SamplingParams(float temperature, int topK, float topP, float repetitionPenalty, float presencePenalty,
		boolean greedy, int maxTokens, int[] stopTokenIds, String[] stopStrings, Long seed, GbnfGrammar grammar) {

	public SamplingParams {
		if (temperature < 0.0f || temperature > 2.0f)
			throw new IllegalArgumentException("temperature must be 0.0-2.0, got: " + temperature);
		if (topK < 0)
			throw new IllegalArgumentException("topK must be >= 0 (0 = disabled), got: " + topK);
		if (topP < 0.0f || topP > 1.0f)
			throw new IllegalArgumentException("topP must be 0.0-1.0, got: " + topP);
		if (repetitionPenalty < 1.0f)
			throw new IllegalArgumentException(
					"repetitionPenalty must be >= 1.0 (1.0 = disabled), got: " + repetitionPenalty);
		if (presencePenalty < -2.0f || presencePenalty > 2.0f)
			throw new IllegalArgumentException("presencePenalty must be -2.0..2.0, got: " + presencePenalty);
		if (maxTokens < 1)
			throw new IllegalArgumentException("maxTokens must be >= 1, got: " + maxTokens);
		stopTokenIds = stopTokenIds != null ? stopTokenIds.clone() : new int[0];
		stopStrings = normalizeStopStrings(stopStrings);
	}

	private static String[] normalizeStopStrings(String[] stopStrings) {
		if (stopStrings == null || stopStrings.length == 0)
			return new String[0];
		return Arrays.stream(stopStrings).filter(Objects::nonNull).map(String::strip).filter(s -> !s.isEmpty())
				.toArray(String[]::new);
	}

	/**
	 * Balanced defaults — suitable for general chat. temperature=0.7, topK=50,
	 * topP=0.9, penalty=1.1, maxTokens=200
	 */
	public static SamplingParams defaults() {
		return new SamplingParams(0.7f, 50, 0.9f, 1.1f, 0.0f, false, 200, new int[0], new String[0], null, null);
	}

	/**
	 * Deterministic — for code generation and factual Q&A. temperature=0.1,
	 * greedy=true
	 */
	public static SamplingParams deterministic() {
		return new SamplingParams(0.1f, 1, 1.0f, 1.0f, 0.0f, true, 512, new int[0], new String[0], null, null);
	}

	/**
	 * Creative — for storytelling and open-ended generation. temperature=1.2,
	 * topK=100, topP=0.95
	 */
	public static SamplingParams creative() {
		return new SamplingParams(1.2f, 100, 0.95f, 1.1f, 0.0f, false, 512, new int[0], new String[0], null, null);
	}

	public SamplingParams withTemperature(float temperature) {
		return new SamplingParams(temperature, topK, topP, repetitionPenalty, presencePenalty, greedy, maxTokens,
				stopTokenIds, stopStrings, seed, grammar);
	}

	public SamplingParams withTopK(int topK) {
		return new SamplingParams(temperature, topK, topP, repetitionPenalty, presencePenalty, greedy, maxTokens,
				stopTokenIds, stopStrings, seed, grammar);
	}

	public SamplingParams withTopP(float topP) {
		return new SamplingParams(temperature, topK, topP, repetitionPenalty, presencePenalty, greedy, maxTokens,
				stopTokenIds, stopStrings, seed, grammar);
	}

	public SamplingParams withRepetitionPenalty(float repetitionPenalty) {
		return new SamplingParams(temperature, topK, topP, repetitionPenalty, presencePenalty, greedy, maxTokens,
				stopTokenIds, stopStrings, seed, grammar);
	}

	public SamplingParams withPresencePenalty(float presencePenalty) {
		return new SamplingParams(temperature, topK, topP, repetitionPenalty, presencePenalty, greedy, maxTokens,
				stopTokenIds, stopStrings, seed, grammar);
	}

	public SamplingParams withGreedy(boolean greedy) {
		return new SamplingParams(temperature, topK, topP, repetitionPenalty, presencePenalty, greedy, maxTokens,
				stopTokenIds, stopStrings, seed, grammar);
	}

	public SamplingParams withMaxTokens(int maxTokens) {
		return new SamplingParams(temperature, topK, topP, repetitionPenalty, presencePenalty, greedy, maxTokens,
				stopTokenIds, stopStrings, seed, grammar);
	}

	public SamplingParams withStopTokenIds(int... stopTokenIds) {
		return new SamplingParams(temperature, topK, topP, repetitionPenalty, presencePenalty, greedy, maxTokens,
				stopTokenIds, stopStrings, seed, grammar);
	}

	public SamplingParams withStopStrings(String... stopStrings) {
		return new SamplingParams(temperature, topK, topP, repetitionPenalty, presencePenalty, greedy, maxTokens,
				stopTokenIds, stopStrings, seed, grammar);
	}

	public SamplingParams withSeed(Long seed) {
		return new SamplingParams(temperature, topK, topP, repetitionPenalty, presencePenalty, greedy, maxTokens,
				stopTokenIds, stopStrings, seed, grammar);
	}

	public SamplingParams withGrammar(GbnfGrammar grammar) {
		return new SamplingParams(temperature, topK, topP, repetitionPenalty, presencePenalty, greedy, maxTokens,
				stopTokenIds, stopStrings, seed, grammar);
	}

	@Override
	public int[] stopTokenIds() {
		return stopTokenIds.clone();
	}

	@Override
	public String[] stopStrings() {
		return stopStrings.clone();
	}
}
