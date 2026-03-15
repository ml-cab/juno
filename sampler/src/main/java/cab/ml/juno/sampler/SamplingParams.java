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
 * Immutable sampling configuration for a single inference request. Use the
 * static factory methods for preset profiles.
 */
public record SamplingParams(float temperature, int topK, float topP, float repetitionPenalty, boolean greedy,
		int maxTokens, int[] stopTokenIds) {

	// ── Validation ────────────────────────────────────────────────────────────

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
		if (maxTokens < 1)
			throw new IllegalArgumentException("maxTokens must be >= 1, got: " + maxTokens);
		stopTokenIds = stopTokenIds != null ? stopTokenIds.clone() : new int[0];
	}

	// ── Preset factory methods ────────────────────────────────────────────────

	/**
	 * Balanced defaults — suitable for general chat. temperature=0.7, topK=50,
	 * topP=0.9, penalty=1.1
	 */
	public static SamplingParams defaults() {
		return new SamplingParams(0.7f, 50, 0.9f, 1.1f, false, 512, new int[0]);
	}

	/**
	 * Deterministic — for code generation and factual Q&A. temperature=0.1,
	 * greedy=true
	 */
	public static SamplingParams deterministic() {
		return new SamplingParams(0.1f, 1, 1.0f, 1.0f, true, 512, new int[0]);
	}

	/**
	 * Creative — for storytelling and open-ended generation. temperature=1.2,
	 * topK=100, topP=0.95
	 */
	public static SamplingParams creative() {
		return new SamplingParams(1.2f, 100, 0.95f, 1.1f, false, 512, new int[0]);
	}

	// ── Fluent builders ───────────────────────────────────────────────────────

	public SamplingParams withTemperature(float temperature) {
		return new SamplingParams(temperature, topK, topP, repetitionPenalty, greedy, maxTokens, stopTokenIds);
	}

	public SamplingParams withTopK(int topK) {
		return new SamplingParams(temperature, topK, topP, repetitionPenalty, greedy, maxTokens, stopTokenIds);
	}

	public SamplingParams withTopP(float topP) {
		return new SamplingParams(temperature, topK, topP, repetitionPenalty, greedy, maxTokens, stopTokenIds);
	}

	public SamplingParams withRepetitionPenalty(float repetitionPenalty) {
		return new SamplingParams(temperature, topK, topP, repetitionPenalty, greedy, maxTokens, stopTokenIds);
	}

	public SamplingParams withGreedy(boolean greedy) {
		return new SamplingParams(temperature, topK, topP, repetitionPenalty, greedy, maxTokens, stopTokenIds);
	}

	public SamplingParams withMaxTokens(int maxTokens) {
		return new SamplingParams(temperature, topK, topP, repetitionPenalty, greedy, maxTokens, stopTokenIds);
	}

	public SamplingParams withStopTokenIds(int... stopTokenIds) {
		return new SamplingParams(temperature, topK, topP, repetitionPenalty, greedy, maxTokens, stopTokenIds);
	}

	// ── Accessors (defensive copy for array) ─────────────────────────────────

	@Override
	public int[] stopTokenIds() {
		return stopTokenIds.clone();
	}
}
