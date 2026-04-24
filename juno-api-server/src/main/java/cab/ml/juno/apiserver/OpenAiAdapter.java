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

package cab.ml.juno.apiserver;

import cab.ml.juno.coordinator.GenerationResult;

/**
 * Pure mapping helpers between Juno internals and the OpenAI wire format.
 */
public final class OpenAiAdapter {

	private OpenAiAdapter() {
	}

	/**
	 * Maps OpenAI {@code frequency_penalty} (−2..2) to Juno {@code repetitionPenalty} (≥ 1).
	 * <p>
	 * Formula from {@code juno-api.yaml}: {@code 1.0 + max(0, frequency_penalty / 2.0)} — negative
	 * values clamp to 1.0 (disabled).
	 */
	public static float repetitionPenaltyFromFrequencyPenalty(float frequencyPenalty) {
		float positive = Math.max(0f, frequencyPenalty / 2.0f);
		return 1.0f + positive;
	}

	/**
	 * @return {@code null} if {@code n} is absent or 1; otherwise a human-readable error message
	 */
	public static String validateCompletionsN(Integer n) {
		if (n == null)
			return null;
		if (n == 1)
			return null;
		return "n must be 1 for this endpoint (got " + n + ")";
	}

	public static String toOpenAiFinishReason(GenerationResult.StopReason reason) {
		return switch (reason) {
		case EOS_TOKEN, STOP_TOKEN -> "stop";
		case MAX_TOKENS -> "length";
		case ERROR -> "error";
		};
	}

	/** OpenAI-style completion id: {@code chatcmpl-} + request UUID without hyphens. */
	public static String chatCompletionId(String requestId) {
		String compact = requestId.replace("-", "");
		return "chatcmpl-" + compact;
	}
}
