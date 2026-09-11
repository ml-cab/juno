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
package cab.ml.juno.coordinator;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import com.fasterxml.jackson.databind.JsonNode;

import cab.ml.juno.tokenizer.Tokenizer;

/**
 * Pure mapping helpers between Juno internals and the OpenAI wire format.
 */
public final class OpenAiAdapter {

	private OpenAiAdapter() {
	}

	/**
	 * Maps OpenAI {@code frequency_penalty} (−2..2) to Juno
	 * {@code repetitionPenalty} (≥ 1).
	 */
	public static float repetitionPenaltyFromFrequencyPenalty(float frequencyPenalty) {
		float positive = Math.max(0f, frequencyPenalty / 2.0f);
		return 1.0f + positive;
	}

	/**
	 * @return {@code null} if {@code n} is absent or 1; otherwise an error message
	 */
	public static String validateCompletionsN(Integer n) {
		if (n == null || n == 1)
			return null;
		return "n must be 1 for this endpoint (got " + n + ")";
	}

	/**
	 * Rejects structured {@code response_format} until grammar support ships.
	 * Absent or {@code type=text} is accepted.
	 *
	 * @return error message, or {@code null} when allowed
	 */
	public static String validateResponseFormat(JsonNode responseFormat) {
		if (responseFormat == null || responseFormat.isNull() || responseFormat.isMissingNode())
			return null;
		if (!responseFormat.isObject())
			return "response_format must be an object";
		JsonNode type = responseFormat.get("type");
		if (type == null || !type.isTextual())
			return "response_format.type is required";
		String t = type.asText();
		if ("text".equals(t))
			return null;
		return "response_format type '" + t + "' is not supported yet (only type=text, or omit the field)";
	}

	/**
	 * Parse OpenAI {@code stop} (string or array of strings, max 4).
	 *
	 * @return stop strings, or empty when absent
	 * @throws IllegalArgumentException when the shape is invalid
	 */
	public static String[] parseStop(JsonNode stop) {
		if (stop == null || stop.isNull() || stop.isMissingNode())
			return new String[0];
		if (stop.isTextual()) {
			String s = stop.asText();
			return s.isBlank() ? new String[0] : new String[] { s };
		}
		if (!stop.isArray())
			throw new IllegalArgumentException("stop must be a string or array of strings");
		if (stop.size() > 4)
			throw new IllegalArgumentException("stop array may contain at most 4 strings");
		List<String> out = new ArrayList<>(stop.size());
		for (JsonNode n : stop) {
			if (n == null || !n.isTextual())
				throw new IllegalArgumentException("stop array entries must be strings");
			String s = n.asText();
			if (!s.isBlank())
				out.add(s);
		}
		return out.toArray(String[]::new);
	}

	/**
	 * Tokenize stop strings; single-token encodings become stop token ids for
	 * early halt. Multi-token stops rely on decoded-text matching.
	 */
	public static int[] stopTokenIdsFromStrings(Tokenizer tokenizer, String[] stopStrings, int[] existing) {
		Set<Integer> ids = new LinkedHashSet<>();
		if (existing != null) {
			for (int id : existing)
				ids.add(id);
		}
		if (tokenizer != null && stopStrings != null) {
			for (String s : stopStrings) {
				if (s == null || s.isBlank())
					continue;
				int[] encoded = tokenizer.encode(s);
				if (encoded != null && encoded.length == 1)
					ids.add(encoded[0]);
			}
		}
		return ids.stream().mapToInt(Integer::intValue).toArray();
	}

	public static String toOpenAiFinishReason(GenerationResult.StopReason reason) {
		return switch (reason) {
		case EOS_TOKEN, STOP_TOKEN -> "stop";
		case MAX_TOKENS -> "length";
		case ERROR -> "error";
		};
	}

	/**
	 * OpenAI-style completion id: {@code chatcmpl-} + request UUID without hyphens.
	 */
	public static String chatCompletionId(String requestId) {
		String compact = requestId.replace("-", "");
		return "chatcmpl-" + compact;
	}
}
