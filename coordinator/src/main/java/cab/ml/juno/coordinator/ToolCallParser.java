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
import java.util.List;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Parses model text into tool calls. Accepts {@code <tool_call>} blocks and a
 * single raw JSON object with {@code name} plus {@code arguments} or
 * {@code parameters}.
 */
public final class ToolCallParser {

	private static final ObjectMapper JSON = new ObjectMapper();
	private static final Pattern TOOL_CALL = Pattern.compile("<tool_call>\\s*([\\s\\S]*?)\\s*</tool_call>",
			Pattern.CASE_INSENSITIVE);

	private ToolCallParser() {
	}

	public record ParsedToolCall(String name, String argumentsJson) {
		public ParsedToolCall {
			if (name == null || name.isBlank())
				throw new IllegalArgumentException("tool name must not be blank");
			if (argumentsJson == null || argumentsJson.isBlank())
				argumentsJson = "{}";
		}
	}

	public static List<ParsedToolCall> parse(String text) {
		if (text == null || text.isBlank())
			return List.of();
		List<ParsedToolCall> tagged = new ArrayList<>();
		Matcher m = TOOL_CALL.matcher(text);
		while (m.find()) {
			parseObject(m.group(1)).ifPresent(tagged::add);
		}
		if (!tagged.isEmpty())
			return List.copyOf(tagged);
		return parseObject(stripFences(text.strip())).map(List::of).orElse(List.of());
	}

	private static Optional<ParsedToolCall> parseObject(String raw) {
		if (raw == null || raw.isBlank())
			return Optional.empty();
		JsonNode node;
		try {
			node = JSON.readTree(raw.strip());
		} catch (Exception e) {
			return Optional.empty();
		}
		if (node == null || !node.isObject())
			return Optional.empty();
		JsonNode name = node.get("name");
		if (name == null || !name.isTextual() || name.asText().isBlank())
			return Optional.empty();
		JsonNode args = node.get("arguments");
		if (args == null || args.isNull() || args.isMissingNode())
			args = node.get("parameters");
		String argumentsJson;
		try {
			if (args == null || args.isNull() || args.isMissingNode())
				argumentsJson = "{}";
			else if (args.isTextual()) {
				String inner = args.asText();
				JSON.readTree(inner);
				argumentsJson = inner;
			} else if (args.isObject())
				argumentsJson = JSON.writeValueAsString(args);
			else
				return Optional.empty();
		} catch (Exception e) {
			return Optional.empty();
		}
		return Optional.of(new ParsedToolCall(name.asText(), argumentsJson));
	}

	private static String stripFences(String text) {
		if (!text.startsWith("```"))
			return text;
		int nl = text.indexOf('\n');
		int end = text.lastIndexOf("```");
		if (nl < 0 || end <= nl)
			return text;
		return text.substring(nl + 1, end).strip();
	}
}
