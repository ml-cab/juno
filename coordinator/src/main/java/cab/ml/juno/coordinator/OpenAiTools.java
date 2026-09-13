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
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import cab.ml.juno.sampler.GbnfGrammar;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.ToolPrompt;

/**
 * OpenAI {@code tools} / {@code tool_choice} parse, prompt bind, and response
 * mapping.
 */
public final class OpenAiTools {

	private static final ObjectMapper JSON = new ObjectMapper();
	private static final Set<String> ROLES = Set.of("system", "user", "assistant", "tool");

	private OpenAiTools() {
	}

	public record FunctionTool(String name, String description, JsonNode parameters) {
		public FunctionTool {
			if (name == null || name.isBlank())
				throw new IllegalArgumentException("tools[].function.name is required");
			name = name.strip();
			if (description == null)
				description = "";
		}
	}

	public record ToolChoice(Kind kind, String functionName) {
		public enum Kind {
			NONE, AUTO, REQUIRED, NAMED
		}

		public boolean allowsCalls() {
			return kind != Kind.NONE;
		}

		public boolean requiresCall() {
			return kind == Kind.REQUIRED || kind == Kind.NAMED;
		}
	}

	public record ParsedRequest(List<FunctionTool> tools, ToolChoice choice) {
		public ParsedRequest {
			tools = tools == null ? List.of() : List.copyOf(tools);
			if (choice == null)
				choice = new ToolChoice(tools.isEmpty() ? ToolChoice.Kind.NONE : ToolChoice.Kind.AUTO, null);
		}

		public boolean active() {
			return !tools.isEmpty() && choice.allowsCalls();
		}
	}

	public static ParsedRequest parse(JsonNode toolsNode, JsonNode toolChoiceNode) {
		List<FunctionTool> tools = parseTools(toolsNode);
		ToolChoice choice = parseChoice(toolChoiceNode, tools);
		if (choice.kind() == ToolChoice.Kind.NAMED && tools.stream().noneMatch(t -> t.name().equals(choice.functionName())))
			throw new IllegalArgumentException("tool_choice function '" + choice.functionName() + "' is not in tools");
		if (choice.requiresCall() && tools.isEmpty())
			throw new IllegalArgumentException("tools must be set when tool_choice is required or names a function");
		return new ParsedRequest(tools, choice);
	}

	public static ChatMessage toChatMessage(String role, JsonNode content, JsonNode toolCalls, String toolCallId) {
		if (role == null || role.isBlank())
			throw new IllegalArgumentException("each message needs a non-blank role");
		String r = role.strip().toLowerCase();
		if (!ROLES.contains(r))
			throw new IllegalArgumentException("unsupported message role '" + role + "' (system, user, assistant, tool)");
		if ("tool".equals(r)) {
			String text = requireText(content, "messages[].content");
			return ChatMessage.tool(ToolPrompt.formatToolResult(text));
		}
		if ("assistant".equals(r) && toolCalls != null && !toolCalls.isNull() && !toolCalls.isMissingNode()) {
			String replay = replayAssistant(toolCalls);
			if (replay.isBlank())
				throw new IllegalArgumentException("messages[].tool_calls must contain at least one function call");
			String extra = optionalText(content);
			if (extra != null && !extra.isBlank())
				replay = extra.strip() + "\n" + replay;
			return ChatMessage.assistant(replay);
		}
		if ("assistant".equals(r) || "system".equals(r) || "user".equals(r)) {
			String text = requireText(content, "messages[].content");
			return new ChatMessage(r, text);
		}
		throw new IllegalArgumentException("unsupported message role '" + role + "'");
	}

	public static List<ChatMessage> bindPrompt(List<ChatMessage> messages, ParsedRequest req, String modelId) {
		if (req == null || !req.active())
			return messages;
		ToolPrompt.requireSupported(modelId);
		return ToolPrompt.inject(messages, toolsJson(req.tools()), req.choice().requiresCall(),
				req.choice().functionName());
	}

	public static GbnfGrammar grammar(ParsedRequest req) {
		if (req == null || !req.active() || !req.choice().requiresCall())
			return null;
		return ToolCallGrammar.compile(req.tools(), req.choice().functionName());
	}

	public static void rejectGrammarConflict(ParsedRequest req, JsonNode responseFormat, String xJunoGrammar,
			boolean processGrammar) {
		if (req == null || !req.active())
			return;
		if (processGrammar)
			throw new IllegalArgumentException(
					"tools cannot be combined with --grammar-file / --json-schema-file");
		if (xJunoGrammar != null && !xJunoGrammar.isBlank())
			throw new IllegalArgumentException("tools cannot be combined with x_juno_grammar");
		String type = responseType(responseFormat);
		if (type != null && !"text".equals(type))
			throw new IllegalArgumentException("tools cannot be combined with response_format.type=" + type);
	}

	public static List<Map<String, Object>> toOpenAiToolCalls(String requestId,
			List<ToolCallParser.ParsedToolCall> calls) {
		String prefix = compactId(requestId);
		List<Map<String, Object>> out = new ArrayList<>(calls.size());
		for (int i = 0; i < calls.size(); i++) {
			ToolCallParser.ParsedToolCall call = calls.get(i);
			Map<String, Object> fn = new LinkedHashMap<>();
			fn.put("name", call.name());
			fn.put("arguments", call.argumentsJson());
			Map<String, Object> tc = new LinkedHashMap<>();
			tc.put("id", "call_" + prefix + i);
			tc.put("type", "function");
			tc.put("function", fn);
			out.add(tc);
		}
		return out;
	}

	public static Map<String, Object> assistantMessage(String text, List<Map<String, Object>> toolCalls) {
		Map<String, Object> message = new LinkedHashMap<>();
		message.put("role", "assistant");
		if (toolCalls != null && !toolCalls.isEmpty()) {
			message.put("content", null);
			message.put("tool_calls", toolCalls);
		} else {
			message.put("content", text == null ? "" : text);
		}
		return message;
	}

	public static String finishReason(GenerationResult.StopReason reason, boolean hasToolCalls) {
		if (hasToolCalls && reason != GenerationResult.StopReason.ERROR)
			return "tool_calls";
		return OpenAiAdapter.toOpenAiFinishReason(reason);
	}

	public static String toolsJson(List<FunctionTool> tools) {
		ArrayNode arr = JSON.createArrayNode();
		for (FunctionTool t : tools) {
			ObjectNode fn = JSON.createObjectNode();
			fn.put("name", t.name());
			if (t.description() != null && !t.description().isBlank())
				fn.put("description", t.description());
			if (t.parameters() != null && !t.parameters().isNull())
				fn.set("parameters", t.parameters());
			ObjectNode item = JSON.createObjectNode();
			item.put("type", "function");
			item.set("function", fn);
			arr.add(item);
		}
		return arr.toString();
	}

	private static List<FunctionTool> parseTools(JsonNode toolsNode) {
		if (toolsNode == null || toolsNode.isNull() || toolsNode.isMissingNode())
			return List.of();
		if (!toolsNode.isArray())
			throw new IllegalArgumentException("tools must be an array");
		List<FunctionTool> out = new ArrayList<>();
		Set<String> names = new LinkedHashSet<>();
		for (JsonNode item : toolsNode) {
			if (item == null || !item.isObject())
				throw new IllegalArgumentException("tools[] entries must be objects");
			JsonNode type = item.get("type");
			if (type != null && type.isTextual() && !"function".equals(type.asText()))
				throw new IllegalArgumentException("only tools[].type=function is supported");
			JsonNode fn = item.get("function");
			if (fn == null || !fn.isObject())
				throw new IllegalArgumentException("tools[].function is required");
			JsonNode name = fn.get("name");
			if (name == null || !name.isTextual() || name.asText().isBlank())
				throw new IllegalArgumentException("tools[].function.name is required");
			String n = name.asText().strip();
			if (!names.add(n))
				throw new IllegalArgumentException("duplicate tool name '" + n + "'");
			JsonNode desc = fn.get("description");
			String description = desc != null && desc.isTextual() ? desc.asText() : "";
			JsonNode params = fn.get("parameters");
			out.add(new FunctionTool(n, description, params));
		}
		return out;
	}

	private static ToolChoice parseChoice(JsonNode toolChoice, List<FunctionTool> tools) {
		if (toolChoice == null || toolChoice.isNull() || toolChoice.isMissingNode()) {
			return new ToolChoice(tools.isEmpty() ? ToolChoice.Kind.NONE : ToolChoice.Kind.AUTO, null);
		}
		if (toolChoice.isTextual()) {
			return switch (toolChoice.asText().strip().toLowerCase()) {
			case "none" -> new ToolChoice(ToolChoice.Kind.NONE, null);
			case "auto" -> new ToolChoice(ToolChoice.Kind.AUTO, null);
			case "required" -> new ToolChoice(ToolChoice.Kind.REQUIRED, null);
			default -> throw new IllegalArgumentException(
					"tool_choice must be none, auto, required, or a function object");
			};
		}
		if (!toolChoice.isObject())
			throw new IllegalArgumentException("tool_choice must be a string or object");
		JsonNode type = toolChoice.get("type");
		if (type == null || !type.isTextual() || !"function".equals(type.asText()))
			throw new IllegalArgumentException("tool_choice.type must be function");
		JsonNode fn = toolChoice.get("function");
		if (fn == null || !fn.isObject())
			throw new IllegalArgumentException("tool_choice.function is required");
		JsonNode name = fn.get("name");
		if (name == null || !name.isTextual() || name.asText().isBlank())
			throw new IllegalArgumentException("tool_choice.function.name is required");
		return new ToolChoice(ToolChoice.Kind.NAMED, name.asText().strip());
	}

	private static String replayAssistant(JsonNode toolCalls) {
		if (!toolCalls.isArray())
			throw new IllegalArgumentException("messages[].tool_calls must be an array");
		List<String> jsonObjects = new ArrayList<>();
		for (JsonNode tc : toolCalls) {
			if (tc == null || !tc.isObject())
				throw new IllegalArgumentException("messages[].tool_calls[] must be objects");
			JsonNode fn = tc.get("function");
			if (fn == null || !fn.isObject())
				throw new IllegalArgumentException("messages[].tool_calls[].function is required");
			JsonNode name = fn.get("name");
			if (name == null || !name.isTextual() || name.asText().isBlank())
				throw new IllegalArgumentException("messages[].tool_calls[].function.name is required");
			ObjectNode obj = JSON.createObjectNode();
			obj.put("name", name.asText());
			JsonNode args = fn.get("arguments");
			if (args == null || args.isNull() || args.isMissingNode())
				obj.set("arguments", JSON.createObjectNode());
			else if (args.isTextual()) {
				try {
					obj.set("arguments", JSON.readTree(args.asText()));
				} catch (Exception e) {
					throw new IllegalArgumentException("messages[].tool_calls[].function.arguments must be JSON");
				}
			} else if (args.isObject())
				obj.set("arguments", args);
			else
				throw new IllegalArgumentException("messages[].tool_calls[].function.arguments must be JSON");
			jsonObjects.add(obj.toString());
		}
		return ToolPrompt.formatAssistantToolCalls(jsonObjects);
	}

	private static String requireText(JsonNode content, String param) {
		String text = optionalText(content);
		if (text == null)
			throw new IllegalArgumentException("Only string text content is supported in " + param);
		return text;
	}

	private static String optionalText(JsonNode content) {
		if (content == null || content.isNull() || content.isMissingNode())
			return "";
		if (!content.isTextual())
			return null;
		return content.asText();
	}

	private static String responseType(JsonNode responseFormat) {
		if (responseFormat == null || responseFormat.isNull() || responseFormat.isMissingNode())
			return null;
		if (!responseFormat.isObject())
			return null;
		JsonNode type = responseFormat.get("type");
		return type != null && type.isTextual() ? type.asText() : null;
	}

	private static String compactId(String requestId) {
		if (requestId == null || requestId.isBlank())
			return "0";
		String compact = requestId.replace("-", "");
		return compact.length() <= 8 ? compact : compact.substring(0, 8);
	}
}
