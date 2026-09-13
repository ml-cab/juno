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
package cab.ml.juno.tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Injects OpenAI-style tool definitions into chat messages for templates that
 * emit {@code role} verbatim (Llama 3, ChatML, Qwen3). Unsupported templates
 * fail closed.
 */
public final class ToolPrompt {

	public static final Set<String> SUPPORTED_MODEL_TYPES = Set.of("llama3", "chatml", "qwen3");

	private ToolPrompt() {
	}

	public static String resolvedModelType(String modelId) {
		return ChatTemplate.forModelType(modelId).modelType();
	}

	public static boolean supported(String modelId) {
		return SUPPORTED_MODEL_TYPES.contains(resolvedModelType(modelId));
	}

	public static void requireSupported(String modelId) {
		String type = resolvedModelType(modelId);
		if (!SUPPORTED_MODEL_TYPES.contains(type))
			throw new IllegalArgumentException("tools are not supported for chat template '" + type
					+ "' (supported: llama3, chatml, qwen3)");
	}

	/**
	 * Merge tool instructions into the first system message, or prepend a system
	 * turn when none exists.
	 */
	public static List<ChatMessage> inject(List<ChatMessage> messages, String toolsJson, boolean required,
			String forcedFunction) {
		if (messages == null || messages.isEmpty())
			throw new IllegalArgumentException("messages must not be null or empty");
		if (toolsJson == null || toolsJson.isBlank())
			throw new IllegalArgumentException("tools JSON must not be blank");
		String block = instructions(toolsJson, required, forcedFunction);
		List<ChatMessage> out = new ArrayList<>(messages.size() + 1);
		boolean merged = false;
		for (ChatMessage m : messages) {
			if (!merged && m.isSystem()) {
				out.add(ChatMessage.system(m.content() + "\n\n" + block));
				merged = true;
			} else {
				out.add(m);
			}
		}
		if (!merged)
			out.add(0, ChatMessage.system(block));
		return List.copyOf(out);
	}

	public static String formatAssistantToolCalls(List<String> callJsonObjects) {
		if (callJsonObjects == null || callJsonObjects.isEmpty())
			return "";
		StringBuilder sb = new StringBuilder();
		for (String json : callJsonObjects) {
			if (json == null || json.isBlank())
				continue;
			if (!sb.isEmpty())
				sb.append('\n');
			sb.append("<tool_call>\n").append(json.strip()).append("\n</tool_call>");
		}
		return sb.toString();
	}

	public static String formatToolResult(String content) {
		String body = content == null ? "" : content;
		return "<tool_response>\n" + body + "\n</tool_response>";
	}

	static String instructions(String toolsJson, boolean required, String forcedFunction) {
		StringBuilder sb = new StringBuilder();
		sb.append("You can use tools by emitting one or more blocks of the form:\n");
		sb.append("<tool_call>\n{\"name\": \"<function-name>\", \"arguments\": {}}\n</tool_call>\n");
		sb.append("Do not wrap the blocks in markdown. Available tools:\n<tools>\n");
		sb.append(toolsJson.strip()).append("\n</tools>\n");
		if (forcedFunction != null && !forcedFunction.isBlank()) {
			sb.append("You must call the function \"").append(forcedFunction.strip())
					.append("\". Emit only a tool_call block, with no other text.");
		} else if (required) {
			sb.append("You must call a tool. Emit only tool_call blocks, with no other text.");
		} else {
			sb.append("Call a tool only when it helps. Otherwise answer the user directly.");
		}
		return sb.toString();
	}
}
