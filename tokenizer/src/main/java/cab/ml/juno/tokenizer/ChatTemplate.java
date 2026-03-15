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

import java.util.List;
import java.util.Map;

/**
 * Formats a list of ChatMessages into a single prompt string in the format
 * expected by a specific model family.
 *
 * Each model family has different special tokens and structure. Templates are
 * stateless — one instance shared across all requests.
 *
 * Supported families: llama3
 * <|begin_of_text|><|start_header_id|>...<|end_header_id|>\n\n...<|eot_id|>
 * mistral [INST] ... [/INST] gemma
 * <start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n chatml
 * <|im_start|>role\n...<|im_end|>\n (default fallback for unknown models)
 */
public interface ChatTemplate {

	/**
	 * Format a list of messages into a model-ready prompt string. The returned
	 * string is passed directly to Tokenizer.encode().
	 */
	String format(List<ChatMessage> messages);

	/** Model family key this template applies to. */
	String modelType();

	// ── Built-in implementations ─────────────────────────────────────────────

	/**
	 * LLaMA 3 chat template. <|begin_of_text|>
	 * <|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>
	 * <|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>
	 * <|start_header_id|>assistant<|end_header_id|>\n\n
	 */
	static ChatTemplate llama3() {
		return new ChatTemplate() {
			@Override
			public String format(List<ChatMessage> messages) {
				StringBuilder sb = new StringBuilder();
				sb.append("<|begin_of_text|>");
				for (ChatMessage msg : messages) {
					sb.append("<|start_header_id|>").append(msg.role()).append("<|end_header_id|>\n\n")
							.append(msg.content()).append("<|eot_id|>");
				}
				sb.append("<|start_header_id|>assistant<|end_header_id|>\n\n");
				return sb.toString();
			}

			@Override
			public String modelType() {
				return "llama3";
			}
		};
	}

	/**
	 * Mistral chat template. [INST] {user} [/INST] {assistant}</s>[INST] ... System
	 * message prepended to first user turn.
	 */
	static ChatTemplate mistral() {
		return new ChatTemplate() {
			@Override
			public String format(List<ChatMessage> messages) {
				StringBuilder sb = new StringBuilder();
				String pendingSystem = null;
				for (ChatMessage msg : messages) {
					switch (msg.role()) {
					case "system" -> pendingSystem = msg.content();
					case "user" -> {
						sb.append("[INST] ");
						if (pendingSystem != null) {
							sb.append(pendingSystem).append("\n\n");
							pendingSystem = null;
						}
						sb.append(msg.content()).append(" [/INST]");
					}
					case "assistant" -> sb.append(" ").append(msg.content()).append("</s>");
					}
				}
				return sb.toString();
			}

			@Override
			public String modelType() {
				return "mistral";
			}
		};
	}

	/**
	 * Gemma chat template. <start_of_turn>user\n{user}<end_of_turn>\n
	 * <start_of_turn>model\n{assistant}<end_of_turn>\n <start_of_turn>model\n
	 */
	static ChatTemplate gemma() {
		return new ChatTemplate() {
			@Override
			public String format(List<ChatMessage> messages) {
				StringBuilder sb = new StringBuilder();
				for (ChatMessage msg : messages) {
					String role = msg.isAssistant() ? "model" : msg.role();
					sb.append("<start_of_turn>").append(role).append("\n").append(msg.content())
							.append("<end_of_turn>\n");
				}
				sb.append("<start_of_turn>model\n");
				return sb.toString();
			}

			@Override
			public String modelType() {
				return "gemma";
			}
		};
	}

	/**
	 * TinyLlama / Zephyr chat template.
	 * <|system|>\n{system}</s>\n<|user|>\n{user}</s>\n<|assistant|>\n
	 *
	 * TinyLlama-1.1B-Chat-v1.0 is fine-tuned with this exact format. Using any
	 * other template (e.g. ChatML) sends tokens the model has never seen and
	 * produces complete garbage output.
	 */
	static ChatTemplate tinyllama() {
		return new ChatTemplate() {
			@Override
			public String format(List<ChatMessage> messages) {
				StringBuilder sb = new StringBuilder();
				for (ChatMessage msg : messages) {
					String tag = switch (msg.role()) {
					case "assistant" -> "<|assistant|>";
					case "system" -> "<|system|>";
					default -> "<|user|>";
					};
					sb.append(tag).append("\n").append(msg.content()).append("</s>\n");
				}
				sb.append("<|assistant|>\n");
				return sb.toString();
			}

			@Override
			public String modelType() {
				return "tinyllama";
			}
		};
	}

	/**
	 * ChatML template — used by Qwen, OpenHermes, and as default fallback.
	 * <|im_start|>role\n{content}<|im_end|>\n
	 */
	static ChatTemplate chatml() {
		return new ChatTemplate() {
			@Override
			public String format(List<ChatMessage> messages) {
				StringBuilder sb = new StringBuilder();
				for (ChatMessage msg : messages) {
					sb.append("<|im_start|>").append(msg.role()).append("\n").append(msg.content())
							.append("<|im_end|>\n");
				}
				sb.append("<|im_start|>assistant\n");
				return sb.toString();
			}

			@Override
			public String modelType() {
				return "chatml";
			}
		};
	}

	// ── Registry ─────────────────────────────────────────────────────────────

	/** All built-in templates keyed by modelType. */
	Map<String, ChatTemplate> BUILT_IN = Map.of("llama3", llama3(), "mistral", mistral(), "gemma", gemma(), "chatml",
			chatml(), "tinyllama", tinyllama(), "zephyr", tinyllama() // same format, alternate lookup key
	);

	/**
	 * Resolve a template by model type string. Falls back to ChatML for unknown
	 * model types.
	 */
	static ChatTemplate forModelType(String modelType) {
		if (modelType == null)
			return chatml();
		return BUILT_IN.getOrDefault(modelType.toLowerCase().strip(), chatml());
	}
}