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

package cab.ml.juno.player;

/**
 * Chat-template formatting for supervised LoRA training text (must match
 * {@link cab.ml.juno.tokenizer.ChatTemplateFormatter} at inference).
 *
 * <p>
 * Splits each turn into {@linkplain #qaPrefix prefix} (user + assistant header)
 * and {@linkplain #qaCompletion completion} (answer + end) so training can apply
 * completion-only loss masks.
 */
public final class ChatTrainingFormats {

	private ChatTrainingFormats() {
	}

	/** User turn through the assistant header (no answer tokens). */
	static String qaPrefix(String question, String modelType) {
		return switch (modelType) {
		case "tinyllama", "zephyr" -> "<|user|>\n" + question + "</s>\n<|assistant|>\n";
		case "phi3", "phi-3" -> "<|user|>\n" + question + "<|end|>\n<|assistant|>\n";
		case "llama3" -> "<|start_header_id|>user<|end_header_id|>\n\n" + question
				+ "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
		case "mistral" -> "[INST] " + question + " [/INST] ";
		case "gemma" -> "<start_of_turn>user\n" + question + "<end_of_turn>\n" + "<start_of_turn>model\n";
		// Qwen3 enable_thinking=false: empty closed <think> before assistant text.
		case "qwen3" -> "<|im_start|>user\n" + question + "<|im_end|>\n" + "<|im_start|>assistant\n"
				+ "<think>\n\n</think>\n\n";
		default -> "<|im_start|>user\n" + question + "<|im_end|>\n" + "<|im_start|>assistant\n";
		};
	}

	/** Answer body plus the turn-terminating special token(s). */
	static String qaCompletion(String answer, String modelType) {
		return switch (modelType) {
		case "tinyllama", "zephyr" -> answer + "</s>\n";
		case "phi3", "phi-3" -> answer + "<|end|>\n";
		case "llama3" -> answer + "<|eot_id|>";
		case "mistral" -> answer + "</s>";
		case "gemma" -> answer + "<end_of_turn>\n";
		case "qwen3" -> answer + "<|im_end|>\n";
		default -> answer + "<|im_end|>\n";
		};
	}

	static String qaTurn(String question, String answer, String modelType) {
		return qaPrefix(question, modelType) + qaCompletion(answer, modelType);
	}

	/**
	 * Four complete Q&amp;A phrasings used by {@code /train-qa}. Hold out whole
	 * variants for validation rather than splitting inside a turn.
	 */
	public static String[] qaQuestionVariants(String question) {
		String q = question.endsWith("?") ? question : question + "?";
		String qLow = q.substring(0, 1).toLowerCase() + q.substring(1);
		return new String[] { q, qLow, "Can you tell me: " + qLow, "Please answer: " + qLow };
	}
}
