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
 */
final class ChatTrainingFormats {

	private ChatTrainingFormats() {
	}

	static String qaTurn(String question, String answer, String modelType) {
		return switch (modelType) {
		case "tinyllama", "zephyr" -> "<|user|>\n" + question + "</s>\n<|assistant|>\n" + answer + "</s>\n";
		case "phi3", "phi-3" -> "<|user|>\n" + question + "<|end|>\n<|assistant|>\n" + answer + "<|end|>\n";
		case "llama3" -> "<|start_header_id|>user<|end_header_id|>\n\n" + question
				+ "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" + answer + "<|eot_id|>";
		case "mistral" -> "[INST] " + question + " [/INST] " + answer + "</s>";
		case "gemma" -> "<start_of_turn>user\n" + question + "<end_of_turn>\n" + "<start_of_turn>model\n" + answer
				+ "<end_of_turn>\n";
		default -> "<|im_start|>user\n" + question + "<|im_end|>\n" + "<|im_start|>assistant\n" + answer
				+ "<|im_end|>\n";
		};
	}
}
