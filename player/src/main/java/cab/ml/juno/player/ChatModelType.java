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
 * Derives the chat template model type from a GGUF file path so the correct
 * template is used (e.g. TinyLlama needs tinyllama, not chatml).
 */
public final class ChatModelType {

	private ChatModelType() {}

	/**
	 * Returns the model type key for ChatTemplate lookup. Path is compared
	 * case-insensitively; unknown paths return "chatml".
	 *
	 * @param path path to the GGUF file (may be null)
	 * @return "tinyllama", "llama3", "mistral", "gemma", or "chatml"
	 */
	public static String fromPath(String path) {
		if (path == null)
			return "chatml";
		String lower = path.toLowerCase();
		if (lower.contains("tinyllama") || lower.contains("zephyr"))
			return "tinyllama";
		if (lower.contains("llama-3") || lower.contains("llama3"))
			return "llama3";
		if (lower.contains("mistral"))
			return "mistral";
		if (lower.contains("gemma"))
			return "gemma";
		return "chatml";
	}
}
