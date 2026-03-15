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

/**
 * A single turn in a chat conversation. Role is one of: "system", "user",
 * "assistant".
 */
public record ChatMessage(String role, String content) {

	public ChatMessage {
		if (role == null || role.isBlank())
			throw new IllegalArgumentException("role must not be blank");
		if (content == null)
			throw new IllegalArgumentException("content must not be null");
		role = role.strip().toLowerCase();
	}

	public static ChatMessage system(String content) {
		return new ChatMessage("system", content);
	}

	public static ChatMessage user(String content) {
		return new ChatMessage("user", content);
	}

	public static ChatMessage assistant(String content) {
		return new ChatMessage("assistant", content);
	}

	public boolean isSystem() {
		return "system".equals(role);
	}

	public boolean isUser() {
		return "user".equals(role);
	}

	public boolean isAssistant() {
		return "assistant".equals(role);
	}
}
