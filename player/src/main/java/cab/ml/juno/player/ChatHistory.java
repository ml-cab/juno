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

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

import cab.ml.juno.tokenizer.ChatMessage;

/**
 * Mutable conversation history for the REPL. Accumulates user and assistant
 * turns so that each request can be sent with full context (multi-turn chat).
 *
 * Each ChatHistory instance owns a stable {@link #sessionId()} that should be
 * passed to {@link cab.ml.juno.coordinator.InferenceRequest#ofSession} for
 * every turn. This lets the GenerationLoop reuse KV cache blocks across turns
 * instead of re-running the full prefill on each request.
 */
public final class ChatHistory {

	private final String sessionId = UUID.randomUUID().toString();
	private final List<ChatMessage> messages = new ArrayList<>();

	/** Stable session identifier — share this across all turns of the conversation. */
	public String sessionId() {
		return sessionId;
	}

	/** Appends a user message. */
	public void addUser(String content) {
		messages.add(ChatMessage.user(content));
	}

	/** Appends an assistant message. */
	public void addAssistant(String content) {
		messages.add(ChatMessage.assistant(content));
	}

	/** Returns a copy of the current message list for building an InferenceRequest. */
	public List<ChatMessage> getMessages() {
		return List.copyOf(messages);
	}
}