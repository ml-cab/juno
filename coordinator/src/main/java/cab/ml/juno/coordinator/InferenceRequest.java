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

import java.time.Instant;
import java.util.List;
import java.util.UUID;

import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;

/**
 * Immutable value object representing a single inference request.
 *
 * Created by the REST/gRPC layer and submitted to the RequestScheduler. Carries
 * everything the GenerationLoop needs: messages, sampling config, model ID, and
 * client-facing metadata.
 *
 * <h3>Single-turn vs multi-turn (session) requests</h3>
 * Use {@link #of} for stateless, single-turn requests. Each call gets a fresh
 * requestId and the KV cache is discarded after generation.
 *
 * Use {@link #ofSession} for multi-turn conversation requests. The caller
 * supplies a stable {@code sessionId} that is shared across all turns of the
 * same conversation. {@link #kvCacheKey()} returns the sessionId so that the
 * GenerationLoop and the underlying pipeline can reuse KV blocks built in
 * earlier turns instead of re-running the full prefill each time.
 *
 * Call {@link cab.ml.juno.coordinator.GenerationLoop#evictSession} when
 * the conversation ends to release KV memory.
 */
public record InferenceRequest(String requestId, String sessionId, String modelId, List<ChatMessage> messages,
		SamplingParams samplingParams, RequestPriority priority, Instant receivedAt)
		implements Comparable<InferenceRequest> {

	public InferenceRequest {
		if (requestId == null || requestId.isBlank())
			throw new IllegalArgumentException("requestId must not be blank");
		if (modelId == null || modelId.isBlank())
			throw new IllegalArgumentException("modelId must not be blank");
		if (messages == null || messages.isEmpty())
			throw new IllegalArgumentException("messages must not be empty");
		if (samplingParams == null)
			throw new IllegalArgumentException("samplingParams must not be null");
		if (priority == null)
			throw new IllegalArgumentException("priority must not be null");
		// sessionId may be null — null means stateless/single-turn.

		messages = List.copyOf(messages); // defensive copy
	}

	/**
	 * Factory for stateless single-turn requests.
	 * Generates a random requestId; sessionId is null.
	 */
	public static InferenceRequest of(String modelId, List<ChatMessage> messages, SamplingParams params,
			RequestPriority priority) {
		return new InferenceRequest(UUID.randomUUID().toString(), null, modelId, messages, params, priority,
				Instant.now());
	}

	/**
	 * Factory for multi-turn session requests.
	 *
	 * @param sessionId stable identifier shared across all turns of the same
	 *                  conversation — must not be blank
	 */
	public static InferenceRequest ofSession(String sessionId, String modelId, List<ChatMessage> messages,
			SamplingParams params, RequestPriority priority) {
		if (sessionId == null || sessionId.isBlank())
			throw new IllegalArgumentException("sessionId must not be blank");
		return new InferenceRequest(UUID.randomUUID().toString(), sessionId, modelId, messages, params, priority,
				Instant.now());
	}

	/**
	 * The key to use for all KV cache operations (prefix trie + pipeline storage).
	 *
	 * Session requests: returns the stable sessionId so that KV blocks survive
	 * across turns and the prefix cache can skip already-processed tokens.
	 *
	 * Stateless requests: returns the per-request UUID so each request is
	 * completely isolated and cleaned up after generation.
	 */
	public String kvCacheKey() {
		return sessionId != null ? sessionId : requestId;
	}

	/** Higher priority = lower compareTo value = scheduled first. */
	@Override
	public int compareTo(InferenceRequest other) {
		int cmp = Integer.compare(other.priority().weight(), this.priority().weight());
		if (cmp != 0)
			return cmp;
		// FIFO within same priority
		return this.receivedAt().compareTo(other.receivedAt());
	}
}