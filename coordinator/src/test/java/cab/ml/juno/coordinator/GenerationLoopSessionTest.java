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

import static org.assertj.core.api.Assertions.assertThat;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.CopyOnWriteArrayList;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.node.InferencePipeline;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.StubTokenizer;

/**
 * Tests for session-aware KV cache reuse in GenerationLoop.
 *
 * Core invariant: when the caller supplies a sessionId, subsequent turns in the
 * same conversation must NOT re-run the forward pass for tokens that were
 * already processed in earlier turns. We verify this by recording every
 * (requestId/sessionKey, startPos) pair that the pipeline sees.
 */
class GenerationLoopSessionTest {

	private StubTokenizer tokenizer;
	private Sampler sampler;
	private KVCacheManager kvCache;

	@BeforeEach
	void setUp() {
		tokenizer = new StubTokenizer();
		sampler = Sampler.create();
		kvCache = new KVCacheManager(new GpuKVCache(64 * 1024 * 1024), new CpuKVCache(1000));
	}

	// ── helpers ───────────────────────────────────────────────────────────────

	private GenerationLoop loopWith(InferencePipeline pipeline) {
		return new GenerationLoop(tokenizer, sampler, pipeline, kvCache);
	}

	/** Build a single-turn session request (user message only). */
	private InferenceRequest sessionRequest(String sessionId, String userText) {
		return InferenceRequest.ofSession(sessionId, "llama3-8b", List.of(ChatMessage.user(userText)),
				SamplingParams.defaults().withMaxTokens(3), RequestPriority.NORMAL);
	}

	/** Build a multi-turn session request (user + assistant alternating). */
	private InferenceRequest sessionRequest(String sessionId, List<ChatMessage> messages) {
		return InferenceRequest.ofSession(sessionId, "llama3-8b", messages, SamplingParams.defaults().withMaxTokens(3),
				RequestPriority.NORMAL);
	}

	/** Build a stateless (no session) request. */
	private InferenceRequest statelessRequest(String userText) {
		return InferenceRequest.of("llama3-8b", List.of(ChatMessage.user(userText)),
				SamplingParams.defaults().withMaxTokens(3), RequestPriority.NORMAL);
	}

	// ── tests ─────────────────────────────────────────────────────────────────

	@Test
	void second_turn_advances_startPos_beyond_zero() {
		// The forward spy records every startPos seen during a forward() call.
		List<Integer> allStartPositions = new CopyOnWriteArrayList<>();
		SpyInferencePipeline spy = new SpyInferencePipeline(allStartPositions);
		GenerationLoop loop = loopWith(spy);

		String sessionId = UUID.randomUUID().toString();

		// Turn 1 — cold cache; prefill starts at 0.
		InferenceRequest turn1 = sessionRequest(sessionId, "hello my name is dima");
		loop.generate(turn1, TokenConsumer.discard());

		int minStartPosTurn1 = allStartPositions.stream().mapToInt(Integer::intValue).min().orElse(-1);
		assertThat(minStartPosTurn1).as("turn 1 cold-start must begin at position 0").isEqualTo(0);

		// Turn 2 — the full turn-1 prompt is now in the prefix cache under sessionId.
		// Turn 2's formatted prompt begins with all of turn 1's prompt tokens
		// (conversation grows monotonically), so findLongestPrefix returns a hit and
		// the first forward() call must start at a position > 0.
		allStartPositions.clear();

		List<ChatMessage> turn2Messages = new ArrayList<>();
		turn2Messages.add(ChatMessage.user("hello my name is dima"));
		turn2Messages.add(ChatMessage.assistant("nice to meet you"));
		turn2Messages.add(ChatMessage.user("what is my name?"));
		InferenceRequest turn2 = sessionRequest(sessionId, turn2Messages);
		loop.generate(turn2, TokenConsumer.discard());

		int minStartPosTurn2 = allStartPositions.stream().mapToInt(Integer::intValue).min().orElse(-1);
		assertThat(minStartPosTurn2).as("turn 2 must skip cached prefix — minStartPos must be > 0").isGreaterThan(0);
	}

	@Test
	void session_reuse_reduces_forward_calls_vs_no_cache() {
		// Without a session, GenerationLoop re-prefills the full prompt each time.
		// With a session, it skips all tokens already processed in earlier turns.
		// We compare the forward() call count for the SAME multi-turn prompt under
		// both conditions. The cached run must issue fewer calls.

		String sessionId = UUID.randomUUID().toString();

		List<ChatMessage> turn2Msgs = new ArrayList<>();
		turn2Msgs.add(ChatMessage.user("hello my name is dima"));
		turn2Msgs.add(ChatMessage.assistant("nice to meet you"));
		turn2Msgs.add(ChatMessage.user("what is my name?"));

		// Warm the session cache with turn 1.
		loopWith(new StubInferencePipeline()).generate(sessionRequest(sessionId, "hello my name is dima"),
				TokenConsumer.discard());

		// Run turn 2 WITHOUT session (stateless — full re-prefill every time).
		List<Integer> noCachePositions = new CopyOnWriteArrayList<>();
		new GenerationLoop(tokenizer, sampler, new SpyInferencePipeline(noCachePositions), kvCache)
				.generate(InferenceRequest.of("llama3-8b", turn2Msgs, SamplingParams.defaults().withMaxTokens(3),
						RequestPriority.NORMAL), TokenConsumer.discard());
		int noCacheCalls = noCachePositions.size();

		// Run turn 2 WITH session — should skip turn 1's prompt tokens.
		List<Integer> cachedPositions = new CopyOnWriteArrayList<>();
		new GenerationLoop(tokenizer, sampler, new SpyInferencePipeline(cachedPositions), kvCache)
				.generate(sessionRequest(sessionId, turn2Msgs), TokenConsumer.discard());
		int cachedCalls = cachedPositions.size();

		assertThat(cachedCalls).as("session reuse must issue fewer forward() calls than full re-prefill "
				+ "(noCacheCalls=" + noCacheCalls + ", cachedCalls=" + cachedCalls + ")").isLessThan(noCacheCalls);
	}

	@Test
	void stateless_request_always_starts_at_zero() {
		// Without a sessionId the loop must behave exactly as before:
		// always prefill from position 0, no prefix-cache involvement.
		List<Integer> positions = new CopyOnWriteArrayList<>();
		GenerationLoop loop = loopWith(new SpyInferencePipeline(positions));

		loop.generate(statelessRequest("hello world"), TokenConsumer.discard());
		int minPos = positions.stream().mapToInt(Integer::intValue).min().orElse(-1);
		assertThat(minPos).as("stateless request must always prefill from 0").isEqualTo(0);
	}

	@Test
	void stateless_request_evicts_kv_on_completion() {
		// After a stateless request the prefix cache must be empty for those tokens.
		GenerationLoop loop = loopWith(new StubInferencePipeline());

		loop.generate(statelessRequest("the quick brown fox"), TokenConsumer.discard());

		// Prefix cache for these tokens should NOT be populated (stateless — no
		// cachePrefix call).
		int[] encoded = tokenizer.encode("the quick brown fox");
		var match = kvCache.findLongestPrefix(encoded);
		assertThat(match.isHit()).as("stateless request must not leave a prefix cache entry").isFalse();
	}

	@Test
	void session_prefix_is_cached_after_first_turn() {
		// After turn 1 completes, the formatted prompt tokens for that turn are
		// stored in the prefix trie. We verify this indirectly: a turn 2 spy must
		// see a non-zero starting position, proving the trie returned a cache hit.
		String sessionId = UUID.randomUUID().toString();

		// Turn 1 — populates the prefix cache.
		loopWith(new StubInferencePipeline()).generate(sessionRequest(sessionId, "the quick brown fox"),
				TokenConsumer.discard());

		// Turn 2 — if the prefix was cached, forward() must be called with startPos >
		// 0.
		List<Integer> positions = new CopyOnWriteArrayList<>();
		List<ChatMessage> turn2Msgs = new ArrayList<>();
		turn2Msgs.add(ChatMessage.user("the quick brown fox"));
		turn2Msgs.add(ChatMessage.assistant("jumps over the lazy dog"));
		turn2Msgs.add(ChatMessage.user("what did the fox do?"));

		new GenerationLoop(tokenizer, sampler, new SpyInferencePipeline(positions), kvCache)
				.generate(sessionRequest(sessionId, turn2Msgs), TokenConsumer.discard());

		int minStartPos = positions.stream().mapToInt(Integer::intValue).min().orElse(-1);
		assertThat(minStartPos).as("session turn 1 must populate the prefix cache — turn 2 must start at pos > 0")
				.isGreaterThan(0);
	}

	@Test
	void evict_session_clears_prefix_cache() {
		String sessionId = UUID.randomUUID().toString();
		GenerationLoop loop = loopWith(new StubInferencePipeline());

		// Turn 1 — warm the cache.
		loop.generate(sessionRequest(sessionId, "hello world"), TokenConsumer.discard());

		// Sanity: turn 2 should get a cache hit (startPos > 0).
		List<Integer> warmPositions = new CopyOnWriteArrayList<>();
		List<ChatMessage> turn2Msgs = List.of(ChatMessage.user("hello world"), ChatMessage.assistant("greetings"),
				ChatMessage.user("how are you?"));
		new GenerationLoop(tokenizer, sampler, new SpyInferencePipeline(warmPositions), kvCache)
				.generate(sessionRequest(sessionId, turn2Msgs), TokenConsumer.discard());
		int minWarm = warmPositions.stream().mapToInt(Integer::intValue).min().orElse(-1);
		assertThat(minWarm).as("sanity: cache must be warm before eviction").isGreaterThan(0);

		// Evict the session — clears KV blocks AND prefix-trie entry.
		loop.evictSession(sessionId);

		// After eviction the same sessionId must behave as cold: prefill from 0.
		List<Integer> coldPositions = new CopyOnWriteArrayList<>();
		new GenerationLoop(tokenizer, sampler, new SpyInferencePipeline(coldPositions), kvCache)
				.generate(sessionRequest(sessionId, "hello world"), TokenConsumer.discard());
		int minCold = coldPositions.stream().mapToInt(Integer::intValue).min().orElse(-1);
		assertThat(minCold).as("evictSession must remove the prefix cache entry — next run must start at 0")
				.isEqualTo(0);
	}

	@Test
	void different_sessions_are_isolated() {
		// Two concurrent sessions must not interfere with each other's prefix state.
		GenerationLoop loop = loopWith(new StubInferencePipeline());

		String sessionA = UUID.randomUUID().toString();
		String sessionB = UUID.randomUUID().toString();

		loop.generate(sessionRequest(sessionA, "session a first message"), TokenConsumer.discard());
		loop.generate(sessionRequest(sessionB, "session b first message"), TokenConsumer.discard());

		// Evict A — B's prefix must survive and still produce a cache hit on turn 2.
		loop.evictSession(sessionA);

		List<ChatMessage> bTurn2Msgs = List.of(ChatMessage.user("session b first message"),
				ChatMessage.assistant("acknowledged"), ChatMessage.user("session b second message"));

		List<Integer> bPositions = new CopyOnWriteArrayList<>();
		new GenerationLoop(tokenizer, sampler, new SpyInferencePipeline(bPositions), kvCache)
				.generate(sessionRequest(sessionB, bTurn2Msgs), TokenConsumer.discard());

		int minB = bPositions.stream().mapToInt(Integer::intValue).min().orElse(-1);
		assertThat(minB).as("session B prefix must survive eviction of session A — turn 2 startPos must be > 0")
				.isGreaterThan(0);
	}

	@Test
	void results_carry_generated_text() {
		// Basic sanity — session requests still return a GenerationResult with text.
		GenerationLoop loop = loopWith(new StubInferencePipeline());
		String sessionId = UUID.randomUUID().toString();

		GenerationResult result = loop.generate(sessionRequest(sessionId, "hi"), TokenConsumer.discard());

		assertThat(result).isNotNull();
		assertThat(result.generatedTokens()).isGreaterThan(0);
	}

	@Test
	void ofSession_rejects_blank_session_id() {
		org.assertj.core.api.Assertions
				.assertThatThrownBy(() -> InferenceRequest.ofSession("  ", "llama3-8b", List.of(ChatMessage.user("hi")),
						SamplingParams.defaults(), RequestPriority.NORMAL))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("sessionId");
	}

	@Test
	void kv_cache_key_is_session_id_when_session_present() {
		String sessionId = UUID.randomUUID().toString();
		InferenceRequest req = InferenceRequest.ofSession(sessionId, "llama3-8b", List.of(ChatMessage.user("hi")),
				SamplingParams.defaults(), RequestPriority.NORMAL);

		assertThat(req.kvCacheKey()).isEqualTo(sessionId);
	}

	@Test
	void kv_cache_key_is_request_id_when_no_session() {
		InferenceRequest req = InferenceRequest.of("llama3-8b", List.of(ChatMessage.user("hi")),
				SamplingParams.defaults(), RequestPriority.NORMAL);

		assertThat(req.kvCacheKey()).isEqualTo(req.requestId());
	}

	// ── inner spy ─────────────────────────────────────────────────────────────

	/**
	 * InferencePipeline spy that records every startPos passed to forward().
	 * Returns fixed logits pointing at token 42 (non-EOS, non-stop).
	 */
	private static final class SpyInferencePipeline implements InferencePipeline {

		private static final int VOCAB = 1000;
		private final List<Integer> recordedStartPositions;

		SpyInferencePipeline(List<Integer> recordedStartPositions) {
			this.recordedStartPositions = recordedStartPositions;
		}

		@Override
		public float[] forward(String requestId, int[] tokens, int startPos) {
			recordedStartPositions.add(startPos);
			float[] logits = new float[VOCAB];
			logits[42] = 100f;
			return logits;
		}

		@Override
		public int vocabSize() {
			return VOCAB;
		}
	}
}