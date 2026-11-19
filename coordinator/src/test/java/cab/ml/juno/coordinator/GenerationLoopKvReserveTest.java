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
import java.util.Collections;
import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.SimpleTokenizer;

/**
 * The generation loop tells the pipeline how many KV positions a request can write (its prompt plus
 * {@code max_tokens}, plus the speculative lookahead when speculation is on), once per request and before its
 * prefill, on the single-request path and for every member of a static batch, so the stages' dense KV grows toward
 * that count instead of doubling past it.
 */
@DisplayName("GenerationLoop - KV positions reserved before the prefill")
class GenerationLoopKvReserveTest {

	private static final int MAX_TOKENS = 7;
	private static final List<ChatMessage> CHAT = List.of(ChatMessage.user("tell me about the sea and the sky"));

	/** Records reservations and the first prefill or forward pass of each request, in order. */
	static final class Recording extends GenerationLoopContextShiftTest.LimitedPipeline {
		final List<String> events = Collections.synchronizedList(new ArrayList<>());

		Recording() {
			super(4096);
		}

		@Override
		public void reserveKv(String requestId, int positions) {
			events.add("reserve " + requestId + " " + positions);
		}

		@Override
		public void prefillBatch(String requestId, int[] newTokens, int startPosition) {
			events.add("prefill " + requestId + " " + (startPosition + newTokens.length));
			super.prefillBatch(requestId, newTokens, startPosition);
		}
	}

	private static GenerationLoop loop(Recording p, SpeculativeDecodeOptions spec) {
		return new GenerationLoop(new SimpleTokenizer(), Sampler.create(), p,
				new KVCacheManager(new GpuKVCache(64 * 1024 * 1024), new CpuKVCache(100)), PrefillMode.BATCHED,
				PrefillBatchOptions.DEFAULT_CHUNK_SIZE, spec);
	}

	private static InferenceRequest request() {
		return InferenceRequest.of("llama3-8b", CHAT, SamplingParams.defaults().withMaxTokens(MAX_TOKENS),
				RequestPriority.NORMAL);
	}

	@Test
	@DisplayName("single request: prompt plus max_tokens, reserved once, before the prefill")
	void singleRequest() {
		Recording p = new Recording();
		InferenceRequest r = request();
		loop(p, SpeculativeDecodeOptions.disabled()).generate(r, TokenConsumer.discard());
		List<String> reserves = p.events.stream().filter(e -> e.startsWith("reserve ")).toList();
		assertThat(reserves).hasSize(1);
		int positions = Integer.parseInt(reserves.get(0).substring(reserves.get(0).lastIndexOf(' ') + 1));
		int prefilled = prefilledPositions(p, r.kvCacheKey());
		assertThat(positions).as("prompt (%d tokens) plus max_tokens", prefilled + 1).isEqualTo(prefilled + 1 + MAX_TOKENS);
		assertThat(reserves.get(0)).startsWith("reserve " + r.kvCacheKey() + " ");
		assertThat(p.events.indexOf(reserves.get(0))).isLessThan(firstPrefill(p, r.kvCacheKey()));
		assertThat(p.maxPosition).isLessThan(positions);
	}

	@Test
	@DisplayName("speculation on: the n-gram lookahead is reserved too")
	void speculationAddsTheLookahead() {
		Recording p = new Recording();
		InferenceRequest r = request();
		SpeculativeDecodeOptions spec = SpeculativeDecodeOptions.of(SpeculativeDecodeOptions.SpecType.NGRAM_SIMPLE, 2, 5);
		loop(p, spec).generate(r, TokenConsumer.discard());
		String reserve = p.events.stream().filter(e -> e.startsWith("reserve ")).findFirst().orElseThrow();
		int positions = Integer.parseInt(reserve.substring(reserve.lastIndexOf(' ') + 1));
		assertThat(positions).isEqualTo(prefilledPositions(p, r.kvCacheKey()) + 1 + MAX_TOKENS + 5);
	}

	@Test
	@DisplayName("static batch: every member reserved before its own prefill")
	void staticBatch() {
		Recording p = new Recording();
		InferenceRequest a = request();
		InferenceRequest b = request();
		loop(p, SpeculativeDecodeOptions.disabled()).generateBatch(
				List.of(new BatchEntry(a, TokenConsumer.discard()), new BatchEntry(b, TokenConsumer.discard())));
		for (InferenceRequest r : List.of(a, b)) {
			String id = r.requestId();
			List<String> reserves = p.events.stream().filter(e -> e.startsWith("reserve " + id + " ")).toList();
			assertThat(reserves).as("reservations of %s", id).hasSize(1);
			int positions = Integer.parseInt(reserves.get(0).substring(reserves.get(0).lastIndexOf(' ') + 1));
			assertThat(positions).isEqualTo(prefilledPositions(p, id) + 1 + MAX_TOKENS);
			assertThat(p.events.indexOf(reserves.get(0))).isLessThan(firstPrefill(p, id));
		}
	}

	/** Positions the request's prefill wrote: the prompt less its last token, which the first decode step feeds. */
	private static int prefilledPositions(Recording p, String id) {
		return p.events.stream().filter(e -> e.startsWith("prefill " + id + " "))
				.mapToInt(e -> Integer.parseInt(e.substring(e.lastIndexOf(' ') + 1))).max().orElseThrow();
	}

	private static int firstPrefill(Recording p, String id) {
		for (int i = 0; i < p.events.size(); i++)
			if (p.events.get(i).startsWith("prefill " + id + " "))
				return i;
		throw new AssertionError("no prefill of " + id + ": " + p.events);
	}
}
