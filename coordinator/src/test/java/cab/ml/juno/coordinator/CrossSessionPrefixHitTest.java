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

import java.util.List;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.kvcache.ServeScheduleOptions;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.ChatTemplateFormatter;
import cab.ml.juno.tokenizer.SimpleTokenizer;

/**
 * A prefix-cache entry belongs to the session that wrote it. A second session whose prompt starts with the first
 * session's cached prompt (the same opening message, say) holds no KV of its own at those positions, so it must not
 * skip their prefill: its reply must equal the reply it gets with nothing cached. Each session still skips its own
 * cached prefix on its next turn, when another session has cached the same prompt and when that session has ended.
 * The single-request path and the continuous engine both.
 */
class CrossSessionPrefixHitTest {

	private static final String MODEL = "tinyllama";
	private static final String SYSTEM = "You are a careful assistant that answers briefly and precisely";
	private static final SamplingParams PARAMS = SamplingParams.deterministic().withMaxTokens(4);

	private SimpleTokenizer tokenizer;
	private KvTrackingPipeline pipeline;
	private GenerationLoop loop;
	private RequestScheduler continuous;

	@BeforeEach
	void setUp() {
		tokenizer = new SimpleTokenizer();
		pipeline = new KvTrackingPipeline();
		loop = newLoop(pipeline);
		continuous = new RequestScheduler(32, loop, BatchConfig.of(4, 20), ServeScheduleOptions.parse("continuous"));
	}

	@AfterEach
	void tearDown() {
		continuous.shutdown();
	}

	private GenerationLoop newLoop(KvTrackingPipeline p) {
		return new GenerationLoop(tokenizer, Sampler.create(), p,
				new KVCacheManager(new GpuKVCache(64 * 1024 * 1024), new CpuKVCache(1000)));
	}

	private static List<ChatMessage> messages(String user) {
		return List.of(ChatMessage.system(SYSTEM), ChatMessage.user(user));
	}

	private static List<ChatMessage> secondTurn(String user) {
		return List.of(ChatMessage.system(SYSTEM), ChatMessage.user(user), ChatMessage.assistant("noted"),
				ChatMessage.user("and then"));
	}

	private static InferenceRequest session(String sessionId, List<ChatMessage> messages) {
		return InferenceRequest.ofSession(sessionId, MODEL, messages, PARAMS, RequestPriority.NORMAL);
	}

	private int promptLength(List<ChatMessage> messages) {
		return tokenizer.encode(ChatTemplateFormatter.forModelType(MODEL).format(messages)).length;
	}

	private GenerationResult single(InferenceRequest request) {
		return loop.generate(request, TokenConsumer.discard());
	}

	private GenerationResult continuous(InferenceRequest request) throws Exception {
		return continuous.submit(request, TokenConsumer.discard()).get(30, TimeUnit.SECONDS);
	}

	private List<Integer> reference(List<ChatMessage> messages) {
		return newLoop(new KvTrackingPipeline())
				.generate(InferenceRequest.of(MODEL, messages, PARAMS, RequestPriority.NORMAL), TokenConsumer.discard())
				.tokenIds();
	}

	@Test
	void single_path_a_second_session_with_the_same_opening_prompt_is_fully_prefilled() {
		single(session("s-a", messages("hello")));
		assertThat(pipeline.underruns()).isEmpty();

		GenerationResult b = single(session("s-b", messages("hello")));

		assertThat(pipeline.underruns()).as("session s-b continued from KV only session s-a wrote").isEmpty();
		assertThat(b.tokenIds()).isEqualTo(reference(messages("hello")));
	}

	@Test
	void single_path_a_second_session_whose_prompt_extends_the_first_sessions_is_fully_prefilled() {
		single(session("s-a", messages("hello")));

		GenerationResult b = single(session("s-b", secondTurn("hello")));

		assertThat(pipeline.underruns()).as("session s-b continued from KV only session s-a wrote").isEmpty();
		assertThat(b.tokenIds()).isEqualTo(reference(secondTurn("hello")));
	}

	@Test
	void single_path_each_session_keeps_its_own_hit_when_both_cached_the_same_prompt() {
		single(session("s-a", messages("hello")));
		single(session("s-b", messages("hello")));
		int cached = promptLength(messages("hello"));

		int mark = pipeline.mark();
		GenerationResult a2 = single(session("s-a", secondTurn("hello")));
		assertThat(pipeline.positionsWrittenSince(mark, "s-a")).as("s-a's turn 2 starts after its cached prompt")
				.first().isEqualTo(cached);

		loop.evictSession("s-a");
		mark = pipeline.mark();
		GenerationResult b2 = single(session("s-b", secondTurn("hello")));
		assertThat(pipeline.positionsWrittenSince(mark, "s-b"))
				.as("s-b keeps its hit after s-a's session ended").first().isEqualTo(cached);

		assertThat(pipeline.underruns()).isEmpty();
		assertThat(a2.tokenIds()).isEqualTo(reference(secondTurn("hello")));
		assertThat(b2.tokenIds()).isEqualTo(reference(secondTurn("hello")));
	}

	@Test
	void single_path_a_session_that_branches_back_does_not_resume_over_positions_it_overwrote() {
		List<ChatMessage> turn2 = secondTurn("hello");
		List<ChatMessage> branch = List.of(ChatMessage.system(SYSTEM), ChatMessage.user("hello"),
				ChatMessage.assistant("something else entirely"), ChatMessage.user("instead"));
		List<ChatMessage> turn4 = List.of(ChatMessage.system(SYSTEM), ChatMessage.user("hello"),
				ChatMessage.assistant("noted"), ChatMessage.user("and then"), ChatMessage.assistant("ok"),
				ChatMessage.user("more"));
		single(session("s-a", messages("hello")));
		single(session("s-a", turn2));
		single(session("s-a", branch)); // overwrites the positions after the first turn's prompt

		GenerationResult back = single(session("s-a", turn4)); // extends turn 2's prompt, not the branch's

		assertThat(pipeline.underruns()).isEmpty();
		assertThat(back.tokenIds()).as("resumed over positions the branch overwrote").isEqualTo(reference(turn4));
	}

	@Test
	void continuous_a_second_session_with_the_same_opening_prompt_is_fully_prefilled() throws Exception {
		continuous(session("s-a", messages("hello")));
		assertThat(pipeline.underruns()).isEmpty();

		GenerationResult b = continuous(session("s-b", messages("hello")));
		GenerationResult c = continuous(session("s-c", secondTurn("hello")));

		assertThat(pipeline.underruns()).as("a session continued from KV only session s-a wrote").isEmpty();
		assertThat(b.tokenIds()).isEqualTo(reference(messages("hello")));
		assertThat(c.tokenIds()).isEqualTo(reference(secondTurn("hello")));
	}

	@Test
	void continuous_each_session_keeps_its_own_hit_when_both_cached_the_same_prompt() throws Exception {
		continuous(session("s-a", messages("hello")));
		continuous(session("s-b", messages("hello")));
		int cached = promptLength(messages("hello"));

		int mark = pipeline.mark();
		GenerationResult a2 = continuous(session("s-a", secondTurn("hello")));
		assertThat(pipeline.positionsWrittenSince(mark, "s-a")).first().isEqualTo(cached);
		mark = pipeline.mark();
		GenerationResult b2 = continuous(session("s-b", secondTurn("hello")));
		assertThat(pipeline.positionsWrittenSince(mark, "s-b")).first().isEqualTo(cached);

		assertThat(pipeline.underruns()).isEmpty();
		assertThat(a2.tokenIds()).isEqualTo(reference(secondTurn("hello")));
		assertThat(b2.tokenIds()).isEqualTo(reference(secondTurn("hello")));
	}
}
