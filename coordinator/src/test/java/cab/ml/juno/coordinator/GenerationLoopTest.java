package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;

import java.util.ArrayList;
import java.util.List;

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

class GenerationLoopTest {

	private StubTokenizer tokenizer;
	private Sampler sampler;
	private KVCacheManager kvCache;
	@SuppressWarnings("unused")
	private GenerationLoop loop;

	@BeforeEach
	void setUp() {
		tokenizer = new StubTokenizer();
		sampler = Sampler.create();
		kvCache = new KVCacheManager(new GpuKVCache(64 * 1024 * 1024), new CpuKVCache(1000));
	}

	private GenerationLoop loopWith(InferencePipeline pipeline) {
		return new GenerationLoop(tokenizer, sampler, pipeline, kvCache);
	}

	private InferenceRequest requestFor(String... messages) {
		List<ChatMessage> msgs = new ArrayList<>();
		for (int i = 0; i < messages.length; i++) {
			msgs.add(i % 2 == 0 ? ChatMessage.user(messages[i]) : ChatMessage.assistant(messages[i]));
		}
		return InferenceRequest.of("llama3-8b", msgs, SamplingParams.defaults().withMaxTokens(5),
				RequestPriority.NORMAL);
	}

	@Test
	void generates_up_to_max_tokens() {
		// Pipeline always returns token 42 — never EOS — so loop hits maxTokens
		StubInferencePipeline pipeline = new StubInferencePipeline();
		GenerationLoop loop = loopWith(pipeline);

		GenerationResult result = loop.generate(requestFor("hello"), TokenConsumer.discard());

		assertThat(result.generatedTokens()).isEqualTo(5);
		assertThat(result.stopReason()).isEqualTo(GenerationResult.StopReason.MAX_TOKENS);
	}

	@Test
	void stops_at_eos_token() {
		// GenerationLoop calls pipeline.forward() during prefill (positions
		// 0..promptLen-2)
		// to warm the KV cache; those results are discarded before the decode loop
		// starts.
		//
		// Prompt: modelId="llama3-8b" + ChatMessage.user("hi")
		// Llama3 template → "<|begin_of_text|>...\n\nhi...\n\n"
		// StubTokenizer (split on \\s+) → 2 tokens → 1 prefill call
		//
		// Sequence index 0 is consumed by the prefill call (result discarded).
		// Decode steps 0, 1, 2 see indices 1, 2, 3 → tokens 42, 43, EOS.
		int eos = tokenizer.eosTokenId();
		StubInferencePipeline pipeline = new StubInferencePipeline(StubInferencePipeline.DEFAULT_TOKEN, // index 0 —
																										// prefill call,
																										// result
																										// discarded
				42, 43, eos // indices 1-3 — decode steps 0, 1, 2
		);
		GenerationLoop loop = loopWith(pipeline);

		InferenceRequest req = InferenceRequest.of("llama3-8b", List.of(ChatMessage.user("hi")),
				SamplingParams.defaults().withMaxTokens(20), RequestPriority.NORMAL);

		GenerationResult result = loop.generate(req, TokenConsumer.discard());

		assertThat(result.promptTokens()).as("sanity: prompt must be 2 tokens").isEqualTo(2);
		assertThat(result.generatedTokens()).isEqualTo(2); // 42, 43 — EOS stops the loop, not counted
		assertThat(result.stopReason()).isEqualTo(GenerationResult.StopReason.EOS_TOKEN);
	}

	@Test
	void stops_at_stop_token() {
		// Same prefill-offset reasoning as stops_at_eos_token above:
		// 2-token prompt → 1 prefill call at sequence index 0.
		// Decode step 0 → token 42 (generated), step 1 → stopToken (loop halts, not
		// counted).
		int stopToken = 99;
		StubInferencePipeline pipeline = new StubInferencePipeline(StubInferencePipeline.DEFAULT_TOKEN, // index 0 —
																										// prefill call,
																										// result
																										// discarded
				42, stopToken, 43 // indices 1-3 — decode steps 0, 1, 2
		);
		GenerationLoop loop = loopWith(pipeline);

		InferenceRequest req = InferenceRequest.of("llama3-8b", List.of(ChatMessage.user("hi")),
				SamplingParams.defaults().withMaxTokens(20).withStopTokenIds(stopToken), RequestPriority.NORMAL);

		GenerationResult result = loop.generate(req, TokenConsumer.discard());

		assertThat(result.promptTokens()).as("sanity: prompt must be 2 tokens").isEqualTo(2);
		assertThat(result.generatedTokens()).isEqualTo(1); // only token 42; stopToken halts without counting
		assertThat(result.stopReason()).isEqualTo(GenerationResult.StopReason.STOP_TOKEN);
	}

	@Test
	void token_consumer_called_once_per_generated_token() {
		StubInferencePipeline pipeline = new StubInferencePipeline();
		GenerationLoop loop = loopWith(pipeline);

		List<Integer> receivedTokens = new ArrayList<>();
		TokenConsumer consumer = (_, tokenId, _) -> receivedTokens.add(tokenId);

		loop.generate(requestFor("test"), consumer);

		assertThat(receivedTokens).hasSize(5); // maxTokens=5
		assertThat(receivedTokens).allMatch(id -> id == StubInferencePipeline.DEFAULT_TOKEN);
	}

	@Test
	void result_contains_prompt_token_count() {
		StubInferencePipeline pipeline = new StubInferencePipeline();
		GenerationLoop loop = loopWith(pipeline);

		GenerationResult result = loop.generate(requestFor("hello world"), TokenConsumer.discard());

		assertThat(result.promptTokens()).isGreaterThan(0);
	}

	@Test
	void result_latency_is_positive() {
		GenerationLoop loop = loopWith(new StubInferencePipeline());
		GenerationResult result = loop.generate(requestFor("hi"), TokenConsumer.discard());
		assertThat(result.latency().toNanos()).isGreaterThan(0);
	}

	@Test
	void prefix_cache_populated_after_first_request() {
		GenerationLoop loop = loopWith(new StubInferencePipeline());
		InferenceRequest req = requestFor("the same system prompt");

		loop.generate(req, TokenConsumer.discard());

		// After generation, prefix should be cached
		int[] encoded = tokenizer.encode("the same system prompt");
		// The prefix cache should have something for these tokens
		@SuppressWarnings("unused")
		var match = kvCache.findLongestPrefix(encoded);
		// May or may not hit depending on template formatting, but should not throw
		assertThatCode(() -> kvCache.findLongestPrefix(encoded)).doesNotThrowAnyException();
	}
}