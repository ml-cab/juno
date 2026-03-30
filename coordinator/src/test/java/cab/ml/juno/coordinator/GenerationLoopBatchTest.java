package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.node.InferencePipeline;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.SimpleTokenizer;

class GenerationLoopBatchTest {

	private GenerationLoop loop;
	private SimpleTokenizer tokenizer;

	@BeforeEach
	void setUp() {
		tokenizer = new SimpleTokenizer();
		loop = new GenerationLoop(tokenizer, Sampler.create(), new StubInferencePipeline(),
				new KVCacheManager(new GpuKVCache(64 * 1024 * 1024), new CpuKVCache(1000)));
	}

	private BatchEntry entry(String prompt) {
		return new BatchEntry(InferenceRequest.of("tinyllama", List.of(ChatMessage.user(prompt)),
				SamplingParams.defaults().withMaxTokens(3), RequestPriority.NORMAL), TokenConsumer.discard());
	}

	// ── Correctness ──────────────────────────────────────────────────────────

	@Test
	void batch_of_one_matches_single_generate() {
		InferenceRequest req = InferenceRequest.of("tinyllama", List.of(ChatMessage.user("hello")),
				SamplingParams.defaults().withMaxTokens(3), RequestPriority.NORMAL);

		GenerationResult single = loop.generate(req, TokenConsumer.discard());

		// Fresh loop for batch (same pipeline, same tokenizer)
		GenerationLoop batchLoop = new GenerationLoop(tokenizer, Sampler.create(), new StubInferencePipeline(),
				new KVCacheManager(new GpuKVCache(64 * 1024 * 1024), new CpuKVCache(1000)));
		List<GenerationResult> batchResults = batchLoop
				.generateBatch(List.of(new BatchEntry(req, TokenConsumer.discard())));

		assertThat(batchResults).hasSize(1);
		assertThat(batchResults.get(0).generatedTokens()).isEqualTo(single.generatedTokens());
		assertThat(batchResults.get(0).stopReason()).isEqualTo(single.stopReason());
	}

	@Test
	void batch_returns_one_result_per_entry() {
		List<GenerationResult> results = loop
				.generateBatch(List.of(entry("prompt one"), entry("prompt two"), entry("prompt three")));
		assertThat(results).hasSize(3);
	}

	@Test
	void each_result_has_correct_request_id() {
		BatchEntry e1 = entry("alpha");
		BatchEntry e2 = entry("beta");

		List<GenerationResult> results = loop.generateBatch(List.of(e1, e2));

		assertThat(results.get(0).requestId()).isEqualTo(e1.request().requestId());
		assertThat(results.get(1).requestId()).isEqualTo(e2.request().requestId());
	}

	@Test
	void each_result_generates_up_to_max_tokens() {
		List<GenerationResult> results = loop.generateBatch(List.of(entry("a"), entry("b"), entry("c")));
		results.forEach(r -> assertThat(r.generatedTokens()).isEqualTo(3));
	}

	@Test
	void each_result_has_positive_prompt_token_count() {
		List<GenerationResult> results = loop.generateBatch(List.of(entry("hello world")));
		assertThat(results.get(0).promptTokens()).isGreaterThan(0);
	}

	@Test
	void stop_reason_is_max_tokens_when_no_eos() {
		List<GenerationResult> results = loop.generateBatch(List.of(entry("test")));
		assertThat(results.get(0).stopReason()).isEqualTo(GenerationResult.StopReason.MAX_TOKENS);
	}

	@Test
	void eos_stops_individual_request_early_without_affecting_others() {
		int eos = tokenizer.eosTokenId();

		InferenceRequest req1 = InferenceRequest.of("m", List.of(ChatMessage.user("stop early")),
				SamplingParams.defaults().withMaxTokens(5), RequestPriority.NORMAL);
		InferenceRequest req2 = InferenceRequest.of("m", List.of(ChatMessage.user("run full")),
				SamplingParams.defaults().withMaxTokens(3), RequestPriority.NORMAL);

		// Pipeline that returns EOS for req1 after the first generated token,
		// and token 42 for everything else.
		InferencePipeline earlyStopPipeline = new InferencePipeline() {
			final AtomicInteger req1Steps = new AtomicInteger(0);

			@Override
			public float[] forward(String requestId, int[] tokens, int startPos) {
				float[] logits = new float[1000];
				if (requestId.equals(req1.requestId()) && req1Steps.getAndIncrement() >= 1) {
					logits[eos] = 100.0f;
				} else {
					logits[42] = 100.0f;
				}
				return logits;
			}

			@Override
			public int vocabSize() {
				return 1000;
			}
		};

		GenerationLoop earlyStopLoop = new GenerationLoop(tokenizer, Sampler.create(), earlyStopPipeline,
				new KVCacheManager(new GpuKVCache(64 * 1024 * 1024), new CpuKVCache(1000)));

		List<GenerationResult> results = earlyStopLoop.generateBatch(
				List.of(new BatchEntry(req1, TokenConsumer.discard()), new BatchEntry(req2, TokenConsumer.discard())));

		assertThat(results.get(0).stopReason()).isEqualTo(GenerationResult.StopReason.EOS_TOKEN);
		assertThat(results.get(1).generatedTokens()).isEqualTo(3);
		assertThat(results.get(1).stopReason()).isEqualTo(GenerationResult.StopReason.MAX_TOKENS);
	}

	@Test
	void token_consumer_called_per_token_per_request() {
		List<Integer> tokensForReq1 = new CopyOnWriteArrayList<>();
		List<Integer> tokensForReq2 = new CopyOnWriteArrayList<>();

		TokenConsumer c1 = (piece, tokenId, pos) -> tokensForReq1.add(tokenId);
		TokenConsumer c2 = (piece, tokenId, pos) -> tokensForReq2.add(tokenId);

		BatchEntry e1 = new BatchEntry(InferenceRequest.of("m", List.of(ChatMessage.user("a")),
				SamplingParams.defaults().withMaxTokens(3), RequestPriority.NORMAL), c1);
		BatchEntry e2 = new BatchEntry(InferenceRequest.of("m", List.of(ChatMessage.user("b")),
				SamplingParams.defaults().withMaxTokens(4), RequestPriority.NORMAL), c2);

		loop.generateBatch(List.of(e1, e2));

		assertThat(tokensForReq1).hasSize(3);
		assertThat(tokensForReq2).hasSize(4);
	}

	@Test
	void forwardBatch_called_once_per_step_not_per_request() {
		AtomicInteger batchCallCount = new AtomicInteger(0);
		int maxTokens = 3;

		InferencePipeline countingPipeline = new InferencePipeline() {
			@Override
			public float[] forward(String requestId, int[] tokens, int startPos) {
				float[] logits = new float[1000];
				logits[42] = 100.0f;
				return logits;
			}

			@Override
			public float[][] forwardBatch(List<String> requestIds, List<int[]> allTokens,
					List<Integer> startPositions) {
				batchCallCount.incrementAndGet();
				float[][] results = new float[requestIds.size()][];
				for (int i = 0; i < requestIds.size(); i++) {
					results[i] = new float[1000];
					results[i][42] = 100.0f;
				}
				return results;
			}

			@Override
			public int vocabSize() {
				return 1000;
			}
		};

		GenerationLoop batchLoop = new GenerationLoop(tokenizer, Sampler.create(), countingPipeline,
				new KVCacheManager(new GpuKVCache(64 * 1024 * 1024), new CpuKVCache(1000)));

		int batchSize = 4;
		List<BatchEntry> entries = new ArrayList<>();
		for (int i = 0; i < batchSize; i++) {
			entries.add(new BatchEntry(
					InferenceRequest.of("m", List.of(ChatMessage.user("prompt " + i)),
							SamplingParams.defaults().withMaxTokens(maxTokens), RequestPriority.NORMAL),
					TokenConsumer.discard()));
		}

		batchLoop.generateBatch(entries);

		// forwardBatch called once per step (maxTokens), not once per request per step
		assertThat(batchCallCount.get()).isEqualTo(maxTokens);
	}

	@Test
	void latency_is_positive_for_all_results() {
		List<GenerationResult> results = loop.generateBatch(List.of(entry("a"), entry("b")));
		results.forEach(r -> assertThat(r.latency().toNanos()).isGreaterThan(0));
	}
}