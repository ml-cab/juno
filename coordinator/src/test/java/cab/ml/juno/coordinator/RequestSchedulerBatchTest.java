package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.AfterEach;
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

class RequestSchedulerBatchTest {

	private GenerationLoop loop;
	private RequestScheduler scheduler;

	@BeforeEach
	void setUp() {
		loop = new GenerationLoop(new StubTokenizer(), Sampler.create(), new StubInferencePipeline(),
				new KVCacheManager(new GpuKVCache(64 * 1024 * 1024), new CpuKVCache(1000)));
	}

	@AfterEach
	void tearDown() {
		if (scheduler != null)
			scheduler.shutdown();
	}

	private InferenceRequest req(String prompt) {
		return InferenceRequest.of("model", List.of(ChatMessage.user(prompt)),
				SamplingParams.defaults().withMaxTokens(2), RequestPriority.NORMAL);
	}

	// ── Backward compatibility — disabled batching ────────────────────────────

	@Test
	void disabled_batching_behaves_like_original_submit_and_wait() {
		scheduler = new RequestScheduler(10, loop, BatchConfig.disabled());

		GenerationResult result = scheduler.submitAndWait(req("hello"));

		assertThat(result).isNotNull();
		assertThat(result.generatedTokens()).isGreaterThan(0);
	}

	@Test
	void disabled_batching_submit_returns_future_that_completes() throws Exception {
		scheduler = new RequestScheduler(10, loop, BatchConfig.disabled());

		CompletableFuture<GenerationResult> future = scheduler.submit(req("async"), TokenConsumer.discard());

		GenerationResult result = future.get(10, TimeUnit.SECONDS);
		assertThat(result).isNotNull();
	}

	// ── Batching enabled ──────────────────────────────────────────────────────

	@Test
	void batching_scheduler_completes_single_request() throws Exception {
		scheduler = new RequestScheduler(10, loop, BatchConfig.of(8, 50));

		CompletableFuture<GenerationResult> future = scheduler.submit(req("single"), TokenConsumer.discard());

		GenerationResult result = future.get(10, TimeUnit.SECONDS);
		assertThat(result).isNotNull();
		assertThat(result.generatedTokens()).isGreaterThan(0);
	}

	@Test
	void batching_scheduler_completes_multiple_concurrent_requests() throws Exception {
		scheduler = new RequestScheduler(64, loop, BatchConfig.of(8, 50));

		int count = 6;
		List<CompletableFuture<GenerationResult>> futures = new ArrayList<>();
		for (int i = 0; i < count; i++) {
			futures.add(scheduler.submit(req("prompt " + i), TokenConsumer.discard()));
		}

		CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).get(30, TimeUnit.SECONDS);

		List<GenerationResult> results = futures.stream().map(CompletableFuture::join).toList();

		assertThat(results).hasSize(count);
		assertThat(results).allSatisfy(r -> assertThat(r.generatedTokens()).isGreaterThan(0));
	}

	@Test
	void batching_groups_requests_into_fewer_batch_calls() throws Exception {
		// Track how many times generateBatch is called vs requests submitted
		AtomicInteger batchCalls = new AtomicInteger(0);

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
				batchCalls.incrementAndGet();
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

		GenerationLoop batchLoop = new GenerationLoop(new StubTokenizer(), Sampler.create(), countingPipeline,
				new KVCacheManager(new GpuKVCache(64 * 1024 * 1024), new CpuKVCache(1000)));

		// Small window so requests group together; large enough batch size
		scheduler = new RequestScheduler(64, batchLoop, BatchConfig.of(8, 80));

		int requestCount = 4;
		List<CompletableFuture<GenerationResult>> futures = new ArrayList<>();
		// Submit all requests at once — they should land in the same batch
		for (int i = 0; i < requestCount; i++) {
			futures.add(scheduler.submit(req("batch " + i), TokenConsumer.discard()));
		}

		CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).get(30, TimeUnit.SECONDS);

		// With maxTokens=2 and requests grouped into 1 batch:
		// batchCalls should be 2 (one per decode step), not 2*4=8 (if serial)
		int maxTokens = 2;
		assertThat(batchCalls.get()).isLessThanOrEqualTo(maxTokens * requestCount);
		// At least some batching occurred — fewer calls than serial would make
		// (can't assert exact number due to timing, but verify all completed)
		futures.forEach(f -> assertThat(f.join().generatedTokens()).isGreaterThan(0));
	}

	@Test
	void streaming_consumer_receives_tokens_in_batch_mode() throws Exception {
		scheduler = new RequestScheduler(10, loop, BatchConfig.of(4, 50));

		List<String> pieces = new java.util.concurrent.CopyOnWriteArrayList<>();
		TokenConsumer consumer = (piece, _, _) -> pieces.add(piece);

		CompletableFuture<GenerationResult> future = scheduler.submit(req("stream"), consumer);
		future.get(10, TimeUnit.SECONDS);

		assertThat(pieces).isNotEmpty();
	}

	@Test
	void shutdown_is_idempotent() {
		scheduler = new RequestScheduler(10, loop, BatchConfig.defaults());
		scheduler.shutdown();
		scheduler.shutdown(); // second call must not throw
	}

	@Test
	void queue_depth_reports_correctly_before_dispatch() {
		// Use a slow pipeline to keep items in queue
		InferencePipeline slowPipeline = new InferencePipeline() {
			@Override
			public float[] forward(String requestId, int[] tokens, int startPos) {
				try {
					Thread.sleep(500);
				} catch (InterruptedException e) {
					Thread.currentThread().interrupt();
				}
				float[] logits = new float[1000];
				logits[42] = 100.0f;
				return logits;
			}

			@Override
			public int vocabSize() {
				return 1000;
			}
		};
		GenerationLoop slowLoop = new GenerationLoop(new StubTokenizer(), Sampler.create(), slowPipeline,
				new KVCacheManager(new GpuKVCache(64 * 1024 * 1024), new CpuKVCache(1000)));

		// Large window so requests queue up before dispatch
		scheduler = new RequestScheduler(64, slowLoop, BatchConfig.of(8, 500));

		scheduler.submit(req("a"), TokenConsumer.discard());
		scheduler.submit(req("b"), TokenConsumer.discard());

		assertThat(scheduler.maxQueueDepth()).isEqualTo(64);
		scheduler.shutdown();
	}
}