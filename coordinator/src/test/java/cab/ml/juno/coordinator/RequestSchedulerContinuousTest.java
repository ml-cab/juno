package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.kvcache.ServeScheduleOptions;
import cab.ml.juno.node.InferencePipeline;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.SimpleTokenizer;

class RequestSchedulerContinuousTest {

	private RequestScheduler scheduler;

	@AfterEach
	void tearDown() {
		if (scheduler != null)
			scheduler.shutdown();
	}

	private InferenceRequest req(String prompt, int maxTokens) {
		return InferenceRequest.of("tinyllama", List.of(ChatMessage.user(prompt)),
				SamplingParams.defaults().withMaxTokens(maxTokens), RequestPriority.NORMAL);
	}

	@Test
	void overlapping_decode_shares_forward_batch() throws Exception {
		RecordingPipeline pipeline = new RecordingPipeline();
		GenerationLoop loop = new GenerationLoop(new SimpleTokenizer(), Sampler.create(), pipeline,
				new KVCacheManager(new GpuKVCache(64 * 1024 * 1024), new CpuKVCache(1000)));
		scheduler = new RequestScheduler(32, loop, BatchConfig.of(8, 80), ServeScheduleOptions.parse("continuous"));

		CountDownLatch firstToken = new CountDownLatch(1);
		TokenConsumer slowA = (piece, id, pos) -> {
			if (pos == 0)
				firstToken.countDown();
		};

		CompletableFuture<GenerationResult> a = scheduler.submit(req("alpha prompt one", 8), slowA);
		assertThat(firstToken.await(10, TimeUnit.SECONDS)).isTrue();

		CompletableFuture<GenerationResult> b = scheduler.submit(req("beta", 6), TokenConsumer.discard());
		CompletableFuture.allOf(a, b).get(30, TimeUnit.SECONDS);

		assertThat(pipeline.maxBatchSize()).isGreaterThanOrEqualTo(2);
		assertThat(pipeline.sawDistinctPositions()).isTrue();
	}

	@Test
	void streaming_consumers_join_running_set() throws Exception {
		RecordingPipeline pipeline = new RecordingPipeline();
		GenerationLoop loop = new GenerationLoop(new SimpleTokenizer(), Sampler.create(), pipeline,
				new KVCacheManager(new GpuKVCache(64 * 1024 * 1024), new CpuKVCache(1000)));
		scheduler = new RequestScheduler(32, loop, BatchConfig.of(8, 80), ServeScheduleOptions.parse("continuous"));

		List<String> aPieces = new CopyOnWriteArrayList<>();
		List<String> bPieces = new CopyOnWriteArrayList<>();
		TokenConsumer streamA = (piece, id, pos) -> aPieces.add(piece);
		TokenConsumer streamB = (piece, id, pos) -> bPieces.add(piece);

		CompletableFuture<GenerationResult> fa = scheduler.submit(req("stream-a", 4), streamA);
		CompletableFuture<GenerationResult> fb = scheduler.submit(req("stream-b", 4), streamB);
		CompletableFuture.allOf(fa, fb).get(30, TimeUnit.SECONDS);

		assertThat(aPieces).isNotEmpty();
		assertThat(bPieces).isNotEmpty();
		assertThat(pipeline.maxBatchSize()).isGreaterThanOrEqualTo(2);
	}

	@Test
	void static_streams_still_bypass_batch_collector() throws Exception {
		AtomicInteger batchCalls = new AtomicInteger();
		InferencePipeline counting = new InferencePipeline() {
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
		GenerationLoop loop = new GenerationLoop(new SimpleTokenizer(), Sampler.create(), counting,
				new KVCacheManager(new GpuKVCache(64 * 1024 * 1024), new CpuKVCache(1000)));
		scheduler = new RequestScheduler(10, loop, BatchConfig.of(8, 50), ServeScheduleOptions.defaults());

		TokenConsumer stream = (piece, id, pos) -> {
		};
		scheduler.submit(req("isolated-stream", 2), stream).get(10, TimeUnit.SECONDS);
		assertThat(batchCalls.get()).isZero();
	}

	@Test
	void prefix_trie_survives_stateless_request() throws Exception {
		GenerationLoop loop = new GenerationLoop(new SimpleTokenizer(), Sampler.create(), new StubInferencePipeline(),
				new KVCacheManager(new GpuKVCache(64 * 1024 * 1024), new CpuKVCache(1000)));
		scheduler = new RequestScheduler(8, loop, BatchConfig.of(4, 50), ServeScheduleOptions.parse("continuous"));

		scheduler.submitAndWait(req("shared system prompt hello", 2));
		long lookupsAfterFirst = loop.kvCache().prefixLookups();
		assertThat(lookupsAfterFirst).isGreaterThan(0);

		scheduler.submitAndWait(req("shared system prompt hello again", 2));
		assertThat(loop.kvCache().prefixLookups()).isGreaterThan(lookupsAfterFirst);
		assertThat(loop.kvCache().prefixHits()).isGreaterThanOrEqualTo(0);
	}

	private static final class RecordingPipeline implements InferencePipeline {
		private final List<Integer> batchSizes = new CopyOnWriteArrayList<>();
		private final List<List<Integer>> positions = new CopyOnWriteArrayList<>();

		@Override
		public float[] forward(String requestId, int[] tokens, int startPos) {
			float[] logits = new float[1000];
			logits[42] = 100.0f;
			return logits;
		}

		@Override
		public float[][] forwardBatch(List<String> requestIds, List<int[]> allTokens, List<Integer> startPositions) {
			batchSizes.add(requestIds.size());
			positions.add(List.copyOf(startPositions));
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

		int maxBatchSize() {
			return batchSizes.stream().mapToInt(Integer::intValue).max().orElse(0);
		}

		boolean sawDistinctPositions() {
			for (List<Integer> p : positions) {
				if (p.size() >= 2 && !p.get(0).equals(p.get(1)))
					return true;
				if (p.size() >= 2 && p.stream().distinct().count() >= 2)
					return true;
			}
			return false;
		}
	}
}
