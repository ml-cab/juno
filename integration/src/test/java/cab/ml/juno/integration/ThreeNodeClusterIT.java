package cab.ml.juno.integration;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;

import cab.ml.juno.coordinator.GenerationLoop;
import cab.ml.juno.coordinator.GenerationResult;
import cab.ml.juno.coordinator.InferenceRequest;
import cab.ml.juno.coordinator.RequestPriority;
import cab.ml.juno.coordinator.RequestScheduler;
import cab.ml.juno.coordinator.TokenConsumer;
import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.player.ClusterHarness;
import cab.ml.juno.player.EmbeddedNodeServer;
import cab.ml.juno.player.ProcessPipelineClient;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.StubTokenizer;

/**
 * Full multi-JVM 3-node cluster integration test.
 *
 * 3 separate JVM processes each running EmbeddedNodeServer (gRPC).
 * ProcessPipelineClient routes forward passes across them in pipeline order.
 * GenerationLoop + RequestScheduler run in this (coordinator) JVM.
 *
 * Memory budget for 16 GB host: 3 node JVMs × -Xmx4g = 12 GB coordinator JVM
 * -Xmx2g = 2 GB OS + overhead = 2 GB
 *
 * Run all ITs: mvn verify -pl integration Run only this: mvn verify -pl
 * integration -Dit.test=ThreeNodeClusterIT
 */
@DisplayName("Three-Node Cluster (3 forked JVMs)")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class ThreeNodeClusterIT {

	private static ClusterHarness harness;
	private static ProcessPipelineClient pipeline;
	private static GenerationLoop generationLoop;
	private static RequestScheduler scheduler;

	@BeforeAll
	static void startCluster() throws Exception {
		harness = ClusterHarness.threeNodes();
		harness.start();

		pipeline = harness.pipelineClient();

		generationLoop = new GenerationLoop(new StubTokenizer(), Sampler.create(), pipeline,
				new KVCacheManager(new GpuKVCache(512L * 1024 * 1024), new CpuKVCache(4096)));
		scheduler = new RequestScheduler(100, generationLoop);
	}

	@AfterAll
	static void stopCluster() throws Exception {
		if (harness != null)
			harness.stop();
	}

	// ── Tests ─────────────────────────────────────────────────────────────────

	@Test
	@Order(1)
	@DisplayName("All 3 node processes are alive after startup")
	void allNodesAlive() {
		assertThat(pipeline).isNotNull();
		assertThat(pipeline.vocabSize()).isEqualTo(EmbeddedNodeServer.VOCAB_SIZE);
	}

	// Order 2 is reserved — single-forward-pass with real weights is covered by
	// TinyLlamaLiveIT (requires the model file; skipped here since this class
	// uses stub CyclicForwardPassHandlers via the no-arg
	// ClusterHarness.threeNodes()).
	@Test
	@Order(3)
	@DisplayName("GenerationLoop generates tokens via gRPC pipeline")
	void generationLoopViaGrpc() {
		int maxTokens = 8;
		InferenceRequest request = InferenceRequest.of("tinyllama",
				List.of(ChatMessage.user("Write a haiku about distributed systems.")),
				SamplingParams.defaults().withMaxTokens(maxTokens), RequestPriority.NORMAL);

		List<String> pieces = new ArrayList<>();
		GenerationResult result = generationLoop.generate(request, (piece, tokenId, step) -> pieces.add(piece));

		assertThat(result.generatedTokens()).isGreaterThan(0).isLessThanOrEqualTo(maxTokens);

		assertThat(pieces).hasSameSizeAs(result.tokenIds());
		assertThat(result.latency()).isPositive();

		System.out.printf("Generated: \"%s\"  tokens=%d  latency=%d ms%n", result.text(), result.generatedTokens(),
				result.latency().toMillis());
	}

	@Test
	@Order(4)
	@DisplayName("Scheduler dispatches 4 concurrent requests via virtual threads")
	void schedulerConcurrentRequests() throws InterruptedException {
		int count = 4;
		List<GenerationResult> results = new CopyOnWriteArrayList<>();
		List<Thread> threads = new ArrayList<>();

		for (int i = 0; i < count; i++) {
			final String id = "sched-" + i;
			Thread t = Thread.ofVirtual().start(() -> {
				InferenceRequest req = InferenceRequest.of("tinyllama", List.of(ChatMessage.user("Request " + id)),
						SamplingParams.defaults().withMaxTokens(4), RequestPriority.NORMAL);
				results.add(scheduler.submitAndWait(req));
			});
			threads.add(t);
		}

		for (Thread t : threads)
			t.join(30_000);

		assertThat(results).hasSize(count);
		assertThat(results).allSatisfy(r -> assertThat(r.generatedTokens()).isGreaterThan(0));
	}

	@Test
	@Order(5)
	@DisplayName("Repeated identical prompts use prefix cache")
	void prefixCacheOnRepeat() {
		SamplingParams params = SamplingParams.defaults().withMaxTokens(5);
		List<ChatMessage> msgs = List.of(ChatMessage.user("What is pipeline parallelism?"));

		GenerationResult r1 = generationLoop.generate(
				InferenceRequest.of("tinyllama", msgs, params, RequestPriority.NORMAL), TokenConsumer.discard());
		GenerationResult r2 = generationLoop.generate(
				InferenceRequest.of("tinyllama", msgs, params, RequestPriority.NORMAL), TokenConsumer.discard());

		assertThat(r1.generatedTokens()).isEqualTo(r2.generatedTokens());
		assertThat(r1.promptTokens()).isEqualTo(r2.promptTokens());

		System.out.printf("Cold: %d ms  Warm: %d ms%n", r1.latency().toMillis(), r2.latency().toMillis());
	}

	@Test
	@Order(6)
	@DisplayName("HIGH priority request completes without starvation")
	void highPriorityCompletes() {
		InferenceRequest high = InferenceRequest.of("tinyllama", List.of(ChatMessage.user("Urgent query")),
				SamplingParams.defaults().withMaxTokens(2), RequestPriority.HIGH);

		GenerationResult result = scheduler.submitAndWait(high);
		assertThat(result.generatedTokens()).isGreaterThan(0);
	}

	// ── Activation compression tests ──────────────────────────────────────────

	@Test
	@Order(7)
	@DisplayName("FLOAT16 compressed pipeline produces same winner token as FLOAT32")
	void float16PipelineProducesSameWinnerToken() throws InterruptedException {
		ProcessPipelineClient f16Pipeline = new ProcessPipelineClient(harness.nodeAddresses(),
				EmbeddedNodeServer.VOCAB_SIZE, cab.ml.juno.node.ActivationDtype.FLOAT16);
		try {
			float[] logitsF32 = pipeline.forward("cmp-f32", new int[] { 1, 2, 3 }, 0);
			float[] logitsF16 = f16Pipeline.forward("cmp-f16", new int[] { 1, 2, 3 }, 0);

			assertThat(logitsF16).hasSize(EmbeddedNodeServer.VOCAB_SIZE);

			int winnerF32 = argmax(logitsF32);
			int winnerF16 = argmax(logitsF16);
			assertThat(winnerF16).as("FLOAT16 pipeline should pick the same winner token as FLOAT32")
					.isEqualTo(winnerF32);

			System.out.printf("FLOAT16 pipeline: winner=%d (same as FLOAT32=%d)%n", winnerF16, winnerF32);
		} finally {
			f16Pipeline.shutdown();
		}
	}

	@Test
	@Order(8)
	@DisplayName("INT8 compressed pipeline produces same winner token as FLOAT32")
	void int8PipelineProducesSameWinnerToken() throws InterruptedException {
		ProcessPipelineClient i8Pipeline = new ProcessPipelineClient(harness.nodeAddresses(),
				EmbeddedNodeServer.VOCAB_SIZE, cab.ml.juno.node.ActivationDtype.INT8);
		try {
			float[] logitsF32 = pipeline.forward("cmp-f32-2", new int[] { 1, 2, 3 }, 0);
			float[] logitsI8 = i8Pipeline.forward("cmp-i8", new int[] { 1, 2, 3 }, 0);

			assertThat(logitsI8).hasSize(EmbeddedNodeServer.VOCAB_SIZE);

			int winnerF32 = argmax(logitsF32);
			int winnerI8 = argmax(logitsI8);
			assertThat(winnerI8).as("INT8 pipeline should pick the same winner token as FLOAT32").isEqualTo(winnerF32);

			System.out.printf("INT8 pipeline: winner=%d (same as FLOAT32=%d)%n", winnerI8, winnerF32);
		} finally {
			i8Pipeline.shutdown();
		}
	}

	@Test
	@Order(9)
	@DisplayName("GenerationLoop produces tokens via FLOAT16 compressed pipeline")
	void generationLoopWithFloat16Compression() throws InterruptedException {
		ProcessPipelineClient f16Pipeline = new ProcessPipelineClient(harness.nodeAddresses(),
				EmbeddedNodeServer.VOCAB_SIZE, cab.ml.juno.node.ActivationDtype.FLOAT16);
		try {
			GenerationLoop f16Loop = new GenerationLoop(new StubTokenizer(), Sampler.create(), f16Pipeline,
					new KVCacheManager(new GpuKVCache(512L * 1024 * 1024), new CpuKVCache(256)));

			InferenceRequest req = InferenceRequest.of("tinyllama", List.of(ChatMessage.user("Hello compressed world")),
					SamplingParams.defaults().withMaxTokens(5), RequestPriority.NORMAL);

			GenerationResult result = f16Loop.generate(req, TokenConsumer.discard());

			assertThat(result.generatedTokens()).isGreaterThan(0);
			System.out.printf("FLOAT16 GenerationLoop: tokens=%d latency=%d ms%n", result.generatedTokens(),
					result.latency().toMillis());
		} finally {
			f16Pipeline.shutdown();
		}
	}

	// ── Helper ────────────────────────────────────────────────────────────────

	private static int argmax(float[] logits) {
		int best = 0;
		for (int i = 1; i < logits.length; i++) {
			if (logits[i] > logits[best])
				best = i;
		}
		return best;
	}
}