package cab.ml.juno.integration;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

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
import cab.ml.juno.node.InferencePipeline;
import cab.ml.juno.player.ClusterHarness;
import cab.ml.juno.player.EmbeddedNodeServer;
import cab.ml.juno.registry.ParallelismType;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.SimpleTokenizer;

/**
 * Full multi-JVM 3-node tensor-parallel cluster integration test.
 *
 * Topology: 3 separate JVM processes each running EmbeddedNodeServer (gRPC).
 * Every node holds ALL transformer layers [0, 22) with hasEmbeddings=true,
 * hasOutputProjection=true and a unique tensorRank in {0, 1, 2}.
 *
 * Forward pass per decode step: Coordinator broadcasts the token sequence to
 * all 3 nodes simultaneously. Each node (CyclicForwardPassHandler) sees
 * hasOutputProjection=true and returns logits[42]=100.0f in FLOAT32.
 * TensorParallelPipelineClient sums the 3 partial logit vectors: logits[42] =
 * 300.0f — token 42 is still the argmax. GenerationLoop samples token 42 each
 * step.
 *
 * Compare with ThreeNodeClusterIT (pipeline parallel): Activation flows node-1
 * → node-2 → node-3 serially. This IT fans out to all 3 nodes in parallel and
 * reduces.
 *
 * Run: mvn verify -pl integration -Dit.test=TensorParallelClusterIT
 */
@DisplayName("Tensor-Parallel 3-Node Cluster (3 forked JVMs)")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class TensorParallelClusterIT {

	private static ClusterHarness harness;
	private static InferencePipeline pipeline;
	private static GenerationLoop generationLoop;
	private static RequestScheduler scheduler;

	@BeforeAll
	static void startCluster() throws Exception {
		harness = ClusterHarness.tensorNodes(); // tensor-parallel, stub mode
		harness.start();

		// pipeline() returns TensorParallelPipelineClient for TENSOR mode
		pipeline = harness.pipeline();

		generationLoop = new GenerationLoop(new SimpleTokenizer(), Sampler.create(), pipeline,
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
	@DisplayName("cluster reports TENSOR parallelism type")
	void cluster_is_tensor_parallel() {
		assertThat(harness.parallelismType()).isEqualTo(ParallelismType.TENSOR);
	}

	@Test
	@Order(2)
	@DisplayName("all 3 node processes alive and pipeline has correct vocab size")
	void all_nodes_alive_after_startup() {
		assertThat(pipeline).isNotNull();
		assertThat(pipeline.vocabSize()).isEqualTo(EmbeddedNodeServer.VOCAB_SIZE);
	}

	@Test
	@Order(3)
	@DisplayName("GenerationLoop generates tokens via tensor-parallel gRPC (AllReduce)")
	void generation_loop_via_tensor_parallel_grpc() {
		int maxTokens = 8;
		InferenceRequest request = InferenceRequest.of("tinyllama",
				List.of(ChatMessage.user("Explain tensor parallelism in one sentence.")),
				SamplingParams.defaults().withMaxTokens(maxTokens), RequestPriority.NORMAL);

		List<String> pieces = new ArrayList<>();
		GenerationResult result = generationLoop.generate(request, (piece, tokenId, step) -> pieces.add(piece));

		assertThat(result.generatedTokens()).as("at least one token generated").isGreaterThan(0)
				.isLessThanOrEqualTo(maxTokens);
		assertThat(pieces).isNotEmpty();
	}

	@Test
	@Order(4)
	@DisplayName("concurrent requests run in parallel through tensor-parallel pipeline")
	void concurrent_requests_via_tensor_parallel() throws Exception {
		int concurrency = 4;
		int maxTokens = 5;

		List<CompletableFuture<GenerationResult>> futures = new ArrayList<>();
		for (int i = 0; i < concurrency; i++) {
			InferenceRequest req = InferenceRequest.of("tinyllama", List.of(ChatMessage.user("Request " + i)),
					SamplingParams.defaults().withMaxTokens(maxTokens), RequestPriority.NORMAL);
			futures.add(scheduler.submit(req, TokenConsumer.discard()));
		}

		CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).get(30,
				java.util.concurrent.TimeUnit.SECONDS);

		for (int i = 0; i < concurrency; i++) {
			GenerationResult r = futures.get(i).get();
			assertThat(r.generatedTokens()).as("request " + i + " generated tokens").isGreaterThan(0);
		}
	}

	@Test
	@Order(5)
	@DisplayName("tensor-parallel pipeline vocabSize matches EmbeddedNodeServer constant")
	void tensor_parallel_vocab_size_matches_constant() {
		assertThat(pipeline.vocabSize()).isEqualTo(EmbeddedNodeServer.VOCAB_SIZE);
	}
}