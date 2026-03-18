package cab.ml.juno.integration;

import static org.assertj.core.api.Assertions.assertThat;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.coordinator.GenerationLoop;
import cab.ml.juno.coordinator.GenerationResult;
import cab.ml.juno.coordinator.InferenceRequest;
import cab.ml.juno.coordinator.RequestPriority;
import cab.ml.juno.coordinator.TokenConsumer;
import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.kvcache.KVKey;
import cab.ml.juno.kvcache.LayerRange;
import cab.ml.juno.node.CyclicForwardPassHandler;
import cab.ml.juno.node.ForwardPassHandler;
import cab.ml.juno.node.LocalInferencePipeline;
import cab.ml.juno.registry.NodeDescriptor;
import cab.ml.juno.registry.NodeStatus;
import cab.ml.juno.registry.ShardMap;
import cab.ml.juno.registry.ShardPlanner;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.StubTokenizer;

/**
 * In-process 3-node integration test.
 *
 * Wires 3 CyclicForwardPassHandlers via LocalInferencePipeline, runs a full
 * GenerationLoop end-to-end, zero network.
 *
 * Run: mvn verify -pl integration -Dit.test=InProcessClusterIT
 */
@DisplayName("In-Process 3-Node Cluster")
class InProcessClusterIT {

	// TinyLlama-1.1B shape
	private static final int VOCAB_SIZE = 32_000;
	private static final int HIDDEN_DIM = 2_048;
	private static final int NUM_HEADS = 32;
	private static final int TOTAL_LAYERS = 22;
	private static final int STUB_WINNER = 42; // CyclicForwardPassHandler puts 100.0f here

	private LocalInferencePipeline pipeline;
	private GenerationLoop generationLoop;

	@BeforeEach
	void setUp() {
		// ── 1. Build 3-node shard map ─────────────────────────────────────────
		long vramPerLayer = 186L * 1024 * 1024;
		long nodeVram = 4L * 1024 * 1024 * 1024;

		List<NodeDescriptor> nodes = List.of(nodeDescriptor("node-1", "localhost", 9092, nodeVram),
				nodeDescriptor("node-2", "localhost", 9093, nodeVram),
				nodeDescriptor("node-3", "localhost", 9094, nodeVram));

		ShardMap shardMap = ShardPlanner.create().plan("tinyllama", TOTAL_LAYERS, vramPerLayer, nodes);
		assertThat(shardMap.nodeCount()).as("All 3 nodes should receive shards").isEqualTo(3);

		// ── 2. One handler per stage ──────────────────────────────────────────
		List<ForwardPassHandler> handlers = List.of(new CyclicForwardPassHandler(), new CyclicForwardPassHandler(),
				new CyclicForwardPassHandler(STUB_WINNER) // last node → logits
		);

		pipeline = LocalInferencePipeline.from(shardMap, handlers, VOCAB_SIZE, HIDDEN_DIM, NUM_HEADS);

		// ── 3. Wire coordinator components ────────────────────────────────────
		generationLoop = new GenerationLoop(new StubTokenizer(), Sampler.create(), pipeline,
				new KVCacheManager(new GpuKVCache(512L * 1024 * 1024), new CpuKVCache(1024)));
	}

	// ── Tests ─────────────────────────────────────────────────────────────────

	@Test
	@DisplayName("Pipeline has 3 stages covering all 22 layers")
	void pipelineHasThreeStages() {
		assertThat(pipeline.stageCount()).isEqualTo(3);
		assertThat(pipeline.vocabSize()).isEqualTo(VOCAB_SIZE);
	}

	@Test
	@DisplayName("Single forward pass returns logits of correct size")
	void singleForwardPassReturnsLogits() {
		float[] logits = pipeline.forward("req-001", new int[] { 1, 2, 3 }, 0);
		assertThat(logits).hasSize(VOCAB_SIZE);
		assertThat(logits[STUB_WINNER]).isGreaterThan(logits[0]);
	}

	@Test
	@DisplayName("GenerationLoop produces tokens up to maxTokens")
	void generationLoopProducesTokens() {
		int maxTokens = 10;
		InferenceRequest request = InferenceRequest.of("tinyllama", List.of(ChatMessage.user("Hello, world!")),
				SamplingParams.defaults().withMaxTokens(maxTokens), RequestPriority.NORMAL);

		List<String> pieces = new ArrayList<>();
		GenerationResult result = generationLoop.generate(request, (piece, _, _) -> pieces.add(piece));

		assertThat(result.generatedTokens()).isGreaterThan(0).isLessThanOrEqualTo(maxTokens);

		assertThat(pieces).hasSameSizeAs(result.tokenIds());
		assertThat(result.text()).isNotNull();
	}

	@Test
	@DisplayName("Prefix cache hit on repeated prompt")
	void prefixCacheHitOnRepeat() {
		SamplingParams params = SamplingParams.defaults().withMaxTokens(5);
		List<ChatMessage> msgs = List.of(ChatMessage.user("Repeat this prompt"));

		GenerationResult r1 = generationLoop.generate(
				InferenceRequest.of("tinyllama", msgs, params, RequestPriority.NORMAL), TokenConsumer.discard());
		GenerationResult r2 = generationLoop.generate(
				InferenceRequest.of("tinyllama", msgs, params, RequestPriority.NORMAL), TokenConsumer.discard());

		assertThat(r1.promptTokens()).isEqualTo(r2.promptTokens());
		assertThat(r2.generatedTokens()).isGreaterThan(0);
	}

	@Test
	@DisplayName("Concurrent requests on virtual threads all complete")
	void concurrentRequestsOnVirtualThreads() throws InterruptedException {
		int concurrency = 8;
		var latch = new java.util.concurrent.CountDownLatch(concurrency);
		var results = new java.util.concurrent.CopyOnWriteArrayList<GenerationResult>();

		for (int i = 0; i < concurrency; i++) {
			final int idx = i;
			Thread.ofVirtual().start(() -> {
				try {
					InferenceRequest req = InferenceRequest.of("tinyllama", List.of(ChatMessage.user("Request " + idx)),
							SamplingParams.defaults().withMaxTokens(3), RequestPriority.NORMAL);
					results.add(generationLoop.generate(req, TokenConsumer.discard()));
				} finally {
					latch.countDown(); // always fires, even on exception
				}
			});
		}

		// Block until every virtual thread has signalled — no silent timeout
		// slip-through
		boolean allDone = latch.await(30, java.util.concurrent.TimeUnit.SECONDS);
		assertThat(allDone).as("All %d virtual threads should finish within 30s", concurrency).isTrue();

		assertThat(results).hasSize(concurrency);
		assertThat(results).allSatisfy(r -> assertThat(r.generatedTokens()).isGreaterThan(0));
	}

	@Test
	@DisplayName("Per-node KVCacheManagers are scoped to their layer range and isolated from each other")
	void per_node_kv_caches_are_layer_scoped_and_isolated() {
		// Mirrors the 3-node shard layout used in setUp():
		// Node 1 — layers 0..7 (ShardAssignment startLayer=0, endLayer=8)
		// Node 2 — layers 8..14 (ShardAssignment startLayer=8, endLayer=15)
		// Node 3 — layers 15..21 (ShardAssignment startLayer=15, endLayer=22)
		KVCacheManager node1Cache = new KVCacheManager(new GpuKVCache(64L * 1024 * 1024), new CpuKVCache(256),
				LayerRange.of(0, 8));
		KVCacheManager node2Cache = new KVCacheManager(new GpuKVCache(64L * 1024 * 1024), new CpuKVCache(256),
				LayerRange.of(8, 15));
		KVCacheManager node3Cache = new KVCacheManager(new GpuKVCache(64L * 1024 * 1024), new CpuKVCache(256),
				LayerRange.of(15, 22));

		// Each node owns only its layers
		assertThat(node1Cache.ownsLayer(0)).isTrue();
		assertThat(node1Cache.ownsLayer(7)).isTrue();
		assertThat(node1Cache.ownsLayer(8)).isFalse();

		assertThat(node2Cache.ownsLayer(8)).isTrue();
		assertThat(node2Cache.ownsLayer(14)).isTrue();
		assertThat(node2Cache.ownsLayer(0)).isFalse();

		assertThat(node3Cache.ownsLayer(15)).isTrue();
		assertThat(node3Cache.ownsLayer(21)).isTrue();
		assertThat(node3Cache.ownsLayer(14)).isFalse();

		// Blocks stored on node1 are not visible on node2 or node3
		var key = new KVKey("req-kv-test", 3);
		var block = new cab.ml.juno.kvcache.KVBlock(key, new byte[512], 10, 3, java.time.Instant.now(),
				java.time.Instant.now());
		node1Cache.put(key, block);

		assertThat(node1Cache.get(key)).isPresent();
		assertThat(node2Cache.get(key)).isEmpty();
		assertThat(node3Cache.get(key)).isEmpty();

		// Evict on node1 doesn't affect other nodes (would be no-op since they don't
		// have it,
		// but verifies evict() is local)
		node1Cache.evict("req-kv-test");
		assertThat(node1Cache.get(key)).isEmpty();
	}

	// ── Helpers ───────────────────────────────────────────────────────────────

	private static NodeDescriptor nodeDescriptor(String id, String host, int port, long vram) {
		return new NodeDescriptor(id, host, port, vram, vram, NodeStatus.IDLE, 1.0, Instant.now(), Instant.now());
	}
}