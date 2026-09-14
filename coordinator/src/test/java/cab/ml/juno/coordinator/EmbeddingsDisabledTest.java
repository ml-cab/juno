package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Instant;
import java.util.List;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.node.LocalInferencePipeline;
import cab.ml.juno.registry.ModelDescriptor;
import cab.ml.juno.registry.ModelRegistry;
import cab.ml.juno.registry.NodeDescriptor;
import cab.ml.juno.registry.NodeStatus;
import cab.ml.juno.registry.QuantizationType;
import cab.ml.juno.registry.ShardAssignment;
import cab.ml.juno.registry.ShardMap;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.tokenizer.SimpleTokenizer;

/**
 * {@code POST /v1/embeddings} is opt-in: a server started without
 * {@code --embeddings} (the {@link InferenceApiServer} 3/4-arg constructors,
 * used by every caller before this tier) must fail closed with a clear 400
 * rather than silently no-op or 404 — see ROADMAP Execution rule 6.
 */
class EmbeddingsDisabledTest {

	private static final int PORT = 28191;
	private static final String BASE = "http://localhost:" + PORT + "/v1";

	private static InferenceApiServer server;
	private static final HttpClient http = HttpClient.newHttpClient();

	@BeforeAll
	static void startServer() {
		int hiddenDim = 8;
		ShardAssignment assignment = new ShardAssignment("n1", "localhost", 9091, 0, 4, true, true);
		ShardMap shardMap = new ShardMap("model", 4, List.of(assignment), Instant.now());
		var pipeline = LocalInferencePipeline.from(shardMap, new PositionValueForwardPassHandler(), 100, hiddenDim,
				2);

		var kvCache = new KVCacheManager(new GpuKVCache(64L * 1024 * 1024), new CpuKVCache(64));
		var loop = new GenerationLoop(new SimpleTokenizer(), Sampler.create(), pipeline, kvCache);
		var scheduler = new RequestScheduler(64, loop);

		var registry = new ModelRegistry(cab.ml.juno.registry.ShardPlanner.create());
		ModelDescriptor model = ModelDescriptor.of("tinyllama", "llama", 4, hiddenDim, 100, 2,
				QuantizationType.Q4_K_M, "/models/tinyllama.gguf");
		registry.register(model,
				List.of(new NodeDescriptor("n1", "localhost", 9092, 4L << 30, 4L << 30, NodeStatus.READY, 1.0,
						Instant.now(), Instant.now())));
		registry.markLoaded("tinyllama");

		// 4-arg constructor — the one every caller used before --embeddings existed.
		server = new InferenceApiServer(scheduler, registry, "BE", null);
		server.start(PORT);
	}

	@AfterAll
	static void stopServer() {
		if (server != null)
			server.stop();
	}

	@Test
	void embeddings_route_fails_closed_when_not_enabled() throws Exception {
		var request = HttpRequest.newBuilder().uri(URI.create(BASE + "/embeddings"))
				.header("Content-Type", "application/json")
				.POST(HttpRequest.BodyPublishers.ofString("""
						{"input": "hello"}
						""")).build();
		HttpResponse<String> response = http.send(request, HttpResponse.BodyHandlers.ofString());

		assertThat(response.statusCode()).isEqualTo(400);
		assertThat(response.body()).contains("--embeddings");
	}
}
