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
import cab.ml.juno.node.PoolingMode;
import cab.ml.juno.registry.ModelDescriptor;
import cab.ml.juno.registry.ModelRegistry;
import cab.ml.juno.registry.NodeDescriptor;
import cab.ml.juno.registry.NodeStatus;
import cab.ml.juno.registry.QuantizationType;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.tokenizer.SimpleTokenizer;

/**
 * {@link StubInferencePipeline} does not override {@link cab.ml.juno.node.InferencePipeline#embedTokens}
 * — it stands in for the gRPC / tensor-parallel / pipeline-parallel pipelines
 * used by {@code cluster} mode, none of which implement embeddings extraction
 * in v1 (see {@code docs/infra-plan/PLAN-Infra-Tier11.md}). Even with
 * {@code --embeddings} enabled, such a pipeline must fail closed with a clear
 * 400 rather than a bare 500 or a silently wrong vector.
 */
class EmbeddingsUnsupportedPipelineTest {

	private static final int PORT = 28192;
	private static final String BASE = "http://localhost:" + PORT + "/v1";

	private static InferenceApiServer server;
	private static final HttpClient http = HttpClient.newHttpClient();

	@BeforeAll
	static void startServer() {
		var pipeline = new StubInferencePipeline();
		var kvCache = new KVCacheManager(new GpuKVCache(64L * 1024 * 1024), new CpuKVCache(64));
		var loop = new GenerationLoop(new SimpleTokenizer(), Sampler.create(), pipeline, kvCache);
		var scheduler = new RequestScheduler(64, loop);

		var registry = new ModelRegistry(cab.ml.juno.registry.ShardPlanner.create());
		ModelDescriptor model = ModelDescriptor.of("tinyllama", "llama", 4, 8, StubInferencePipeline.VOCAB_SIZE, 2,
				QuantizationType.Q4_K_M, "/models/tinyllama.gguf");
		registry.register(model,
				List.of(new NodeDescriptor("n1", "localhost", 9092, 4L << 30, 4L << 30, NodeStatus.READY, 1.0,
						Instant.now(), Instant.now())));
		registry.markLoaded("tinyllama");

		server = new InferenceApiServer(scheduler, registry, "BE", null, true, PoolingMode.MEAN);
		server.start(PORT);
	}

	@AfterAll
	static void stopServer() {
		if (server != null)
			server.stop();
	}

	@Test
	void unsupported_pipeline_fails_closed_with_400() throws Exception {
		var request = HttpRequest.newBuilder().uri(URI.create(BASE + "/embeddings"))
				.header("Content-Type", "application/json")
				.POST(HttpRequest.BodyPublishers.ofString("""
						{"input": "hello"}
						""")).build();
		HttpResponse<String> response = http.send(request, HttpResponse.BodyHandlers.ofString());

		assertThat(response.statusCode()).isEqualTo(400);
		assertThat(response.body()).contains("not supported");
	}
}
