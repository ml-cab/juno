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
import cab.ml.juno.node.CyclicForwardPassHandler;
import cab.ml.juno.node.ForwardPassHandler;
import cab.ml.juno.node.LocalInferencePipeline;
import cab.ml.juno.registry.ModelDescriptor;
import cab.ml.juno.registry.ModelRegistry;
import cab.ml.juno.registry.NodeDescriptor;
import cab.ml.juno.registry.NodeStatus;
import cab.ml.juno.registry.QuantizationType;
import cab.ml.juno.registry.ShardMap;
import cab.ml.juno.registry.ShardPlanner;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.tokenizer.StubTokenizer;

class InferenceApiServerTest {

	private static final int PORT = 28080;
	private static final String BASE = "http://localhost:" + PORT + "/v1";

	private static InferenceApiServer server;
	private static final HttpClient http = HttpClient.newHttpClient();

	// ── Constants matching TinyLlama stub ─────────────────────────────────────
	private static final int VOCAB_SIZE = 32_000;
	private static final int HIDDEN_DIM = 2_048;
	private static final int NUM_HEADS = 32;
	private static final int TOTAL_LAYERS = 22;
	private static final long GB = 1024L * 1024 * 1024;

	@BeforeAll
	static void startServer() {
		// Build stub cluster
		long vramPerLayer = 186L * 1024 * 1024;
		List<NodeDescriptor> nodes = List.of(node("n1", 4 * GB), node("n2", 4 * GB), node("n3", 4 * GB));

		ShardMap shardMap = ShardPlanner.create().plan("tinyllama", TOTAL_LAYERS, vramPerLayer, nodes);

		List<ForwardPassHandler> handlers = List.of(new CyclicForwardPassHandler(), new CyclicForwardPassHandler(),
				new CyclicForwardPassHandler(42));
		var pipeline = LocalInferencePipeline.from(shardMap, handlers, VOCAB_SIZE, HIDDEN_DIM, NUM_HEADS);

		var kvCache = new KVCacheManager(new GpuKVCache(128L * 1024 * 1024), new CpuKVCache(256));

		var loop = new GenerationLoop(new StubTokenizer(), Sampler.create(), pipeline, kvCache);
		var scheduler = new RequestScheduler(64, loop);

		// Build a registry with tinyllama already loaded
		var registry = new ModelRegistry(ShardPlanner.create());
		ModelDescriptor tinyllama = ModelDescriptor.of("tinyllama", "llama", TOTAL_LAYERS, HIDDEN_DIM, VOCAB_SIZE,
				NUM_HEADS, QuantizationType.Q4_K_M, "/models/tinyllama.gguf");
		registry.register(tinyllama, nodes);
		registry.markLoaded("tinyllama");

		server = new InferenceApiServer(scheduler, registry);
		server.start(PORT);
	}

	@AfterAll
	static void stopServer() {
		if (server != null)
			server.stop();
	}

	// ── POST /v1/inference ────────────────────────────────────────────────────

	@Test
	void blocking_inference_returns_200_with_text() throws Exception {
		var response = post("/inference", """
				{
				  "modelId": "tinyllama",
				  "messages": [{"role":"user","content":"Hello"}],
				  "sampling": {"maxTokens": 3}
				}
				""");

		assertThat(response.statusCode()).isEqualTo(200);
		assertThat(response.body()).contains("requestId");
		assertThat(response.body()).contains("text");
		assertThat(response.body()).contains("tokenCount");
		assertThat(response.body()).contains("finishReason");
	}

	@Test
	void blocking_inference_without_model_id_uses_default() throws Exception {
		// no modelId — server picks first loaded model
		var response = post("/inference", """
				{
				  "messages": [{"role":"user","content":"Hi"}],
				  "sampling": {"maxTokens": 2}
				}
				""");

		assertThat(response.statusCode()).isEqualTo(200);
	}

	@Test
	void blocking_inference_empty_messages_returns_400() throws Exception {
		var response = post("/inference", """
				{
				  "modelId": "tinyllama",
				  "messages": []
				}
				""");

		assertThat(response.statusCode()).isEqualTo(400);
	}

	@Test
	void blocking_inference_unknown_model_returns_503() throws Exception {
		var response = post("/inference", """
				{
				  "modelId": "does-not-exist",
				  "messages": [{"role":"user","content":"Hi"}]
				}
				""");

		assertThat(response.statusCode()).isEqualTo(503);
	}

	// ── POST /v1/inference/stream ─────────────────────────────────────────────

	@Test
	void streaming_inference_returns_200_with_event_stream_content_type() throws Exception {
		var request = HttpRequest.newBuilder().uri(URI.create(BASE + "/inference/stream"))
				.header("Content-Type", "application/json").POST(HttpRequest.BodyPublishers.ofString("""
						{
						  "modelId": "tinyllama",
						  "messages": [{"role":"user","content":"Stream test"}],
						  "sampling": {"maxTokens": 2}
						}
						""")).build();

		var response = http.send(request, HttpResponse.BodyHandlers.ofString());

		assertThat(response.statusCode()).isEqualTo(200);
		assertThat(response.headers().firstValue("Content-Type").orElse("")).contains("text/event-stream");
	}

	@Test
	void streaming_inference_body_contains_sse_data_events() throws Exception {
		var request = HttpRequest.newBuilder().uri(URI.create(BASE + "/inference/stream"))
				.header("Content-Type", "application/json").POST(HttpRequest.BodyPublishers.ofString("""
						{
						  "modelId": "tinyllama",
						  "messages": [{"role":"user","content":"Stream me"}],
						  "sampling": {"maxTokens": 3}
						}
						""")).build();

		var response = http.send(request, HttpResponse.BodyHandlers.ofString());
		String body = response.body();

		// SSE format: each event starts with "data: "
		assertThat(body).contains("data: ");
		// Each data line must be valid JSON with required fields
		String[] lines = body.split("\n");
		for (String line : lines) {
			if (line.startsWith("data: ")) {
				String json = line.substring(6).trim();
				assertThat(json).contains("requestId");
				assertThat(json).contains("token");
				assertThat(json).contains("isComplete");
			}
		}
	}

	@Test
	void streaming_inference_final_event_has_is_complete_true() throws Exception {
		var request = HttpRequest.newBuilder().uri(URI.create(BASE + "/inference/stream"))
				.header("Content-Type", "application/json").POST(HttpRequest.BodyPublishers.ofString("""
						{
						  "modelId": "tinyllama",
						  "messages": [{"role":"user","content":"Final event test"}],
						  "sampling": {"maxTokens": 2}
						}
						""")).build();

		var response = http.send(request, HttpResponse.BodyHandlers.ofString());
		String body = response.body();

		// The final data line must have isComplete:true
		assertThat(body).contains("\"isComplete\":true");
	}

	// ── GET /v1/models ────────────────────────────────────────────────────────

	@Test
	void list_models_returns_200_with_tinyllama() throws Exception {
		var response = get("/models");

		assertThat(response.statusCode()).isEqualTo(200);
		assertThat(response.body()).contains("tinyllama");
		assertThat(response.body()).contains("total");
	}

	// ── GET /v1/models/{modelId} ──────────────────────────────────────────────

	@Test
	void get_model_returns_200_for_loaded_model() throws Exception {
		var response = get("/models/tinyllama");

		assertThat(response.statusCode()).isEqualTo(200);
		assertThat(response.body()).contains("tinyllama");
		assertThat(response.body()).contains("LOADED");
	}

	@Test
	void get_model_returns_404_for_unknown() throws Exception {
		var response = get("/models/ghost");
		assertThat(response.statusCode()).isEqualTo(404);
	}

	// ── GET /v1/cluster/health ────────────────────────────────────────────────

	@Test
	void cluster_health_returns_200() throws Exception {
		var response = get("/cluster/health");

		assertThat(response.statusCode()).isEqualTo(200);
		assertThat(response.body()).contains("status");
		assertThat(response.body()).contains("queueDepth");
	}

	// ── Helpers ───────────────────────────────────────────────────────────────

	private HttpResponse<String> get(String path) throws Exception {
		var request = HttpRequest.newBuilder().uri(URI.create(BASE + path)).GET().build();
		return http.send(request, HttpResponse.BodyHandlers.ofString());
	}

	private HttpResponse<String> post(String path, String body) throws Exception {
		var request = HttpRequest.newBuilder().uri(URI.create(BASE + path)).header("Content-Type", "application/json")
				.POST(HttpRequest.BodyPublishers.ofString(body)).build();
		return http.send(request, HttpResponse.BodyHandlers.ofString());
	}

	private static NodeDescriptor node(String id, long vram) {
		return new NodeDescriptor(id, "localhost", 9092, vram, vram, NodeStatus.READY, 1.0, Instant.now(),
				Instant.now());
	}
}