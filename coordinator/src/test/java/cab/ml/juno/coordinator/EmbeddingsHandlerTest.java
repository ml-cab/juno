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

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.node.ForwardPassHandler;
import cab.ml.juno.node.LocalInferencePipeline;
import cab.ml.juno.node.PoolingMode;
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
 * HTTP-level tests for {@code POST /v1/embeddings}, mirroring
 * {@link InferenceApiServerTest}'s harness (real Javalin server, real HTTP
 * calls, a deterministic {@link ForwardPassHandler} test double — no model
 * file, no GPU).
 */
class EmbeddingsHandlerTest {

	private static final int PORT = 28190;
	private static final String BASE = "http://localhost:" + PORT + "/v1";

	private static InferenceApiServer server;
	private static final HttpClient http = HttpClient.newHttpClient();
	private static final ObjectMapper JSON = new ObjectMapper();

	private static final int VOCAB_SIZE = 100;
	private static final int HIDDEN_DIM = 8;
	private static final int NUM_HEADS = 2;
	private static final int TOTAL_LAYERS = 4;

	@BeforeAll
	static void startServer() {
		ShardAssignment assignment = new ShardAssignment("n1", "localhost", 9091, 0, TOTAL_LAYERS, true, true);
		ShardMap shardMap = new ShardMap("model", TOTAL_LAYERS, List.of(assignment), Instant.now());
		var pipeline = LocalInferencePipeline.from(shardMap, new PositionValueForwardPassHandler(), VOCAB_SIZE,
				HIDDEN_DIM, NUM_HEADS);

		var kvCache = new KVCacheManager(new GpuKVCache(64L * 1024 * 1024), new CpuKVCache(64));
		var loop = new GenerationLoop(new SimpleTokenizer(), Sampler.create(), pipeline, kvCache);
		var scheduler = new RequestScheduler(64, loop);

		var registry = new ModelRegistry(cab.ml.juno.registry.ShardPlanner.create());
		ModelDescriptor model = ModelDescriptor.of("tinyllama", "llama", TOTAL_LAYERS, HIDDEN_DIM, VOCAB_SIZE,
				NUM_HEADS, QuantizationType.Q4_K_M, "/models/tinyllama.gguf");
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
	void mean_pooling_averages_every_position() throws Exception {
		// SimpleTokenizer splits on whitespace: "a b c" -> 3 tokens -> positions
		// 0,1,2 -> PositionValueForwardPassHandler fills each with (pos+1): 1,2,3.
		var response = post("""
				{"input": "a b c"}
				""");
		assertThat(response.statusCode()).isEqualTo(200);
		JsonNode body = JSON.readTree(response.body());
		assertThat(body.get("object").asText()).isEqualTo("list");
		JsonNode data = body.get("data");
		assertThat(data).hasSize(1);
		JsonNode embedding = data.get(0).get("embedding");
		assertThat(embedding).hasSize(HIDDEN_DIM);
		for (JsonNode v : embedding)
			assertThat(v.floatValue()).isEqualTo(2.0f); // mean(1,2,3)
		assertThat(body.get("model").asText()).isEqualTo("tinyllama");
		assertThat(body.get("usage").get("prompt_tokens").asLong()).isEqualTo(3);
	}

	@Test
	void cls_pooling_returns_first_position_value() throws Exception {
		var response = post("""
				{"input": "a b c", "x_juno_pooling": "cls"}
				""");
		JsonNode embedding = JSON.readTree(response.body()).get("data").get(0).get("embedding");
		for (JsonNode v : embedding)
			assertThat(v.floatValue()).isEqualTo(1.0f);
	}

	@Test
	void last_pooling_returns_final_position_value() throws Exception {
		var response = post("""
				{"input": "a b c", "x_juno_pooling": "last"}
				""");
		JsonNode embedding = JSON.readTree(response.body()).get("data").get(0).get("embedding");
		for (JsonNode v : embedding)
			assertThat(v.floatValue()).isEqualTo(3.0f);
	}

	@Test
	void batch_input_returns_one_embedding_per_string() throws Exception {
		var response = post("""
				{"input": ["a b", "a b c d"]}
				""");
		assertThat(response.statusCode()).isEqualTo(200);
		JsonNode data = JSON.readTree(response.body()).get("data");
		assertThat(data).hasSize(2);
		assertThat(data.get(0).get("index").asInt()).isEqualTo(0);
		assertThat(data.get(1).get("index").asInt()).isEqualTo(1);
		// "a b" -> positions 0,1 -> mean(1,2) = 1.5; "a b c d" -> mean(1,2,3,4) = 2.5
		assertThat(data.get(0).get("embedding").get(0).floatValue()).isEqualTo(1.5f);
		assertThat(data.get(1).get("embedding").get(0).floatValue()).isEqualTo(2.5f);
	}

	@Test
	void deterministic_for_the_same_input_across_two_calls() throws Exception {
		var r1 = post("""
				{"input": "same prompt twice"}
				""");
		var r2 = post("""
				{"input": "same prompt twice"}
				""");
		assertThat(r1.body()).isEqualTo(r2.body());
	}

	@Test
	void rejects_missing_input() throws Exception {
		var response = post("{}");
		assertThat(response.statusCode()).isEqualTo(400);
	}

	@Test
	void rejects_empty_array_input() throws Exception {
		var response = post("""
				{"input": []}
				""");
		assertThat(response.statusCode()).isEqualTo(400);
	}

	@Test
	void rejects_invalid_pooling_value() throws Exception {
		var response = post("""
				{"input": "a b c", "x_juno_pooling": "softmax"}
				""");
		assertThat(response.statusCode()).isEqualTo(400);
	}

	@Test
	void chat_completions_route_still_works_with_embeddings_enabled() throws Exception {
		var request = HttpRequest.newBuilder().uri(URI.create(BASE + "/chat/completions"))
				.header("Content-Type", "application/json")
				.POST(HttpRequest.BodyPublishers.ofString("""
						{"messages": [{"role":"user","content":"hi"}], "max_tokens": 1}
						""")).build();
		var response = http.send(request, HttpResponse.BodyHandlers.ofString());
		assertThat(response.statusCode()).isEqualTo(200);
	}

	private static HttpResponse<String> post(String body) throws Exception {
		var request = HttpRequest.newBuilder().uri(URI.create(BASE + "/embeddings"))
				.header("Content-Type", "application/json").POST(HttpRequest.BodyPublishers.ofString(body)).build();
		return http.send(request, HttpResponse.BodyHandlers.ofString());
	}
}
