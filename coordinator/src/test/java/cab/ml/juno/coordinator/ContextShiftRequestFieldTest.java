package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.List;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.node.InferencePipeline;
import cab.ml.juno.registry.ModelDescriptor;
import cab.ml.juno.registry.ModelRegistry;
import cab.ml.juno.registry.NodeDescriptor;
import cab.ml.juno.registry.NodeStatus;
import cab.ml.juno.registry.QuantizationType;
import cab.ml.juno.registry.ShardPlanner;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.tokenizer.SimpleTokenizer;

/**
 * The context-shift opt-in has to arrive through both REST surfaces, not only
 * through {@link InferenceRequest}: {@code x_juno_context_shift} on chat
 * completions and {@code contextShift} on the native API. With it a request runs
 * past the context limit; without it the request fails there as before; and on a
 * deployment that cannot shift, asking for it is a 400 naming the field, before
 * any work is queued.
 */
@DisplayName("context shift on the REST surfaces")
class ContextShiftRequestFieldTest {

	private static final int LIMIT = 24;
	private static final int SHIFTING_PORT = 28181;
	private static final int REFUSING_PORT = 28182;
	private static final HttpClient http = HttpClient.newHttpClient();
	private static InferenceApiServer shifting;
	private static InferenceApiServer refusing;

	private static InferenceApiServer server(InferencePipeline pipeline, int port) {
		var kvCache = new KVCacheManager(new GpuKVCache(64L * 1024 * 1024), new CpuKVCache(256));
		var scheduler = new RequestScheduler(8, new GenerationLoop(new SimpleTokenizer(), Sampler.create(), pipeline,
				kvCache));
		var registry = new ModelRegistry(ShardPlanner.create());
		List<NodeDescriptor> nodes = List.of(new NodeDescriptor("n1", "localhost", 9092, 4L << 30, 4L << 30,
				NodeStatus.READY, 1.0, java.time.Instant.now(), java.time.Instant.now()));
		registry.register(ModelDescriptor.of("tiny", "llama", 2, 64, 1000, 2, QuantizationType.Q4_K_M, "/m.gguf"),
				nodes);
		registry.markLoaded("tiny");
		InferenceApiServer s = new InferenceApiServer(scheduler, registry, "BE");
		s.start(port);
		return s;
	}

	@BeforeAll
	static void start() {
		shifting = server(new GenerationLoopContextShiftTest.LimitedPipeline(LIMIT), SHIFTING_PORT);
		refusing = server(new GenerationLoopContextShiftTest.LimitedPipeline(LIMIT) {
			@Override
			public boolean supportsContextShift() {
				return false;
			}
		}, REFUSING_PORT);
	}

	@AfterAll
	static void stop() {
		if (shifting != null)
			shifting.stop();
		if (refusing != null)
			refusing.stop();
	}

	private static HttpResponse<String> post(int port, String path, String json) throws Exception {
		return http.send(HttpRequest.newBuilder(URI.create("http://localhost:" + port + "/v1" + path))
				.header("Content-Type", "application/json").POST(HttpRequest.BodyPublishers.ofString(json)).build(),
				HttpResponse.BodyHandlers.ofString());
	}

	private static String chat(String extra) {
		return """
				{"model":"tiny","messages":[{"role":"system","content":"be brief"},{"role":"user","content":"hi"}],
				 "max_tokens":%d%s}
				""".formatted(3 * LIMIT, extra);
	}

	private static String nativeBody(String extra) {
		return """
				{"modelId":"tiny","messages":[{"role":"user","content":"hi"}],"sampling":{"maxTokens":%d}%s}
				""".formatted(3 * LIMIT, extra);
	}

	@Test
	@DisplayName("chat completions: x_juno_context_shift true runs past the limit; absent, the request fails there")
	void chatCompletionsField() throws Exception {
		HttpResponse<String> on = post(SHIFTING_PORT, "/chat/completions", chat(",\"x_juno_context_shift\":true"));
		assertThat(on.statusCode()).as(on.body()).isEqualTo(200);
		assertThat(on.body()).contains("\"completion_tokens\":" + 3 * LIMIT);

		HttpResponse<String> off = post(SHIFTING_PORT, "/chat/completions", chat(""));
		assertThat(off.statusCode()).isEqualTo(500);
		assertThat(off.body()).contains("context limit");
	}

	@Test
	@DisplayName("native inference: contextShift true runs past the limit; absent, the request fails there")
	void nativeField() throws Exception {
		HttpResponse<String> on = post(SHIFTING_PORT, "/inference", nativeBody(",\"contextShift\":true"));
		assertThat(on.statusCode()).as(on.body()).isEqualTo(200);
		assertThat(on.body()).contains("\"tokenCount\":" + 3 * LIMIT);

		HttpResponse<String> off = post(SHIFTING_PORT, "/inference", nativeBody(""));
		assertThat(off.statusCode()).isGreaterThanOrEqualTo(500);
	}

	@Test
	@DisplayName("a deployment that cannot shift answers 400 naming the field, on both surfaces")
	void refusedWith400() throws Exception {
		HttpResponse<String> chat = post(REFUSING_PORT, "/chat/completions", chat(",\"x_juno_context_shift\":true"));
		assertThat(chat.statusCode()).isEqualTo(400);
		assertThat(chat.body()).contains("x_juno_context_shift").contains("context shift");

		HttpResponse<String> nat = post(REFUSING_PORT, "/inference", nativeBody(",\"contextShift\":true"));
		assertThat(nat.statusCode()).isEqualTo(400);
		assertThat(nat.body()).contains("context shift");
	}
}
