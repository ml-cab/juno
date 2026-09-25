/*
 * Copyright 2026 Dmytro Soloviov (soulaway)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.registry.ModelDescriptor;
import cab.ml.juno.registry.ModelRegistry;
import cab.ml.juno.registry.NodeDescriptor;
import cab.ml.juno.registry.NodeStatus;
import cab.ml.juno.registry.QuantizationType;
import cab.ml.juno.registry.ShardPlanner;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.tokenizer.SimpleTokenizer;

/**
 * The native surface answers 400 for a sampling value it cannot honour.
 *
 * <p>The sampling parameters validate their own ranges and throw. Nothing on this
 * surface used to catch that, so a plainly bad request — a minimum above the
 * maximum, a temperature out of range — came back as a server error, which tells
 * a caller that Juno broke rather than that the request was wrong. The chat
 * surface already mapped these to 400; this one did not.
 */
@DisplayName("Native inference surface — invalid sampling fails closed")
class NativeSamplingRejectionTest {

	private static final int PORT = 28084;
	private static final String BASE = "http://localhost:" + PORT + "/v1";
	private static final long GB = 1024L * 1024 * 1024;

	private static InferenceApiServer server;
	private static final HttpClient http = HttpClient.newHttpClient();

	@BeforeAll
	static void startServer() {
		List<NodeDescriptor> nodes = List.of(new NodeDescriptor("n1", "localhost", 9092, 4 * GB, 4 * GB,
				NodeStatus.READY, 1.0, Instant.now(), Instant.now()));
		var kvCache = new KVCacheManager(new GpuKVCache(64L * 1024 * 1024), new CpuKVCache(256));
		var loop = new GenerationLoop(new SimpleTokenizer(), Sampler.create(), new StubInferencePipeline(), kvCache);
		var scheduler = new RequestScheduler(16, loop);

		var registry = new ModelRegistry(ShardPlanner.create());
		ModelDescriptor tinyllama = ModelDescriptor.of("tinyllama", "llama", 22, 2048, 32_000, 32,
				QuantizationType.Q4_K_M, "/models/tinyllama.gguf");
		registry.register(tinyllama, nodes);
		registry.markLoaded("tinyllama");

		server = new InferenceApiServer(scheduler, registry, "BE");
		server.start(PORT);
	}

	@AfterAll
	static void stopServer() {
		if (server != null)
			server.stop();
	}

	@Test
	@DisplayName("a minimum above the maximum is a bad request, not a server error")
	void minimumAboveMaximumIsRejected() throws Exception {
		var response = post("/inference", """
				{
				  "modelId": "tinyllama",
				  "messages": [{"role":"user","content":"Hello"}],
				  "sampling": {"maxTokens": 8, "minTokens": 64}
				}
				""");

		assertThat(response.statusCode()).isEqualTo(400);
		assertThat(response.body()).contains("minTokens");
	}

	@Test
	@DisplayName("the same holds on the streaming route")
	void minimumAboveMaximumIsRejectedWhenStreaming() throws Exception {
		var response = post("/inference/stream", """
				{
				  "modelId": "tinyllama",
				  "messages": [{"role":"user","content":"Hello"}],
				  "sampling": {"maxTokens": 8, "minTokens": 64}
				}
				""");

		assertThat(response.statusCode()).isEqualTo(400);
	}

	@Test
	@DisplayName("an out-of-range temperature is rejected the same way, not only the new field")
	void outOfRangeTemperatureIsRejected() throws Exception {
		var response = post("/inference", """
				{
				  "modelId": "tinyllama",
				  "messages": [{"role":"user","content":"Hello"}],
				  "sampling": {"maxTokens": 8, "temperature": 9.0}
				}
				""");

		assertThat(response.statusCode()).isEqualTo(400);
		assertThat(response.body()).contains("temperature");
	}

	@Test
	@DisplayName("a valid minimum is accepted")
	void aValidMinimumIsAccepted() throws Exception {
		var response = post("/inference", """
				{
				  "modelId": "tinyllama",
				  "messages": [{"role":"user","content":"Hello"}],
				  "sampling": {"maxTokens": 4, "minTokens": 2}
				}
				""");

		assertThat(response.statusCode()).isEqualTo(200);
	}

	private HttpResponse<String> post(String path, String body) throws Exception {
		var request = HttpRequest.newBuilder().uri(URI.create(BASE + path)).header("Content-Type", "application/json")
				.POST(HttpRequest.BodyPublishers.ofString(body)).build();
		return http.send(request, HttpResponse.BodyHandlers.ofString());
	}
}
