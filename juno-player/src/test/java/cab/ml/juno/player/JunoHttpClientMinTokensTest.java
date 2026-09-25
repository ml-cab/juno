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
package cab.ml.juno.player;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import com.sun.net.httpserver.HttpServer;

import cab.ml.juno.tokenizer.ChatMessage;

/**
 * The embedding facade has to be able to reach a minimum token count, or an
 * embedder is left hand-building JSON for a capability the REST surfaces already
 * expose.
 *
 * <p>These assert on the request body the facade puts on the wire, against a stub
 * server, because that body is the contract between the facade and the engine.
 * The existing single-argument methods now delegate to the new ones, so the same
 * cases also cover that the delegation did not change what they send.
 */
@DisplayName("JunoHttpClient — minimum token count")
class JunoHttpClientMinTokensTest {

	private HttpServer server;
	private final List<String> bodies = new CopyOnWriteArrayList<>();
	private JunoHttpClient client;

	@BeforeEach
	void startStub() throws IOException {
		server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
		server.createContext("/v1/inference", exchange -> respond(exchange, "{\"text\":\"ok\"}"));
		server.createContext("/v1/chat/completions",
				exchange -> respond(exchange, "{\"choices\":[{\"message\":{\"content\":\"ok\"}}]}"));
		server.createContext("/v1/bad", exchange -> {
			exchange.sendResponseHeaders(400, -1);
			exchange.close();
		});
		server.start();
		client = new JunoHttpClient(URI.create("http://127.0.0.1:" + server.getAddress().getPort()));
	}

	@AfterEach
	void stopStub() {
		if (server != null)
			server.stop(0);
	}

	private void respond(com.sun.net.httpserver.HttpExchange exchange, String body) throws IOException {
		bodies.add(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
		byte[] out = body.getBytes(StandardCharsets.UTF_8);
		exchange.getResponseHeaders().add("Content-Type", "application/json");
		exchange.sendResponseHeaders(200, out.length);
		try (OutputStream os = exchange.getResponseBody()) {
			os.write(out);
		}
	}

	private String lastBody() {
		assertThat(bodies).as("the stub received a request").isNotEmpty();
		return bodies.get(bodies.size() - 1);
	}

	@Test
	@DisplayName("native inference sends minTokens inside sampling")
	void nativeInferenceSendsMinTokens() throws Exception {
		String text = client.blockingInference("tinyllama", List.of(ChatMessage.user("Ping")), 64, 64);

		assertThat(text).isEqualTo("ok");
		assertThat(lastBody()).contains("\"minTokens\":64").contains("\"maxTokens\":64");
	}

	@Test
	@DisplayName("chat completions sends min_tokens")
	void chatCompletionsSendsMinTokens() throws Exception {
		String text = client.blockingOpenAiChat("tinyllama", List.of(ChatMessage.user("Ping")), 64, 0.7f, 64);

		assertThat(text).isEqualTo("ok");
		assertThat(lastBody()).contains("\"min_tokens\":64").contains("\"max_completion_tokens\":64");
	}

	@Test
	@DisplayName("the existing native method still sends no minimum")
	void nativeInferenceWithoutAMinimumIsUnchanged() throws Exception {
		String text = client.blockingInference("tinyllama", List.of(ChatMessage.user("Ping")), 64);

		assertThat(text).isEqualTo("ok");
		assertThat(lastBody()).contains("\"maxTokens\":64").doesNotContain("minTokens");
	}

	@Test
	@DisplayName("the existing chat method still sends no minimum")
	void chatCompletionsWithoutAMinimumIsUnchanged() throws Exception {
		String text = client.blockingOpenAiChat("tinyllama", List.of(ChatMessage.user("Ping")), 64, 0.7f);

		assertThat(text).isEqualTo("ok");
		assertThat(lastBody()).contains("\"max_completion_tokens\":64").doesNotContain("min_tokens");
	}

	@Test
	@DisplayName("a null minimum is omitted rather than sent as zero")
	void aNullMinimumIsOmitted() throws Exception {
		client.blockingInference("tinyllama", List.of(ChatMessage.user("Ping")), 64, null);

		assertThat(lastBody()).doesNotContain("minTokens");
	}

	@Test
	@DisplayName("a minimum with no maximum still reaches the sampling block")
	void aMinimumWithoutAMaximumIsStillSent() throws Exception {
		client.blockingInference("tinyllama", List.of(ChatMessage.user("Ping")), null, 8);

		assertThat(lastBody()).contains("\"minTokens\":8");
	}

	@Test
	@DisplayName("a rejected request surfaces its status rather than an empty answer")
	void aRejectedRequestThrows() {
		JunoHttpClient badClient = new JunoHttpClient(
				URI.create("http://127.0.0.1:" + server.getAddress().getPort() + "/v1/bad"));

		assertThatThrownBy(() -> badClient.blockingInference("tinyllama", List.of(ChatMessage.user("Ping")), 8, 8))
				.isInstanceOf(IllegalStateException.class).hasMessageContaining("400");
	}
}
