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
 * An embedder reaches context shifting through the facade: a client made with
 * {@link JunoHttpClient#withContextShift} sends the opt-in on both REST shapes;
 * a plain client sends nothing, leaving the server default.
 */
@DisplayName("JunoHttpClient - context shift")
class JunoHttpClientContextShiftTest {

	private HttpServer server;
	private final List<String> bodies = new CopyOnWriteArrayList<>();
	private JunoHttpClient client;

	@BeforeEach
	void startStub() throws IOException {
		server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
		server.createContext("/v1/inference", exchange -> respond(exchange, "{\"text\":\"ok\"}"));
		server.createContext("/v1/chat/completions",
				exchange -> respond(exchange, "{\"choices\":[{\"message\":{\"content\":\"ok\"}}]}"));
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
		assertThat(bodies).isNotEmpty();
		return bodies.get(bodies.size() - 1);
	}

	@Test
	@DisplayName("withContextShift(true) sends contextShift on native and x_juno_context_shift on chat completions")
	void optInIsSent() throws Exception {
		JunoHttpClient shifting = client.withContextShift(true);
		shifting.blockingInference("m", List.of(ChatMessage.user("Ping")), 8);
		assertThat(lastBody()).contains("\"contextShift\":true");
		shifting.blockingOpenAiChat("m", List.of(ChatMessage.user("Ping")), 8, 0.7f);
		assertThat(lastBody()).contains("\"x_juno_context_shift\":true");
	}

	@Test
	@DisplayName("withContextShift(false) sends an explicit false")
	void explicitFalseIsSent() throws Exception {
		client.withContextShift(false).blockingOpenAiChat("m", List.of(ChatMessage.user("Ping")), 8, 0.7f);
		assertThat(lastBody()).contains("\"x_juno_context_shift\":false");
	}

	@Test
	@DisplayName("a plain client sends no context-shift field")
	void plainClientSendsNothing() throws Exception {
		client.blockingInference("m", List.of(ChatMessage.user("Ping")), 8);
		assertThat(lastBody()).doesNotContain("contextShift");
		client.blockingOpenAiChat("m", List.of(ChatMessage.user("Ping")), 8, 0.7f);
		assertThat(lastBody()).doesNotContain("context_shift");
	}

	@Test
	@DisplayName("JunoPlayer: a player built with contextShift(true) sends every request with the opt-in")
	void playerRequestCarriesOptIn() {
		var params = cab.ml.juno.sampler.SamplingParams.defaults();
		assertThat(JunoPlayer.request("m", List.of(ChatMessage.user("hi")), params, true).contextShift()).isTrue();
		assertThat(JunoPlayer.request("m", List.of(ChatMessage.user("hi")), params, null).contextShift()).isNull();
	}
}
