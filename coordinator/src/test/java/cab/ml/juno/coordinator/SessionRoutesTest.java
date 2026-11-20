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

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.kvcache.KvSessionFileException;
import cab.ml.juno.kvcache.KvSessionFileException.Reason;
import cab.ml.juno.kvcache.KvSessionStamp;
import cab.ml.juno.node.InferencePipeline;
import cab.ml.juno.node.SessionKvPart;
import cab.ml.juno.registry.ModelDescriptor;
import cab.ml.juno.registry.ModelRegistry;
import cab.ml.juno.registry.NodeDescriptor;
import cab.ml.juno.registry.NodeStatus;
import cab.ml.juno.registry.QuantizationType;
import cab.ml.juno.registry.ShardPlanner;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.tokenizer.SimpleTokenizer;

/**
 * The session routes: {@code POST /v1/sessions/{id}/save}, {@code POST /v1/sessions/{id}/restore} and
 * {@code DELETE /v1/sessions/{id}}, over a recording pipeline. A saved and deleted session, restored, continues with
 * only its new turn prefilled; every refusal answers the status the API names.
 */
@DisplayName("Session save, restore and delete on the REST surface")
class SessionRoutesTest {

	private static final int PORT_A = 28191, PORT_B = 28192, PORT_C = 28193;
	private static final HttpClient http = HttpClient.newHttpClient();
	private static final ObjectMapper json = new ObjectMapper();

	private static Path dir;
	private static RecordingPipeline pipelineA;
	private static InferenceApiServer a, b, c;

	/** KV as token lists per id; save and restore write and read them; blocks where told to. */
	static final class RecordingPipeline implements InferencePipeline {
		final Map<String, List<Integer>> kv = new ConcurrentHashMap<>();
		final List<String> writes = Collections.synchronizedList(new ArrayList<>());
		final List<String> errors = Collections.synchronizedList(new ArrayList<>());
		final List<String> dropped = Collections.synchronizedList(new ArrayList<>());
		volatile String blockForwardOf, blockSaveOf;
		final CountDownLatch entered = new CountDownLatch(1);
		final CountDownLatch release = new CountDownLatch(1);

		private void write(String id, int pos, int token) {
			List<Integer> row = kv.computeIfAbsent(id, k -> Collections.synchronizedList(new ArrayList<>()));
			if (row.size() < pos)
				errors.add(id + ": write at " + pos + " over " + row.size() + " written positions");
			while (row.size() > pos)
				row.remove(row.size() - 1);
			row.add(token);
		}

		private void maybeBlock(String id, String which) {
			if (id.equals(which)) {
				entered.countDown();
				try {
					release.await(30, TimeUnit.SECONDS);
				} catch (InterruptedException e) {
					Thread.currentThread().interrupt();
				}
			}
		}

		@Override
		public float[] forward(String requestId, int[] tokens, int startPos) {
			writes.add(requestId + "@" + startPos);
			maybeBlock(requestId, blockForwardOf);
			write(requestId, startPos, tokens[tokens.length - 1]);
			float[] logits = new float[vocabSize()];
			logits[50] = 100f;
			return logits;
		}

		@Override
		public void prefillBatch(String requestId, int[] newTokens, int startPosition) {
			writes.add(requestId + "@" + startPosition);
			for (int p = 0; p < newTokens.length; p++)
				write(requestId, startPosition + p, newTokens[p]);
		}

		@Override
		public int vocabSize() {
			return 1000;
		}

		@Override
		public void evict(String requestId) {
			kv.remove(requestId);
		}

		@Override
		public void dropKv(String requestId) {
			dropped.add(requestId);
			kv.remove(requestId);
		}

		@Override
		public List<SessionKvPart> saveKv(String requestId, KvSessionStamp stamp, Path d) throws IOException {
			maybeBlock(requestId, blockSaveOf);
			if (requestId.equals("img"))
				throw new KvSessionFileException(Reason.REFUSED, "session img holds an image");
			List<Integer> row = kv.get(requestId);
			int[] held = row.stream().mapToInt(Integer::intValue).limit(stamp.positions()).toArray();
			if (!Arrays.equals(held, stamp.tokens()))
				errors.add(requestId + ": the stamp's tokens are not the KV's");
			String name = SessionKvPart.fileName(requestId, "layers-0-2");
			Files.writeString(d.resolve(name), stamp.model() + "\n" + Arrays.toString(stamp.tokens()));
			return List.of(new SessionKvPart("layers-0-2", name, Files.size(d.resolve(name))));
		}

		@Override
		public List<SessionKvPart> restoreKv(String requestId, KvSessionStamp stamp, Path d) throws IOException {
			String name = SessionKvPart.fileName(requestId, "layers-0-2");
			Path f = d.resolve(name);
			if (!Files.exists(f))
				throw new KvSessionFileException(Reason.NOT_FOUND, "no saved session file " + name);
			List<Integer> row = Collections.synchronizedList(new ArrayList<>());
			for (int t : stamp.tokens())
				row.add(t);
			kv.put(requestId, row);
			return List.of(new SessionKvPart("layers-0-2", name, Files.size(f)));
		}
	}

	private static InferenceApiServer server(InferencePipeline pipeline, int port, Path sessions, String model) {
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
		if (sessions != null)
			s.setSessionStore(new SessionStore(scheduler, sessions, new SessionStore.Identity(model, "")));
		s.start(port);
		return s;
	}

	@BeforeAll
	static void start() throws IOException {
		dir = Files.createTempDirectory("juno-sessions");
		pipelineA = new RecordingPipeline();
		a = server(pipelineA, PORT_A, dir, "model-a");
		b = server(new RecordingPipeline(), PORT_B, dir, "model-b");
		c = server(new RecordingPipeline(), PORT_C, null, null);
	}

	@AfterAll
	static void stop() {
		for (InferenceApiServer s : new InferenceApiServer[] { a, b, c })
			if (s != null)
				s.stop();
	}

	@Test
	@DisplayName("saved, deleted and restored: 200 with the id, tokens, bytes and time; the next turn prefills only its own tokens")
	void roundTrip() throws Exception {
		HttpResponse<String> turn1 = chat(PORT_A, "s1", "hello there");
		assertThat(turn1.statusCode()).as(turn1.body()).isEqualTo(200);
		String reply = json.readTree(turn1.body()).at("/choices/0/message/content").asText();

		JsonNode saved = ok(post(PORT_A, "/sessions/s1/save"));
		int tokens = saved.get("tokens").asInt();
		assertThat(saved.get("session_id").asText()).isEqualTo("s1");
		assertThat(tokens).isPositive();
		assertThat(saved.get("bytes").asLong()).isPositive();
		assertThat(saved.has("elapsed_ms")).isTrue();
		assertThat(dir.resolve("s1.session")).exists();
		assertThat(dir.resolve("s1.layers-0-2.kv")).exists();

		JsonNode deleted = ok(delete(PORT_A, "/sessions/s1"));
		assertThat(deleted.get("session_id").asText()).isEqualTo("s1");
		assertThat(pipelineA.dropped).contains("s1");
		assertThat(pipelineA.kv).doesNotContainKey("s1");

		JsonNode restored = ok(post(PORT_A, "/sessions/s1/restore"));
		assertThat(restored.get("tokens").asInt()).isEqualTo(tokens);
		assertThat(restored.get("bytes").asLong()).isEqualTo(saved.get("bytes").asLong());

		int before = pipelineA.writes.size();
		HttpResponse<String> turn2 = http.send(HttpRequest.newBuilder(URI.create("http://localhost:" + PORT_A
				+ "/v1/chat/completions")).header("Content-Type", "application/json")
				.POST(HttpRequest.BodyPublishers.ofString(json.writeValueAsString(Map.of("model", "tiny",
						"max_tokens", 2, "x_juno_session_id", "s1", "messages", List.of(
								Map.of("role", "user", "content", "hello there"),
								Map.of("role", "assistant", "content", reply),
								Map.of("role", "user", "content", "and more"))))))
				.build(), HttpResponse.BodyHandlers.ofString());
		assertThat(turn2.statusCode()).as(turn2.body()).isEqualTo(200);
		assertThat(pipelineA.writes.get(before)).as("the first write after the restore").isEqualTo("s1@" + tokens);
		assertThat(pipelineA.errors).isEmpty();
	}

	@Test
	@DisplayName("no session directory: save and restore 400 naming --session-dir; delete still frees memory")
	void noDirectory() throws Exception {
		HttpResponse<String> save = post(PORT_C, "/sessions/x/save");
		assertThat(save.statusCode()).isEqualTo(400);
		assertThat(save.body()).contains("--session-dir");
		assertThat(post(PORT_C, "/sessions/x/restore").statusCode()).isEqualTo(400);
		assertThat(delete(PORT_C, "/sessions/x").statusCode()).isEqualTo(200);
	}

	@Test
	@DisplayName("an id outside letters, digits, '-', '_' and '.', or longer than 128: 400")
	void badId() throws Exception {
		assertThat(post(PORT_A, "/sessions/bad,id/save").statusCode()).isEqualTo(400);
		assertThat(post(PORT_A, "/sessions/" + "x".repeat(129) + "/restore").statusCode()).isEqualTo(400);
		assertThat(delete(PORT_A, "/sessions/bad,id").statusCode()).isEqualTo(400);
	}

	@Test
	@DisplayName("nothing saved under the id: restore 404; a session with no KV: save 404")
	void notFound() throws Exception {
		assertThat(post(PORT_A, "/sessions/nobody/restore").statusCode()).isEqualTo(404);
		assertThat(post(PORT_A, "/sessions/never/save").statusCode()).isEqualTo(404);
	}

	@Test
	@DisplayName("saved by another model: 400 naming the model; a damaged manifest: 400")
	void mismatchAndCorrupt() throws Exception {
		assertThat(chat(PORT_A, "m1", "hi").statusCode()).isEqualTo(200);
		ok(post(PORT_A, "/sessions/m1/save"));
		HttpResponse<String> other = post(PORT_B, "/sessions/m1/restore");
		assertThat(other.statusCode()).isEqualTo(400);
		assertThat(other.body()).contains("model");

		Files.writeString(dir.resolve("junk.session"), "{ not a manifest");
		assertThat(post(PORT_A, "/sessions/junk/restore").statusCode()).isEqualTo(400);
	}

	@Test
	@DisplayName("a session the pipeline refuses to save (one holding an image): 400 with its reason")
	void refused() throws Exception {
		assertThat(chat(PORT_A, "img", "look").statusCode()).isEqualTo(200);
		HttpResponse<String> r = post(PORT_A, "/sessions/img/save");
		assertThat(r.statusCode()).isEqualTo(400);
		assertThat(r.body()).contains("image");
	}

	@Test
	@DisplayName("a request of the session in flight: save 409; a session being saved: its chat request 409")
	void conflicts() throws Exception {
		pipelineA.blockForwardOf = "busy";
		CompletableFuture<HttpResponse<String>> running = chatAsync(PORT_A, "busy", "wait");
		assertThat(pipelineA.entered.await(30, TimeUnit.SECONDS)).isTrue();
		assertThat(post(PORT_A, "/sessions/busy/save").statusCode()).isEqualTo(409);
		assertThat(delete(PORT_A, "/sessions/busy").statusCode()).isEqualTo(409);
		pipelineA.blockForwardOf = null;
		pipelineA.release.countDown();
		assertThat(running.get(30, TimeUnit.SECONDS).statusCode()).isEqualTo(200);
	}

	@Test
	@DisplayName("a chat request for a session held by a save: 409")
	void heldSessionRefusesChat() throws Exception {
		RecordingPipeline p = new RecordingPipeline();
		InferenceApiServer d = server(p, 28194, dir, "model-a");
		try {
			assertThat(chat(28194, "slow", "hi").statusCode()).isEqualTo(200);
			p.blockSaveOf = "slow";
			CompletableFuture<HttpResponse<String>> saving = http.sendAsync(HttpRequest.newBuilder(URI.create(
					"http://localhost:28194/v1/sessions/slow/save")).POST(HttpRequest.BodyPublishers.noBody()).build(),
					HttpResponse.BodyHandlers.ofString());
			assertThat(p.entered.await(30, TimeUnit.SECONDS)).isTrue();
			HttpResponse<String> held = chat(28194, "slow", "again");
			assertThat(held.statusCode()).as(held.body()).isEqualTo(409);
			p.release.countDown();
			assertThat(saving.get(30, TimeUnit.SECONDS).statusCode()).isEqualTo(200);
			assertThat(chat(28194, "slow", "again").statusCode()).isEqualTo(200);
		} finally {
			d.stop();
		}
	}

	// ── helpers ──────────────────────────────────────────────────────────────

	private static JsonNode ok(HttpResponse<String> r) throws IOException {
		assertThat(r.statusCode()).as(r.body()).isEqualTo(200);
		return json.readTree(r.body());
	}

	private static HttpResponse<String> chat(int port, String session, String text) throws Exception {
		return chatAsync(port, session, text).get(60, TimeUnit.SECONDS);
	}

	private static CompletableFuture<HttpResponse<String>> chatAsync(int port, String session, String text)
			throws Exception {
		String body = json.writeValueAsString(Map.of("model", "tiny", "max_tokens", 2, "x_juno_session_id", session,
				"messages", List.of(Map.of("role", "user", "content", text))));
		return http.sendAsync(HttpRequest.newBuilder(URI.create("http://localhost:" + port + "/v1/chat/completions"))
				.header("Content-Type", "application/json").POST(HttpRequest.BodyPublishers.ofString(body)).build(),
				HttpResponse.BodyHandlers.ofString());
	}

	private static HttpResponse<String> post(int port, String path) throws Exception {
		return http.send(HttpRequest.newBuilder(URI.create("http://localhost:" + port + "/v1" + path))
				.POST(HttpRequest.BodyPublishers.noBody()).build(), HttpResponse.BodyHandlers.ofString());
	}

	private static HttpResponse<String> delete(int port, String path) throws Exception {
		return http.send(HttpRequest.newBuilder(URI.create("http://localhost:" + port + "/v1" + path)).DELETE().build(),
				HttpResponse.BodyHandlers.ofString());
	}
}
