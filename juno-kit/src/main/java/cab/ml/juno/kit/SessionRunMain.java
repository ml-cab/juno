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
package cab.ml.juno.kit;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import cab.ml.juno.coordinator.GenerationResult;
import cab.ml.juno.coordinator.InferenceRequest;
import cab.ml.juno.coordinator.RequestPriority;
import cab.ml.juno.coordinator.SessionStore;
import cab.ml.juno.player.JunoPlayer;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;

/**
 * One half of a session save and restore for {@link SessionProbe}: {@code SessionRunMain <model> <dir> save|restore
 * <gpu>}, a local server built through {@link JunoPlayer} with {@code <dir>} as its session directory and the KV types
 * this JVM's {@code JUNO_CACHE_TYPE_K/V} properties select. {@code save}: the session's first turn, a save, then its
 * second turn uninterrupted; the first reply is left in {@code <dir>/turn1.txt}. {@code restore}, in a new JVM: a
 * restore, then the same second turn. Greedy. Prints one line after {@link #PREFIX}.
 */
public final class SessionRunMain {

	static final String PREFIX = "JUNO-KIT-SESSION ";
	static final String SESSION = "kit-session";
	private static final SamplingParams PARAMS = SamplingParams.deterministic().withMaxTokens(24);
	private static final List<ChatMessage> TURN1 = List.of(
			ChatMessage.system("You are a helpful assistant. Answer in one short sentence."),
			ChatMessage.user("Name three colors of the rainbow."));

	private SessionRunMain() {
	}

	public static void main(String[] args) throws Exception {
		Path model = Path.of(args[0]);
		Path dir = Path.of(args[1]);
		boolean save = args[2].equals("save");
		boolean gpu = Boolean.parseBoolean(args[3]);
		Map<String, Object> out = new LinkedHashMap<>();
		try (JunoPlayer player = JunoPlayer.builder(model).useGpu(gpu).samplingParams(PARAMS).sessionDir(dir).build()) {
			if (save) {
				GenerationResult first = player.scheduler().submitAndWait(request(TURN1));
				Files.writeString(dir.resolve("turn1.txt"), first.text(), StandardCharsets.UTF_8);
				SessionStore.Outcome saved = player.sessions().save(SESSION);
				GenerationResult second = player.scheduler().submitAndWait(request(turn2(first.text())));
				out.put("tokens", saved.tokens());
				out.put("bytes", saved.bytes());
				out.put("prefixHit", true);
				out.put("secondTurn", second.tokenIds());
				out.put("secondTurnText", second.text());
			} else {
				String firstText = Files.readString(dir.resolve("turn1.txt"), StandardCharsets.UTF_8);
				SessionStore.Outcome restored = player.sessions().restore(SESSION);
				long hits = player.scheduler().prefixHits();
				GenerationResult second = player.scheduler().submitAndWait(request(turn2(firstText)));
				out.put("tokens", restored.tokens());
				out.put("bytes", restored.bytes());
				out.put("prefixHit", player.scheduler().prefixHits() == hits + 1);
				out.put("secondTurn", second.tokenIds());
				out.put("secondTurnText", second.text());
			}
		}
		System.out.println(PREFIX + Json.MAPPER.writer()
				.without(com.fasterxml.jackson.databind.SerializationFeature.INDENT_OUTPUT).writeValueAsString(out));
		System.exit(0);
	}

	private static InferenceRequest request(List<ChatMessage> messages) {
		return InferenceRequest.ofSession(SESSION, "kit", messages, PARAMS, RequestPriority.NORMAL);
	}

	private static List<ChatMessage> turn2(String firstReply) {
		List<ChatMessage> m = new ArrayList<>(TURN1);
		m.add(ChatMessage.assistant(firstReply));
		m.add(ChatMessage.user("Now name one more."));
		return m;
	}
}
