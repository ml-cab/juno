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

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.condition.EnabledIf;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import cab.ml.juno.coordinator.GenerationResult;
import cab.ml.juno.coordinator.InferenceRequest;
import cab.ml.juno.coordinator.RequestPriority;
import cab.ml.juno.coordinator.SessionStore;
import cab.ml.juno.kvcache.CacheTypeOptions;
import cab.ml.juno.kvcache.ServeScheduleOptions;
import cab.ml.juno.node.CudaAvailability;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;

/**
 * A real session saved after its first turn, its server closed, and restored into a new server built from the same
 * model file, as after a restart: its second turn equals, token for token, the second turn of the session that was
 * never interrupted, and is served from the restored KV (a prefix hit, so only the new turn is prefilled). TinyLlama,
 * greedy; CPU and CUDA, {@code f16} and {@code q8_0} host KV; one case saved under the static schedule and restored
 * under the continuous one. Runs with {@code -Djuno.test.sessionLive=true} when the model is on disk; the CUDA cases
 * need a CUDA device.
 */
@DisplayName("Session save and restore across a restart, TinyLlama")
@EnabledIf("enabled")
class SessionSaveRestoreLiveTest {

	private static final String MODEL_FILE = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
	private static final Path MODEL = Path.of(System.getProperty("user.dir")).endsWith("juno-player")
			? Path.of(System.getProperty("user.dir")).getParent().resolve(MODEL_FILE)
			: Path.of(MODEL_FILE);
	private static final String SYSTEM = "You are a helpful assistant. Answer in one short sentence.";
	private static final SamplingParams PARAMS = SamplingParams.deterministic().withMaxTokens(24);

	private final String prevK = System.getProperty(CacheTypeOptions.ENV_K);
	private final String prevV = System.getProperty(CacheTypeOptions.ENV_V);
	private final String prevSchedule = System.getProperty(ServeScheduleOptions.ENV);

	@TempDir
	Path dir;

	static boolean enabled() {
		return Boolean.getBoolean("juno.test.sessionLive") && Files.exists(MODEL);
	}

	@AfterEach
	void restoreProps() {
		restore(CacheTypeOptions.ENV_K, prevK);
		restore(CacheTypeOptions.ENV_V, prevV);
		restore(ServeScheduleOptions.ENV, prevSchedule);
	}

	static Stream<Arguments> cases() {
		return Stream.of(Arguments.of("cpu", "f16", "static", "static"), Arguments.of("cpu", "q8_0", "static", "static"),
				Arguments.of("cpu", "f16", "static", "continuous"), Arguments.of("cuda", "f16", "static", "static"),
				Arguments.of("cuda", "q8_0", "static", "static"));
	}

	@ParameterizedTest(name = "{0}, {1}, saved {2}, restored {3}")
	@MethodSource("cases")
	void restoredSecondTurnEqualsTheUnbrokenOne(String backend, String type, String saveSchedule,
			String restoreSchedule) throws Exception {
		boolean gpu = backend.equals("cuda");
		Assumptions.assumeTrue(!gpu || CudaAvailability.isAvailable(), "no CUDA device");
		System.setProperty(CacheTypeOptions.ENV_K, type);
		System.setProperty(CacheTypeOptions.ENV_V, type);

		List<ChatMessage> turn1 = List.of(ChatMessage.system(SYSTEM), ChatMessage.user("Name three colors of the rainbow."));
		GenerationResult first, unbroken;
		SessionStore.Outcome saved;
		System.setProperty(ServeScheduleOptions.ENV, saveSchedule);
		try (JunoPlayer player = build(gpu)) {
			first = player.scheduler().submitAndWait(session(turn1));
			saved = player.sessions().save("chat-1");
			unbroken = player.scheduler().submitAndWait(session(turn2(turn1, first)));
		}
		assertThat(saved.tokens()).isEqualTo(first.promptTokens());
		assertThat(saved.bytes()).isPositive();
		assertThat(unbroken.generatedTokens()).isPositive();

		System.setProperty(ServeScheduleOptions.ENV, restoreSchedule);
		try (JunoPlayer player = build(gpu)) {
			SessionStore.Outcome restored = player.sessions().restore("chat-1");
			assertThat(restored.tokens()).isEqualTo(saved.tokens());
			assertThat(restored.bytes()).isEqualTo(saved.bytes());
			long hits = player.scheduler().prefixHits();
			GenerationResult resumed = player.scheduler().submitAndWait(session(turn2(turn1, first)));
			assertThat(player.scheduler().prefixHits()).as("the second turn starts from the restored KV")
					.isEqualTo(hits + 1);
			assertThat(resumed.tokenIds()).as("second turn after the restart").isEqualTo(unbroken.tokenIds());
			assertThat(resumed.text()).isEqualTo(unbroken.text());
		}
	}

	private JunoPlayer build(boolean gpu) throws Exception {
		return JunoPlayer.builder(MODEL).useGpu(gpu).samplingParams(PARAMS).sessionDir(dir).build();
	}

	private static InferenceRequest session(List<ChatMessage> messages) {
		return InferenceRequest.ofSession("chat-1", "tinyllama", messages, PARAMS, RequestPriority.NORMAL);
	}

	private static List<ChatMessage> turn2(List<ChatMessage> turn1, GenerationResult first) {
		List<ChatMessage> m = new ArrayList<>(turn1);
		m.add(ChatMessage.assistant(first.text()));
		m.add(ChatMessage.user("Now name one more."));
		return m;
	}

	private static void restore(String key, String prev) {
		if (prev == null)
			System.clearProperty(key);
		else
			System.setProperty(key, prev);
	}
}
