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
package cab.ml.juno.master;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Stream;

import cab.ml.juno.coordinator.GenerationLoop;
import cab.ml.juno.coordinator.GenerationResult;
import cab.ml.juno.coordinator.InferenceRequest;
import cab.ml.juno.coordinator.RequestPriority;
import cab.ml.juno.coordinator.RequestScheduler;
import cab.ml.juno.coordinator.SessionStore;
import cab.ml.juno.kvcache.SessionDirOptions;
import cab.ml.juno.node.LlamaConfig;
import cab.ml.juno.player.ClusterHarness;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.GgufTokenizer;

/**
 * Check 11: a session saved on a cluster continues after the cluster restarts. On a pipeline-parallel and then a
 * tensor-parallel cluster: a session's first turn, a save (each node writes its own layers under the session directory,
 * the coordinator the manifest), its second turn uninterrupted; the cluster stops; a new cluster (new node processes, the
 * same directory) restores the session and runs the same second turn, which must equal the uninterrupted one token for
 * token and start from the restored KV (a prefix hit). Greedy.
 */
final class SessionSaveRestoreCheck {

	private static final SamplingParams GREEDY = SamplingParams.deterministic().withMaxTokens(16);
	private static final String SESSION = "live-session";
	private static final List<ChatMessage> TURN1 = List.of(
			ChatMessage.system("You are a helpful assistant. Answer in one short sentence."),
			ChatMessage.user("Name three colors of the rainbow."));

	private SessionSaveRestoreCheck() {
	}

	static String run(String modelPath, LlamaConfig cfg, GgufTokenizer tokenizer) throws IOException {
		Path dir = Files.createTempDirectory("juno-live-sessions");
		String saved = System.getProperty(SessionDirOptions.ENV);
		System.setProperty(SessionDirOptions.ENV, dir.toString()); // ClusterHarness forwards it to every node
		try {
			String pipeline = mode(modelPath, tokenizer, dir.resolve("pipeline"),
					d -> ClusterHarness.threeNodes(modelPath, cfg.numLayers()), "pipeline");
			String tensor = mode(modelPath, tokenizer, dir.resolve("tensor"),
					d -> ClusterHarness.tensorNodes(modelPath, cfg.numLayers(), cfg.numHeads()), "tensor");
			return pipeline + "; " + tensor;
		} finally {
			if (saved == null)
				System.clearProperty(SessionDirOptions.ENV);
			else
				System.setProperty(SessionDirOptions.ENV, saved);
			deleteTree(dir);
		}
	}

	private static String mode(String modelPath, GgufTokenizer tokenizer, Path dir,
			Function<Path, ClusterHarness> cluster, String label) throws IOException {
		System.setProperty(SessionDirOptions.ENV, dir.toString());
		Files.createDirectories(dir);
		SessionStore.Identity identity = SessionStore.Identity.of(Path.of(modelPath), null);
		GenerationResult first, unbroken;
		SessionStore.Outcome save;
		ClusterHarness a = cluster.apply(dir);
		try {
			a.start();
			RequestScheduler scheduler = scheduler(a, tokenizer);
			SessionStore store = new SessionStore(scheduler, dir, identity);
			first = scheduler.submitAndWait(request(TURN1));
			save = store.save(SESSION);
			unbroken = scheduler.submitAndWait(request(turn2(first.text())));
			scheduler.shutdown();
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
			throw new IOException(e);
		} finally {
			stop(a);
		}
		ClusterHarness b = cluster.apply(dir);
		try {
			b.start();
			RequestScheduler scheduler = scheduler(b, tokenizer);
			SessionStore store = new SessionStore(scheduler, dir, identity);
			SessionStore.Outcome restore = store.restore(SESSION);
			ModelLiveChecks.check(restore.tokens() == save.tokens() && restore.bytes() == save.bytes(),
					label + ": restored " + restore.tokens() + " tokens / " + restore.bytes() + " bytes, saved "
							+ save.tokens() + " / " + save.bytes());
			long hits = scheduler.prefixHits();
			GenerationResult resumed = scheduler.submitAndWait(request(turn2(first.text())));
			ModelLiveChecks.check(scheduler.prefixHits() == hits + 1,
					label + ": the second turn after the restart did not start from the restored KV");
			ModelLiveChecks.check(resumed.tokenIds().equals(unbroken.tokenIds()),
					label + ": second turn after the restart \"" + resumed.text() + "\", uninterrupted \""
							+ unbroken.text() + "\"");
			scheduler.shutdown();
			return label + ": " + save.tokens() + " tokens, " + save.bytes() + " bytes, second turns equal";
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
			throw new IOException(e);
		} finally {
			stop(b);
		}
	}

	private static RequestScheduler scheduler(ClusterHarness h, GgufTokenizer tokenizer) {
		GenerationLoop loop = new GenerationLoop(tokenizer, Sampler.create(), h.pipeline(),
				ModelLiveChecks.newKvCache(4096));
		return new RequestScheduler(4, loop);
	}

	private static InferenceRequest request(List<ChatMessage> messages) {
		return InferenceRequest.ofSession(SESSION, "model", messages, GREEDY, RequestPriority.NORMAL);
	}

	private static List<ChatMessage> turn2(String firstReply) {
		List<ChatMessage> m = new ArrayList<>(TURN1);
		m.add(ChatMessage.assistant(firstReply));
		m.add(ChatMessage.user("Now name one more."));
		return m;
	}

	private static void stop(ClusterHarness h) {
		try {
			h.stop();
		} catch (Exception ignored) {
			// the next cluster starts on its own ports
		}
	}

	private static void deleteTree(Path dir) throws IOException {
		try (Stream<Path> files = Files.walk(dir)) {
			for (Path f : files.sorted(Comparator.reverseOrder()).toList())
				Files.deleteIfExists(f);
		}
	}
}
