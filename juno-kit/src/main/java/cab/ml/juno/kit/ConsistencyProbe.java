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


import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

import com.fasterxml.jackson.databind.node.ObjectNode;

import cab.ml.juno.player.JunoHttpClient;
import cab.ml.juno.tokenizer.ChatMessage;

/**
 * The consistency smoke's checks, through the launcher in child JVMs:
 * <ul>
 * <li>local mode and both cluster types refuse each architecture Juno has no verified handler for, naming it,
 * and a refused cluster start leaves no node JVM behind (on header-only files, so nothing is downloaded);</li>
 * <li>local mode loads a supported model and exits cleanly;</li>
 * <li>static batching ({@code --parallel 2}) never resumes from KV it did not write: with prompts repeated
 * across and within batches, every OpenAI and native answer equals that prompt's single-request answer, a
 * session request after the batched traffic does too, and stateless traffic gets no prefix-cache hit;</li>
 * <li>native streaming ({@code /v1/inference/stream}) joins to the native blocking answer;</li>
 * <li>on the continuous schedule, a session request repeating a stateless one prefills in full and equals it.</li>
 * </ul>
 * With {@code gpu=true} only the static batching checks run, on the GPU.
 */
final class ConsistencyProbe implements KitProbe {

	static final List<String> UNSUPPORTED = List.of("qwen35", "gemma4", "mistral3", "minimax-m2");
	static final String SYSTEM = "You are a helpful assistant. Answer in one short sentence.";
	static final List<String> PROMPTS = List.of("What is the capital of France?", "Name one primary color.",
			"How many legs does a spider have?");
	private static final int MAX_TOKENS = 8;
	private static final Duration START = Duration.ofMinutes(10);

	@Override
	public ProbeReport run(String id, Map<String, String> args) throws Exception {
		Path model = Path.of(args.get("model"));
		boolean gpu = Boolean.parseBoolean(args.getOrDefault("gpu", "false"));
		String heap = args.getOrDefault("serverHeap", "4g");
		Path work = Path.of(args.getOrDefault("logDir", ".")).resolve(id);
		List<ProbeReport.Check> checks = new ArrayList<>();
		if (!gpu) {
			String archs = args.getOrDefault("architectures", String.join(",", UNSUPPORTED));
			for (String arch : archs.split(","))
				refusals(work.resolve("refuse-" + arch), arch.trim(), heap, checks);
			List<String> good = JunoProcess.localArgs(model, MAX_TOKENS);
			good.add("--cpu");
			JunoProcess.Result r = JunoProcess.runOnce(work.resolve("known-good"), heap, good, "hello\nexit\n", START);
			boolean clean = r.exit() == 0 && !r.output().contains("Unsupported model architecture")
					&& !r.output().contains("Exception in thread");
			checks.add(new ProbeReport.Check("local mode loads a supported model and exits cleanly", clean,
					"exit " + r.exit() + (clean ? "" : "; see " + id + "/known-good/" + JunoProcess.LOG)));
		}
		staticBatching(work, model, heap, gpu ? "gpu" : "cpu", checks);
		if (!gpu)
			continuousSession(work, model, heap, checks);
		Map<String, Object> facts = new LinkedHashMap<>();
		facts.put("gpu", gpu);
		return new ProbeReport(id, ProbeKind.CHECK, checks.stream().allMatch(ProbeReport.Check::passed), checks,
				List.of(), facts);
	}

	private static void refusals(Path dir, String arch, String heap, List<ProbeReport.Check> checks)
			throws Exception {
		Path file = HeaderOnlyGguf.write(dir.resolve(arch + ".gguf"), arch);
		String named = "Unsupported model architecture '" + arch + "'";
		List<String> local = new ArrayList<>(List.of("--model-path", file.toString(), "--cpu", "--nodes", "1",
				"--max-tokens", "8", "--temperature", "0", "--local"));
		JunoProcess.Result r = JunoProcess.runOnce(dir.resolve("local"), heap, local, "hello\nexit\n", START);
		checks.add(new ProbeReport.Check("local mode refuses architecture " + arch + " and names it",
				r.exit() != 0 && r.output().contains(named), "exit " + r.exit()
						+ (r.output().contains(named) ? "; names it" : "; the output does not contain: " + named)));
		for (String type : List.of("pipeline", "tensor")) {
			Instant t0 = Instant.now();
			List<String> cluster = List.of("--model-path", file.toString(), "--pType", type, "--cpu", "--max-tokens",
					"8", "--temperature", "0");
			r = JunoProcess.runOnce(dir.resolve(type), heap, cluster, "hello\nexit\n", START);
			Thread.sleep(2000);
			List<ProcessHandle> left = JunoProcess.nodesStartedSince(t0);
			left.forEach(ProcessHandle::destroyForcibly);
			boolean ok = r.exit() != 0 && r.output().contains(named) && left.isEmpty();
			checks.add(new ProbeReport.Check(type + " cluster refuses architecture " + arch
					+ ", names it, leaves no node JVM", ok, "exit " + r.exit()
							+ (r.output().contains(named) ? "; names it" : "; the output does not contain: " + named)
							+ "; " + left.size() + " node JVMs left"));
		}
	}

	private static void staticBatching(Path work, Path model, String heap, String mode, List<ProbeReport.Check> checks)
			throws Exception {
		String label = "static batching, " + mode + ": ";
		List<String> args = JunoProcess.localArgs(model, MAX_TOKENS);
		args.addAll(List.of("--parallel", "2", "--batch-window-ms", "300", "--" + mode));
		try (JunoProcess s = JunoProcess.serve(work.resolve("static-batch-" + mode), heap, args, START)) {
			JunoHttpClient client = new JunoHttpClient(s.base());
			String modelId = s.modelId();
			List<String> baseline = new ArrayList<>();
			for (String p : PROMPTS)
				baseline.add(openAi(client, modelId, p));
			List<int[]> rounds = List.of(new int[] { 0, 1, 2, 0 }, new int[] { 0, 0, 1, 2 });
			for (int k = 0; k < rounds.size(); k++) {
				int[] order = rounds.get(k);
				int round = k + 1;
				compare(label + "OpenAI round " + round, order, baseline, i -> openAi(client, modelId, PROMPTS.get(i)),
						checks);
				compare(label + "native round " + round, order, baseline, i -> nativeText(s, modelId, PROMPTS.get(i)),
						checks);
			}
			String session = session(s, PROMPTS.get(0), "kit-" + mode + "-session");
			checks.add(new ProbeReport.Check(
					label + "a session request after batched stateless traffic equals single-request output",
					session.equals(baseline.get(0)), "single '" + baseline.get(0) + "', session '" + session + "'"));
			var health = s.get("/v1/cluster/health").json();
			long hits = health.path("prefixHits").asLong(-1);
			checks.add(new ProbeReport.Check(label + "no prefix-cache hit for stateless traffic", hits == 0,
					hits + " hits, " + health.path("prefixLookups").asLong(-1) + " lookups"));
			String blocking = nativeText(s, modelId, PROMPTS.get(1));
			String streamed = nativeStream(s, modelId, PROMPTS.get(1));
			checks.add(new ProbeReport.Check(label + "native streaming joins to the native blocking answer",
					streamed.equals(blocking) && !blocking.startsWith("ERROR"),
					"blocking '" + blocking + "', streamed '" + streamed + "'"));
		}
	}

	private static void continuousSession(Path work, Path model, String heap, List<ProbeReport.Check> checks)
			throws Exception {
		List<String> args = JunoProcess.localArgs(model, MAX_TOKENS);
		args.addAll(List.of("--parallel", "4", "--batch-window-ms", "300", "--cpu", "--schedule", "continuous"));
		try (JunoProcess s = JunoProcess.serve(work.resolve("continuous"), heap, args, START)) {
			String base = openAi(new JunoHttpClient(s.base()), s.modelId(), PROMPTS.get(0));
			String session = session(s, PROMPTS.get(0), "kit-continuous-session");
			checks.add(new ProbeReport.Check("continuous: a session request after the same stateless request equals it",
					base.equals(session) && !base.startsWith("ERROR"),
					"stateless '" + base + "', session '" + session + "'"));
		}
	}

	interface Ask {
		String ask(int prompt) throws Exception;
	}

	/** Sends the prompts of {@code order} at once and compares each answer with its single-request one. */
	private static void compare(String label, int[] order, List<String> baseline, Ask ask,
			List<ProbeReport.Check> checks) {
		List<CompletableFuture<String>> answers = new ArrayList<>();
		for (int i : order)
			answers.add(CompletableFuture.supplyAsync(() -> {
				try {
					return ask.ask(i);
				} catch (Exception e) {
					return "ERROR: " + e.getMessage();
				}
			}, Thread::startVirtualThread));
		for (int k = 0; k < order.length; k++) {
			String got = answers.get(k).join();
			String want = baseline.get(order[k]);
			checks.add(new ProbeReport.Check(label + ", prompt " + order[k] + " equals its single-request output",
					got.equals(want), got.equals(want) ? "'" + got + "'" : "single '" + want + "', batched '" + got + "'"));
		}
	}

	private static List<ChatMessage> messages(String prompt) {
		return List.of(ChatMessage.system(SYSTEM), ChatMessage.user(prompt));
	}

	private static String openAi(JunoHttpClient c, String modelId, String prompt) {
		try {
			return c.blockingOpenAiChat(modelId, messages(prompt), MAX_TOKENS, 0f);
		} catch (Exception e) {
			return "ERROR: " + e.getMessage();
		}
	}

	/**
	 * A native request as the smoke sends it, greedy by its own sampling block: the facade's native call
	 * sends no temperature, so the server's default sampling would apply.
	 */
	private static String nativeText(JunoProcess s, String modelId, String prompt) throws Exception {
		JunoProcess.Answer a = s.post("/v1/inference", nativeBody(modelId, prompt).toString());
		return a.status() == 200 ? a.json().path("text").asText("") : "ERROR: HTTP " + a.status() + ": " + a.body();
	}

	/** {@code POST /v1/inference/stream}: its events' tokens, joined, up to the completing event. */
	private static String nativeStream(JunoProcess s, String modelId, String prompt) throws Exception {
		JunoProcess.Answer a = s.post("/v1/inference/stream", nativeBody(modelId, prompt).toString());
		if (a.status() != 200)
			return "ERROR: HTTP " + a.status() + ": " + a.body();
		StringBuilder text = new StringBuilder();
		for (String line : a.body().split("\n")) {
			if (!line.startsWith("data: "))
				continue;
			var event = Json.MAPPER.readTree(line.substring("data: ".length()));
			if (event.path("isComplete").asBoolean(false))
				break;
			text.append(event.path("token").asText(""));
		}
		return text.toString();
	}

	private static ObjectNode nativeBody(String modelId, String prompt) {
		ObjectNode body = Json.MAPPER.createObjectNode().put("modelId", modelId);
		var msgs = body.putArray("messages");
		msgs.addObject().put("role", "system").put("content", SYSTEM);
		msgs.addObject().put("role", "user").put("content", prompt);
		body.putObject("sampling").put("temperature", 0).put("maxTokens", MAX_TOKENS);
		return body;
	}

	/** An OpenAI request carrying {@code x_juno_session_id}, which the facade does not send. */
	private static String session(JunoProcess s, String prompt, String sessionId) throws Exception {
		ObjectNode body = Json.MAPPER.createObjectNode();
		var msgs = body.putArray("messages");
		msgs.addObject().put("role", "system").put("content", SYSTEM);
		msgs.addObject().put("role", "user").put("content", prompt);
		body.put("temperature", 0).put("max_tokens", MAX_TOKENS).put("x_juno_session_id", sessionId);
		JunoProcess.Answer a = s.post("/v1/chat/completions", body.toString());
		return a.status() == 200 ? a.json().path("choices").path(0).path("message").path("content").asText("")
				: "ERROR: HTTP " + a.status() + ": " + a.body();
	}
}
