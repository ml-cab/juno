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
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

/**
 * The context-policy smoke's checks over OpenAI REST, through the launcher in a child JVM. For each schedule
 * a server starts with context shifting off and runs, in this order (a failed request keeps its KV until the
 * server stops, so the failing run goes last):
 * <ol>
 * <li>a conversation grown turn by turn (canned assistant replies, so every run sends the same messages) far
 * past the context limit, with {@code x_juno_context_shift} and a session id, answers every turn;</li>
 * <li>a prompt just under the limit, held open by {@code min_tokens}, shifts during decode and returns every
 * requested token;</li>
 * <li>the same conversation without the opt-in fails once a turn reaches the limit, with the error naming the
 * model's original context length, after answering the earlier turns.</li>
 * </ol>
 * Then, on the static schedule, a server started with shifting on answers the crossing turn with no field and
 * fails it when the request says {@code false}. {@code limit} is the request's limit in positions: 4,096 on
 * Phi-3.5-mini, its original training context, which is why the smoke runs on it; every other family's limit
 * is the KV cache's 32,768 positions, too far for a conversation a probe can afford to grow.
 */
final class ContextPolicyProbe implements KitProbe {

	static final String LEDGER = "You keep a ledger of notes. Answer briefly.";
	private static final Duration START = Duration.ofMinutes(10);

	@Override
	public ProbeReport run(String id, Map<String, String> args) throws Exception {
		Path model = Path.of(args.get("model"));
		boolean gpu = Boolean.parseBoolean(args.getOrDefault("gpu", "false"));
		int limit = KitProbe.intArg(args, "limit", 4096);
		int turns = KitProbe.intArg(args, "turns", 10);
		int notesPerTurn = KitProbe.intArg(args, "notesPerTurn", 25);
		int replyTokens = KitProbe.intArg(args, "replyTokens", 16);
		int decodeTokens = KitProbe.intArg(args, "decodeTokens", 200);
		int fitNotes = KitProbe.intArg(args, "fitNotes", 101);
		String heap = args.getOrDefault("serverHeap", "16g");
		Path work = Path.of(args.getOrDefault("logDir", ".")).resolve(id);
		Conversation conv = new Conversation(turns, notesPerTurn, replyTokens);
		List<ProbeReport.Check> checks = new ArrayList<>();
		Map<String, Object> facts = new LinkedHashMap<>();
		int crossingStatic = 0;
		for (String schedule : args.getOrDefault("schedules", "static,continuous").split(",")) {
			try (JunoProcess s = JunoProcess.serve(work.resolve("off-" + schedule), heap,
					serverArgs(model, gpu, schedule, "off", replyTokens), START)) {
				ObjectNode optIn = Json.MAPPER.createObjectNode().put("x_juno_context_shift", true)
						.put("x_juno_session_id", "kit-" + schedule);
				int shiftedFailed = conv.firstFailingTurn(s, optIn);
				checks.add(new ProbeReport.Check(schedule + ": with x_juno_context_shift every one of " + turns
						+ " turns answers", shiftedFailed == 0,
						shiftedFailed == 0 ? turns + " turns" : "turn " + shiftedFailed + " failed: " + conv.lastError));

				// The note count for a prompt 100 tokens under the limit, from the engine's counts for 1 and k notes.
				JsonNode one = send(s, single(LedgerNotes.notes(1, 1)), Json.MAPPER.createObjectNode(), 1, 0).json;
				JsonNode many = send(s, single(LedgerNotes.notes(1, fitNotes)), Json.MAPPER.createObjectNode(), 1, 0).json;
				int p1 = one.path("usage").path("prompt_tokens").asInt(0);
				int pk = many.path("usage").path("prompt_tokens").asInt(0);
				int near = pk > p1 ? 1 + (limit - 100 - p1) * (fitNotes - 1) / (pk - p1) : 0;
				Sent decode = send(s, single(LedgerNotes.notes(1, Math.max(1, near))),
						Json.MAPPER.createObjectNode().put("x_juno_context_shift", true), decodeTokens, decodeTokens);
				int done = decode.json.path("usage").path("completion_tokens").asInt(0);
				int positions = decode.json.path("usage").path("prompt_tokens").asInt(0) + done;
				checks.add(new ProbeReport.Check(schedule + ": a request held open past the limit shifts during decode",
						decode.status == 200 && done == decodeTokens && positions > limit,
						"HTTP " + decode.status + ", " + done + " of " + decodeTokens + " tokens, " + positions
								+ " positions against a limit of " + limit));

				int failedAt = conv.firstFailingTurn(s, Json.MAPPER.createObjectNode());
				if (schedule.equals("static"))
					crossingStatic = failedAt;
				boolean documented = failedAt > 1 && conv.lastError.contains("original context length");
				checks.add(new ProbeReport.Check(schedule + ": without the opt-in a later turn fails at the limit,"
						+ " naming the original context length", documented,
						(failedAt == 0 ? "every turn answered" : "turn " + failedAt + ": " + conv.lastError)
								+ "; prompt tokens per turn " + conv.promptTokens));
				facts.put(schedule + ".crossingTurn", failedAt);
			}
		}
		if (crossingStatic > 0) {
			try (JunoProcess s = JunoProcess.serve(work.resolve("on-static"), heap,
					serverArgs(model, gpu, "static", "on", replyTokens), START)) {
				Sent byDefault = send(s, conv.history(crossingStatic), Json.MAPPER.createObjectNode(), replyTokens, 0);
				Sent refused = send(s, conv.history(crossingStatic),
						Json.MAPPER.createObjectNode().put("x_juno_context_shift", false), replyTokens, 0);
				checks.add(new ProbeReport.Check("static, --context-shift on: the crossing turn answers by default and"
						+ " fails with an explicit false", byDefault.status == 200 && refused.status != 200,
						"default HTTP " + byDefault.status + ", explicit false HTTP " + refused.status));
			}
		} else if (args.getOrDefault("schedules", "static,continuous").contains("static")) {
			checks.add(new ProbeReport.Check("static, --context-shift on: the crossing turn answers by default and"
					+ " fails with an explicit false", false, "not run: no crossing turn known from the static run"));
		}
		facts.put("limit", limit);
		facts.put("gpu", gpu);
		return new ProbeReport(id, ProbeKind.CHECK, checks.stream().allMatch(ProbeReport.Check::passed), checks,
				List.of(), facts);
	}

	private static List<String> serverArgs(Path model, boolean gpu, String schedule, String shift, int replyTokens) {
		List<String> a = JunoProcess.localArgs(model, replyTokens);
		a.addAll(List.of(gpu ? "--gpu" : "--cpu", "--schedule", schedule, "--context-shift", shift));
		return a;
	}

	private static ArrayNode single(String user) {
		ArrayNode m = Json.MAPPER.createArrayNode();
		m.addObject().put("role", "system").put("content", "Answer briefly.");
		m.addObject().put("role", "user").put("content", user);
		return m;
	}

	record Sent(int status, JsonNode json, String body) {
	}

	static Sent send(JunoProcess s, ArrayNode messages, ObjectNode extra, int maxTokens, int minTokens)
			throws Exception {
		ObjectNode b = Json.MAPPER.createObjectNode();
		b.set("messages", messages);
		b.put("temperature", 0).put("max_tokens", maxTokens).put("min_tokens", minTokens);
		b.setAll(extra);
		JunoProcess.Answer a = s.post(GrammarProbe.PATH, b.toString());
		JsonNode json;
		try {
			json = a.json();
		} catch (Exception e) {
			json = Json.MAPPER.missingNode();
		}
		return new Sent(a.status(), json, a.body());
	}

	/** The growing conversation; each turn adds {@code notesPerTurn} new notes and a canned reply. */
	static final class Conversation {
		private final int turns;
		private final int notesPerTurn;
		private final int replyTokens;
		String lastError = "";
		final List<Integer> promptTokens = new ArrayList<>();

		Conversation(int turns, int notesPerTurn, int replyTokens) {
			this.turns = turns;
			this.notesPerTurn = notesPerTurn;
			this.replyTokens = replyTokens;
		}

		/** The messages up to and including user turn {@code n}. */
		ArrayNode history(int n) {
			ArrayNode m = Json.MAPPER.createArrayNode();
			m.addObject().put("role", "system").put("content", LEDGER);
			for (int t = 0; t < n; t++) {
				m.addObject().put("role", "user").put("content",
						"Add these notes to the ledger:\n" + LedgerNotes.notes(t * notesPerTurn + 1, notesPerTurn));
				if (t + 1 < n)
					m.addObject().put("role", "assistant").put("content", "Noted.");
			}
			return m;
		}

		/** Sends turns 1 to {@code turns}; the first that is not HTTP 200, or 0. */
		int firstFailingTurn(JunoProcess s, ObjectNode extra) throws Exception {
			promptTokens.clear();
			for (int t = 1; t <= turns; t++) {
				Sent r = send(s, history(t), extra, replyTokens, 0);
				promptTokens.add(r.json.path("usage").path("prompt_tokens").asInt(0));
				if (r.status != 200) {
					lastError = "HTTP " + r.status + ": " + (r.body.length() > 300 ? r.body.substring(0, 300) : r.body);
					return t;
				}
			}
			lastError = "";
			return 0;
		}
	}
}
