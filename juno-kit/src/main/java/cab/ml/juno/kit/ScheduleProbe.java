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
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import cab.ml.juno.player.JunoHttpClient;

/**
 * The static and continuous schedules on the schedule script's three workloads, one server per schedule at
 * {@code --parallel} with a JFR recording: {@code sessions} concurrent blocking chats (aggregate throughput, after
 * one untimed round); as many concurrent streams (mean time to first token and per output token); and one
 * session of {@code turns} turns under a shared system prompt and {@code x_juno_session_id} (first against later
 * turn latency, prefix-cache lookups and hits from the health endpoint, counted over that session alone). Checks:
 * the health endpoint reports the schedule asked for, every request answers, and on the continuous schedule the
 * recording shows decode steps shared by concurrent requests. The continuous server also releases idle KV pages
 * every {@value #RECLAIM_PERIOD_MS} ms: once the workloads end and the health endpoint counts pages reclaimed, a
 * greedy request taken before them gets the same reply again, on pages made after the reclaim.
 */
final class ScheduleProbe implements KitProbe {

	static final String SYSTEM_PROMPT = "You are a helpful assistant that answers briefly and accurately.";
	private static final Duration START = Duration.ofMinutes(10);
	/** The continuous server's KV page reclaim period, short so that the probe sees pages released. */
	static final int RECLAIM_PERIOD_MS = 500;
	private static final Duration RECLAIM_WAIT = Duration.ofSeconds(30);

	@Override
	public ProbeReport run(String id, Map<String, String> args) throws Exception {
		Path model = Path.of(args.get("model"));
		String heap = args.getOrDefault("serverHeap", "4g");
		Path work = Path.of(args.getOrDefault("logDir", ".")).resolve(id);
		boolean gpu = Boolean.parseBoolean(args.getOrDefault("gpu", "false"));
		int sessions = KitProbe.intArg(args, "sessions", 8);
		int maxTokens = KitProbe.intArg(args, "maxTokens", 64);
		int turns = KitProbe.intArg(args, "turns", 8);
		String parallel = args.getOrDefault("parallel", "8");
		List<ProbeReport.Check> checks = new ArrayList<>();
		List<ProbeReport.Reading> readings = new ArrayList<>();
		Map<String, Object> facts = new LinkedHashMap<>();
		for (String schedule : List.of("static", "continuous")) {
			List<String> a = JunoProcess.localArgs(model, maxTokens);
			a.addAll(List.of(gpu ? "--gpu" : "--cpu", "--parallel", parallel, "--batch-window-ms",
					args.getOrDefault("batchWindowMs", "50"), "--schedule", schedule, "--jfr", "30m"));
			boolean continuous = schedule.equals("continuous");
			List<String> props = continuous ? List.of("-Djuno.kv.reclaimPeriodMs=" + RECLAIM_PERIOD_MS) : List.of();
			JunoProcess s = JunoProcess.serve(work.resolve(schedule), heap, props, a, START);
			ConcurrentChats.Round tps;
			List<StreamTiming.Result> streams;
			List<Double> turnMs = new ArrayList<>();
			int turnsOk = 0;
			long lookups;
			long hits;
			String beforeReclaim = null;
			String afterReclaim = null;
			long reclaimed = 0;
			long liveAfter = 0;
			try {
				String got = s.get("/v1/cluster/health").json().path("schedule").asText("");
				checks.add(new ProbeReport.Check(schedule + ": the server runs the " + schedule + " schedule",
						got.equals(schedule), "the health endpoint reports '" + got + "'"));
				String m = s.modelId();
				if (continuous)
					beforeReclaim = greedy(s, m, maxTokens);
				ConcurrentChats.run(s, m, ParallelProbe.PROMPT, sessions, maxTokens);
				tps = ConcurrentChats.run(s, m, ParallelProbe.PROMPT, sessions, maxTokens);
				streams = StreamTiming.concurrent(new JunoHttpClient(s.base()), m,
						Collections.nCopies(sessions, ParallelProbe.PROMPT), maxTokens);
				JsonNode before = s.get("/v1/cluster/health").json();
				ObjectNode body = Json.MAPPER.createObjectNode().put("model", m).put("max_tokens", maxTokens)
						.put("temperature", 0).put("stream", false)
						.put("x_juno_session_id", "kit-prefix-" + schedule + "-" + System.nanoTime());
				ArrayNode history = body.putArray("messages");
				history.addObject().put("role", "system").put("content", SYSTEM_PROMPT);
				for (int t = 1; t <= turns; t++) {
					history.addObject().put("role", "user").put("content", "Turn " + t + ": name one color.");
					long t0 = System.nanoTime();
					JunoProcess.Answer ans = s.post(GrammarProbe.PATH, body.toString());
					turnMs.add((System.nanoTime() - t0) / 1e6);
					if (ans.status() == 200) {
						turnsOk++;
						history.addObject().put("role", "assistant").put("content", GrammarProbe.content(ans));
					}
				}
				JsonNode after = s.get("/v1/cluster/health").json();
				lookups = after.path("prefixLookups").asLong(0) - before.path("prefixLookups").asLong(0);
				hits = after.path("prefixHits").asLong(0) - before.path("prefixHits").asLong(0);
				if (continuous) {
					long deadline = System.nanoTime() + RECLAIM_WAIT.toNanos();
					JsonNode h = s.get("/v1/cluster/health").json();
					while (h.path("kvPagesReclaimed").asLong(0) == 0 && System.nanoTime() < deadline) {
						Thread.sleep(200);
						h = s.get("/v1/cluster/health").json();
					}
					reclaimed = h.path("kvPagesReclaimed").asLong(0);
					liveAfter = h.path("kvPagesLive").asLong(0);
					afterReclaim = greedy(s, m, maxTokens);
				}
			} finally {
				s.close();
			}
			long streamsOk = streams.stream().filter(StreamTiming.Result::ok).count();
			checks.add(new ProbeReport.Check(schedule + ": every request answered",
					tps.allOk() && streamsOk == sessions && turnsOk == turns, "blocking " + (tps.allOk() ? "all"
							: tps.failures()) + "; streams " + streamsOk + "/" + sessions + "; turns " + turnsOk + "/"
							+ turns));
			if (continuous) {
				boolean same = beforeReclaim != null && beforeReclaim.equals(afterReclaim);
				checks.add(new ProbeReport.Check("continuous: idle KV pages reclaimed, replies unchanged",
						reclaimed > 0 && same, reclaimed + " pages reclaimed within " + RECLAIM_WAIT.toSeconds()
								+ " s, " + liveAfter + " still live (the session's); the greedy reply after the"
								+ " reclaim " + (same ? "equals" : "differs from") + " the one before"));
				facts.put("continuous.kvPagesReclaimed", reclaimed);
				facts.put("continuous.kvPagesLiveAfterReclaim", liveAfter);
				JsonNode jfr = s.metrics();
				double maxBatch = jfr.path("juno.ContinuousStep.max_decode_batch").asDouble(0);
				double shared = jfr.path("juno.ContinuousStep.shared_steps").asDouble(0);
				checks.add(new ProbeReport.Check("continuous: concurrent requests shared decode steps",
						maxBatch >= 2 || shared >= 1, jfr.isMissingNode() ? "no JFR summary in the server's directory"
								: "largest decode batch " + (int) maxBatch + ", shared steps " + (int) shared));
			}
			readings.add(new ProbeReport.Reading(schedule + ": aggregate t/s", "t/s", KitProbe.one(tps.aggregateTps()),
					true, true));
			readings.add(new ProbeReport.Reading(schedule + ": mean TTFT", "ms",
					KitProbe.one(StreamTiming.mean(streams.stream().map(StreamTiming.Result::ttftMs).toList())), false,
					false));
			readings.add(new ProbeReport.Reading(schedule + ": mean TPOT", "ms",
					KitProbe.one(StreamTiming.mean(streams.stream().map(StreamTiming.Result::tpotMs).toList())), false,
					false));
			readings.add(new ProbeReport.Reading(schedule + ": first turn latency", "ms",
					KitProbe.one(turnMs.isEmpty() ? Double.NaN : turnMs.get(0)), false, false));
			readings.add(new ProbeReport.Reading(schedule + ": mean later turn latency", "ms",
					KitProbe.one(StreamTiming.mean(turnMs.subList(Math.min(1, turnMs.size()), turnMs.size()))), false,
					false));
			facts.put(schedule + ".prefixLookups", lookups);
			facts.put(schedule + ".prefixHits", hits);
			facts.put(schedule + ".prefixHitRate", lookups > 0 ? (double) hits / lookups : 0.0);
		}
		facts.put("sessions", sessions);
		facts.put("parallel", Integer.parseInt(parallel));
		facts.put("backend", gpu ? "gpu" : "cpu");
		return new ProbeReport(id, ProbeKind.CHECK, checks.stream().allMatch(ProbeReport.Check::passed), checks,
				readings, facts);
	}

	/** One greedy chat reply to the parallel probe's prompt; null when the request fails. */
	private static String greedy(JunoProcess s, String model, int maxTokens) throws Exception {
		ObjectNode body = Json.MAPPER.createObjectNode().put("model", model).put("max_tokens", maxTokens)
				.put("temperature", 0).put("stream", false);
		body.putArray("messages").addObject().put("role", "user").put("content", ParallelProbe.PROMPT);
		JunoProcess.Answer ans = s.post(GrammarProbe.PATH, body.toString());
		return ans.status() == 200 ? GrammarProbe.content(ans) : null;
	}
}
