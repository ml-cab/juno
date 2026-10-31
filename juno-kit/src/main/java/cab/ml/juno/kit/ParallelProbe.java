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


import java.io.IOException;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * {@code --parallel} at two or more settings over the same concurrent clients, as the parallel script runs it:
 * {@code sessions} greedy chats of one prompt, sent at once to a server started at each setting, after untimed
 * warmup rounds (a cold first round reads well below a warm one). Aggregate throughput is every completion token
 * over the round's wall time. The settings alternate between repetitions (1 8, then 8 1). At {@code --parallel 1}
 * the server decodes each request on its own thread, so the comparison is batched against concurrent unbatched
 * decoding, not against a queue; the ratio is host data, not a gain to expect. Each setting's distinct replies are
 * recorded: greedy requests of one prompt should agree.
 */
final class ParallelProbe implements KitProbe {

	static final String PROMPT = "could you please write me a short poem about love and war";
	private static final Duration START = Duration.ofMinutes(10);

	@Override
	public ProbeReport run(String id, Map<String, String> args) throws Exception {
		Path model = Path.of(args.get("model"));
		String heap = args.getOrDefault("serverHeap", "4g");
		Path work = Path.of(args.getOrDefault("logDir", ".")).resolve(id);
		boolean gpu = Boolean.parseBoolean(args.getOrDefault("gpu", "false"));
		int sessions = KitProbe.intArg(args, "sessions", 8);
		int maxTokens = KitProbe.intArg(args, "maxTokens", 64);
		int warmups = KitProbe.intArg(args, "warmups", 2);
		int reps = KitProbe.intArg(args, "reps", 3);
		String window = args.getOrDefault("batchWindowMs", "50");
		List<Integer> values = new ArrayList<>();
		for (String v : args.getOrDefault("parallel", "1,8").split(","))
			values.add(Integer.parseInt(v.trim()));
		Map<Integer, List<Double>> tps = new LinkedHashMap<>();
		Map<Integer, List<Double>> latency = new LinkedHashMap<>();
		Map<Integer, Set<String>> replies = new LinkedHashMap<>();
		for (int p : values) {
			tps.put(p, new ArrayList<>());
			latency.put(p, new ArrayList<>());
			replies.put(p, new LinkedHashSet<>());
		}
		for (int rep = 1; rep <= reps; rep++) {
			List<Integer> order = new ArrayList<>(values);
			if (rep % 2 == 0)
				order = order.reversed();
			for (int p : order) {
				List<String> a = JunoProcess.localArgs(model, maxTokens);
				a.addAll(List.of(gpu ? "--gpu" : "--cpu", "--parallel", String.valueOf(p), "--batch-window-ms", window));
				try (JunoProcess s = JunoProcess.serve(work.resolve("parallel" + p + "-rep" + rep), heap, a, START)) {
					String m = s.modelId();
					for (int w = 0; w < warmups; w++)
						ConcurrentChats.run(s, m, PROMPT, sessions, maxTokens);
					ConcurrentChats.Round r = ConcurrentChats.run(s, m, PROMPT, sessions, maxTokens);
					if (!r.allOk())
						throw new IOException("--parallel " + p + ", repetition " + rep + ": " + r.failures());
					tps.get(p).add(r.aggregateTps());
					latency.get(p).add(r.meanLatencyMs());
					replies.get(p).addAll(r.replies());
				}
			}
		}
		List<ProbeReport.Reading> readings = new ArrayList<>();
		Map<String, Object> facts = new LinkedHashMap<>();
		for (int p : values) {
			readings.add(new ProbeReport.Reading("aggregate t/s, --parallel " + p, "t/s", tps.get(p), true, true));
			readings.add(new ProbeReport.Reading("mean request latency, --parallel " + p, "ms", latency.get(p), false,
					false));
			facts.put("parallel" + p + ".distinctReplies", replies.get(p).size());
		}
		int first = values.get(0);
		int last = values.get(values.size() - 1);
		facts.put("parallel" + last + "OverParallel" + first,
				readings.get(2 * (values.size() - 1)).median() / readings.get(0).median());
		facts.put("sessions", sessions);
		facts.put("maxTokens", maxTokens);
		facts.put("warmupRounds", warmups);
		facts.put("backend", gpu ? "gpu" : "cpu");
		return new ProbeReport(id, ProbeKind.READING, null, List.of(), readings, facts);
	}
}
