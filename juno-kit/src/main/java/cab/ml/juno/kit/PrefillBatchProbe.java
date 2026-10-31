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
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.databind.JsonNode;

/**
 * Prefill throughput at two or more {@code --prefill-batch} sizes, as the prefill-batch script reads it: a server
 * with a JFR recording per size and repetition answers one request whose prompt is {@code nPrompt} words of
 * {@code x}; prefill t/s is its prompt tokens over the recording's prefill forward-pass time. The sizes alternate
 * between repetitions. Each server is fresh, so a reading includes the JIT's first pass over the prefill path, as
 * the script's does.
 */
final class PrefillBatchProbe implements KitProbe {

	private static final Duration START = Duration.ofMinutes(10);
	private static final String PREFILL_MS = "juno.ForwardPass.prefill.total_ms";

	@Override
	public ProbeReport run(String id, Map<String, String> args) throws Exception {
		Path model = Path.of(args.get("model"));
		String heap = args.getOrDefault("serverHeap", "4g");
		Path work = Path.of(args.getOrDefault("logDir", ".")).resolve(id);
		boolean gpu = Boolean.parseBoolean(args.getOrDefault("gpu", "false"));
		int maxTokens = KitProbe.intArg(args, "maxTokens", 8);
		int reps = KitProbe.intArg(args, "reps", 3);
		String prompt = String.join(" ", Collections.nCopies(KitProbe.intArg(args, "nPrompt", 256), "x"));
		List<Integer> values = new ArrayList<>();
		for (String v : args.getOrDefault("values", "1,32").split(","))
			values.add(Integer.parseInt(v.trim()));
		Map<Integer, List<Double>> tps = new LinkedHashMap<>();
		Map<String, Object> facts = new LinkedHashMap<>();
		for (int v : values)
			tps.put(v, new ArrayList<>());
		for (int rep = 1; rep <= reps; rep++) {
			List<Integer> order = new ArrayList<>(values);
			if (rep % 2 == 0)
				order = order.reversed();
			for (int v : order) {
				List<String> a = JunoProcess.localArgs(model, maxTokens);
				a.addAll(List.of(gpu ? "--gpu" : "--cpu", "--prefill-batch", String.valueOf(v), "--jfr", "10m"));
				String label = "--prefill-batch " + v + ", repetition " + rep;
				JunoProcess s = JunoProcess.serve(work.resolve("prefill-batch" + v + "-rep" + rep), heap, a, START);
				JunoProcess.Answer ans;
				try {
					ans = s.post(GrammarProbe.PATH, GrammarProbe.body(s.modelId(), prompt, maxTokens).toString());
				} finally {
					s.close();
				}
				if (ans.status() != 200)
					throw new IOException(label + ": " + GrammarProbe.describe(ans));
				int promptTokens = ans.json().path("usage").path("prompt_tokens").asInt(0);
				JsonNode jfr = s.metrics();
				double ms = jfr.path(PREFILL_MS).asDouble(0);
				if (promptTokens <= 0 || ms <= 0)
					throw new IOException(label + ": no prefill time in the server's JFR summary (" + promptTokens
							+ " prompt tokens, " + ms + " ms)");
				tps.get(v).add(promptTokens / (ms / 1000.0));
				facts.put("prefillBatch" + v + ".promptTokens", promptTokens);
				facts.put("prefillBatch" + v + ".prefillCalls", jfr.path("juno.ForwardPass.prefill.count").asInt(0));
			}
		}
		List<ProbeReport.Reading> readings = new ArrayList<>();
		for (int v : values)
			readings.add(new ProbeReport.Reading("prefill t/s, --prefill-batch " + v, "t/s", tps.get(v), true, true));
		facts.put("backend", gpu ? "gpu" : "cpu");
		return new ProbeReport(id, ProbeKind.READING, null, List.of(), readings, facts);
	}
}
