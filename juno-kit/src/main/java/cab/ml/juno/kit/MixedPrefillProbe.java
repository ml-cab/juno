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

import cab.ml.juno.player.JunoHttpClient;

/**
 * The continuous schedule's mixed chunked prefill against full prefill at admission, as the mixed-prefill script
 * runs it: one stream with a long prompt and {@code shorts} streams with a short one, started together, on a server
 * with {@code -Djuno.continuous.mixedPrefill=true} and then {@code false}. Short-request time to first token and per
 * output token show whether the long prompt starves them; the long request's time to first token is its cost. In
 * mixed mode the server's recording must show the long prompt's chunks prefilled inside decode steps.
 */
final class MixedPrefillProbe implements KitProbe {

	static final String SHORT_PROMPT = "Say hi in one word.";
	private static final Duration START = Duration.ofMinutes(10);

	@Override
	public ProbeReport run(String id, Map<String, String> args) throws Exception {
		Path model = Path.of(args.get("model"));
		String heap = args.getOrDefault("serverHeap", "4g");
		Path work = Path.of(args.getOrDefault("logDir", ".")).resolve(id);
		boolean gpu = Boolean.parseBoolean(args.getOrDefault("gpu", "false"));
		int nPrompt = KitProbe.intArg(args, "nPrompt", 256);
		int shorts = KitProbe.intArg(args, "shorts", 3);
		int maxTokens = KitProbe.intArg(args, "maxTokens", 16);
		String longPrompt = "Please summarize this passage carefully. " + String.join(" ", Collections.nCopies(nPrompt,
				"word"));
		List<String> prompts = new ArrayList<>(List.of(longPrompt));
		prompts.addAll(Collections.nCopies(shorts, SHORT_PROMPT));
		List<ProbeReport.Check> checks = new ArrayList<>();
		List<ProbeReport.Reading> readings = new ArrayList<>();
		Map<String, Object> facts = new LinkedHashMap<>();
		for (String mode : List.of("mixed", "admit")) {
			List<String> a = JunoProcess.localArgs(model, maxTokens);
			a.addAll(List.of(gpu ? "--gpu" : "--cpu", "--schedule", "continuous", "--parallel",
					args.getOrDefault("parallel", "8"), "--prefill-batch", args.getOrDefault("prefillBatch", "32"),
					"--batch-window-ms", args.getOrDefault("batchWindowMs", "50"), "--jfr", "30m"));
			List<String> props = List.of("-Djuno.continuous.mixedPrefill=" + mode.equals("mixed"));
			JunoProcess s = JunoProcess.serve(work.resolve(mode), heap, props, a, START);
			List<StreamTiming.Result> r;
			try {
				r = StreamTiming.concurrent(new JunoHttpClient(s.base()), s.modelId(), prompts, maxTokens);
			} finally {
				s.close();
			}
			long ok = r.stream().filter(StreamTiming.Result::ok).count();
			checks.add(new ProbeReport.Check(mode + ": every stream answered", ok == r.size(),
					ok + " of " + r.size() + " streams produced text"));
			JsonNode jfr = s.metrics();
			int chunks = jfr.path("juno.ContinuousStep.prefill_chunks").asInt(0);
			int steps = jfr.path("juno.ContinuousStep.steps_with_prefill").asInt(0);
			if (mode.equals("mixed"))
				checks.add(new ProbeReport.Check("mixed: the long prompt was prefilled in chunks inside decode steps",
						chunks >= 2, jfr.isMissingNode() ? "no JFR summary in the server's directory"
								: chunks + " prefill chunks in " + steps + " steps"));
			List<StreamTiming.Result> shortOnes = r.subList(1, r.size());
			readings.add(new ProbeReport.Reading(mode + ": short-request TTFT", "ms",
					KitProbe.one(StreamTiming.mean(shortOnes.stream().map(StreamTiming.Result::ttftMs).toList())),
					false, false));
			readings.add(new ProbeReport.Reading(mode + ": short-request TPOT", "ms",
					KitProbe.one(StreamTiming.mean(shortOnes.stream().map(StreamTiming.Result::tpotMs).toList())),
					false, false));
			readings.add(new ProbeReport.Reading(mode + ": long-request TTFT", "ms", KitProbe.one(r.get(0).ttftMs()),
					false, false));
			facts.put(mode + ".prefillChunks", chunks);
			facts.put(mode + ".stepsWithPrefill", steps);
		}
		facts.put("longPromptWords", nPrompt);
		facts.put("shortRequests", shorts);
		facts.put("backend", gpu ? "gpu" : "cpu");
		return new ProbeReport(id, ProbeKind.CHECK, checks.stream().allMatch(ProbeReport.Check::passed), checks,
				readings, facts);
	}
}
