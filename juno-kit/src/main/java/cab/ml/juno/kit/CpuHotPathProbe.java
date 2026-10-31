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
import java.util.Locale;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * The CPU hot-path smoke's text checks on one model, through the launcher on the CPU at two {@code --threads}
 * values: the server logs the thread count it runs matmuls on; a short greedy request decodes; a prompt of
 * about 512 tokens (461 to 563, calibrated against the engine's count) prefills and answers; and both replies
 * are identical at every thread count, since a row is computed by one thread whatever the count. The replies
 * are recorded, so {@code ingest} compares them with the reference run's, as the kernel smoke compares two
 * builds.
 */
final class CpuHotPathProbe implements KitProbe {

	private static final Duration START = Duration.ofMinutes(10);
	private static final int MAX_TOKENS = 16;

	@Override
	public ProbeReport run(String id, Map<String, String> args) throws Exception {
		Path model = Path.of(args.get("model"));
		String heap = args.getOrDefault("serverHeap", "4g");
		Path work = Path.of(args.getOrDefault("logDir", ".")).resolve(id);
		List<Integer> counts = new ArrayList<>();
		for (String t : args.getOrDefault("threads", String.join(",", defaultThreads(Runtime.getRuntime()
				.availableProcessors()))).split(","))
			counts.add(Integer.parseInt(t.trim()));
		List<ProbeReport.Check> checks = new ArrayList<>();
		Map<String, Object> facts = new LinkedHashMap<>();
		int words = 390;
		String firstShort = null;
		String firstLong = null;
		for (int t : counts) {
			String label = "threads " + t + ": ";
			List<String> a = JunoProcess.localArgs(model, MAX_TOKENS);
			a.addAll(List.of("--cpu", "--threads", String.valueOf(t)));
			try (JunoProcess s = JunoProcess.serve(work.resolve("threads" + t), heap, a, START)) {
				boolean logged = Pattern.compile("CPU matmul threads: " + t + "\\b").matcher(s.log()).find();
				checks.add(new ProbeReport.Check(label + "the server runs CPU matmuls on " + t + " threads", logged,
						logged ? "logged" : "no 'CPU matmul threads: " + t + "' line in the server log"));
				String m = s.modelId();
				JunoProcess.Answer shortA = s.post(GrammarProbe.PATH, GrammarProbe.body(m, "Name three colours.",
						MAX_TOKENS).toString());
				String shortText = GrammarProbe.content(shortA);
				checks.add(new ProbeReport.Check(label + "a short greedy request decodes",
						shortA.status() == 200 && !shortText.isBlank(), GrammarProbe.describe(shortA)));
				JunoProcess.Answer longA = null;
				int pt = 0;
				for (int attempt = 0; attempt < 3; attempt++) {
					longA = s.post(GrammarProbe.PATH, GrammarProbe.body(m, longPrompt(words), 4).toString());
					pt = longA.status() == 200 ? longA.json().path("usage").path("prompt_tokens").asInt(0) : 0;
					if (pt <= 0 || (pt >= 461 && pt <= 563))
						break;
					words = words * 512 / pt;
				}
				String longText = GrammarProbe.content(longA);
				checks.add(new ProbeReport.Check(label + "a prompt of about 512 tokens prefills and answers",
						longA.status() == 200 && !longText.isBlank() && pt >= 461 && pt <= 563,
						pt + " prompt tokens; " + GrammarProbe.describe(longA)));
				facts.put("threads" + t + ".short", shortText);
				facts.put("threads" + t + ".long", longText);
				if (firstShort == null) {
					firstShort = shortText;
					firstLong = longText;
				} else {
					boolean same = shortText.equals(firstShort) && longText.equals(firstLong);
					checks.add(new ProbeReport.Check("greedy replies identical at " + counts.get(0) + " and " + t
							+ " threads", same, same ? "identical" : "short '" + firstShort + "' / '" + shortText
									+ "'; long '" + firstLong + "' / '" + longText + "'"));
				}
			}
		}
		facts.put("threads", counts);
		return new ProbeReport(id, ProbeKind.CHECK, checks.stream().allMatch(ProbeReport.Check::passed), checks,
				List.of(), facts);
	}

	/** 4 and every processor, as the smoke runs; on a machine with 4 or fewer, half of them and all. */
	static List<String> defaultThreads(int processors) {
		int low = processors > 4 ? 4 : Math.max(1, processors / 2);
		return low == processors ? List.of(String.valueOf(low))
				: List.of(String.valueOf(low), String.valueOf(processors));
	}

	/** The smoke's long prompt: numbered notes and a question, about 1.3 tokens per word. */
	static String longPrompt(int words) {
		StringBuilder b = new StringBuilder("Read the notes below, then answer the question at the end.\n");
		for (int i = 1; i * 12 <= words; i++)
			b.append(String.format(Locale.ROOT, "Note %d: the bridge in district %d was painted in the year %d by crew %d.\n",
					i, i * 7 % 31, 1800 + i * 13, i * 3 % 17));
		return b.append("Question: in which year was the bridge in note 1 painted? Answer in one sentence.").toString();
	}
}
