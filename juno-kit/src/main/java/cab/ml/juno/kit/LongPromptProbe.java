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
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Flow;
import java.util.concurrent.TimeUnit;

import com.fasterxml.jackson.databind.JsonNode;

import cab.ml.juno.node.GgufReader;
import cab.ml.juno.player.JunoHttpClient;
import cab.ml.juno.tokenizer.ChatMessage;

/**
 * The long-prompt smoke's checks over OpenAI REST, through the launcher in a child JVM: for each schedule and
 * each prompt length (capped where the model's context cannot hold it with the generated tokens), a greedy
 * request answers with HTTP 200 and text, its prompt is within 10% of the length, and the same request
 * streamed returns the same text. Time to first token, the greedy text (which {@code ingest} compares with
 * the reference run's) and whether the answer names the right colour are recorded. The note and filler-word
 * counts per length are calibrated once against the engine's own prompt token counts.
 */
final class LongPromptProbe implements KitProbe {

	private static final Duration START = Duration.ofMinutes(10);

	@Override
	public ProbeReport run(String id, Map<String, String> args) throws Exception {
		Path model = Path.of(args.get("model"));
		boolean gpu = Boolean.parseBoolean(args.getOrDefault("gpu", "false"));
		int maxTokens = KitProbe.intArg(args, "maxTokens", 16);
		String heap = args.getOrDefault("serverHeap", "4g");
		Path work = Path.of(args.getOrDefault("logDir", ".")).resolve(id);
		List<Integer> lengths = new ArrayList<>();
		for (String l : args.getOrDefault("lengths", "128,512,2048").split(","))
			lengths.add(Integer.parseInt(l.trim()));
		int context = contextLength(model);
		List<ProbeReport.Check> checks = new ArrayList<>();
		Map<String, Object> facts = new LinkedHashMap<>();
		facts.put("context", context);
		Map<Integer, int[]> plan = null;
		for (String schedule : args.getOrDefault("schedules", "static,continuous").split(",")) {
			List<String> serverArgs = JunoProcess.localArgs(model, maxTokens);
			serverArgs.addAll(List.of(gpu ? "--gpu" : "--cpu", "--schedule", schedule));
			try (JunoProcess s = JunoProcess.serve(work.resolve(schedule), heap, serverArgs, START)) {
				if (plan == null) {
					plan = calibrate(s, lengths, context, maxTokens, KitProbe.intArg(args, "calibA", 2),
							KitProbe.intArg(args, "calibB", 40));
					if (plan == null) {
						checks.add(new ProbeReport.Check("prompt calibration against the engine's token counts", false,
								"a calibration request failed; see " + id + "/" + schedule + "/" + JunoProcess.LOG));
						break;
					}
					plan.forEach((len, p) -> facts.put("plan." + len, p[0] + " notes + " + p[1] + " filler words, target "
							+ p[2] + " tokens"));
				}
				JunoHttpClient client = new JunoHttpClient(s.base());
				String modelId = s.modelId();
				for (int len : lengths) {
					int[] p = plan.get(len);
					String prompt = LedgerNotes.question(p[0], p[1]);
					String label = schedule + " " + len + ": ";
					JunoProcess.Answer a = s.post(GrammarProbe.PATH, GrammarProbe.body(modelId, prompt, maxTokens)
							.toString());
					String text = GrammarProbe.content(a);
					int pt = a.status() == 200 ? a.json().path("usage").path("prompt_tokens").asInt(0) : 0;
					checks.add(new ProbeReport.Check(label + "HTTP 200, non-empty answer",
							a.status() == 200 && !text.isBlank(), GrammarProbe.describe(a)));
					checks.add(new ProbeReport.Check(label + "prompt tokens within 10% of " + p[2],
							Math.abs(pt - p[2]) <= 0.10 * p[2], pt + " prompt tokens"));
					Streamed st = stream(client, modelId, prompt, maxTokens);
					checks.add(new ProbeReport.Check(label + "streamed text equals non-streamed",
							st.error == null && st.text.equals(text),
							st.error != null ? st.error : st.text.equals(text) ? text.length() + " characters"
									: "streamed '" + st.text + "', non-streamed '" + text + "'"));
					facts.put(schedule + "." + len + ".ttftMs", st.ttftMs);
					facts.put(schedule + "." + len + ".text", text);
					facts.put(schedule + "." + len + ".answerNamesColour",
							text.toLowerCase(Locale.ROOT).contains(LedgerNotes.COLOURS.get(3)));
				}
			}
		}
		return new ProbeReport(id, ProbeKind.CHECK, checks.stream().allMatch(ProbeReport.Check::passed), checks,
				List.of(), facts);
	}

	/** {@code <arch>.context_length} from the model file, 0 when absent. */
	static int contextLength(Path model) throws Exception {
		try (GgufReader r = GgufReader.open(model)) {
			String arch = r.metaString("general.architecture");
			return arch == null ? 0 : r.metaInt(arch + ".context_length", 0);
		}
	}

	/**
	 * Per length: notes, filler words, target tokens. A linear fit over {@code a} and {@code b} notes, then
	 * filler words for the remainder with one correction; tokenization does not depend on the schedule.
	 */
	private static Map<Integer, int[]> calibrate(JunoProcess s, List<Integer> lengths, int context, int maxTokens,
			int a, int b) throws Exception {
		int ta = promptTokens(s, a, 0);
		int tb = promptTokens(s, b, 0);
		if (ta <= 0 || tb <= ta)
			return null;
		int per = (tb - ta) * 100 / (b - a);
		Map<Integer, int[]> plan = new LinkedHashMap<>();
		for (int len : lengths) {
			int target = len;
			if (context > 0 && target + maxTokens + 8 > context)
				target = context - maxTokens - 8;
			int n = Math.max(1, a + (target - ta) * 100 / per);
			int t = promptTokens(s, n, 0);
			if (t <= 0)
				return null;
			int w = Math.max(0, target - t);
			if (w > 0) {
				t = promptTokens(s, n, w);
				if (t <= 0)
					return null;
				w = Math.max(0, w + target - t);
			}
			plan.put(len, new int[] { n, w, target });
		}
		return plan;
	}

	private static int promptTokens(JunoProcess s, int notes, int words) throws Exception {
		JunoProcess.Answer a = s.post(GrammarProbe.PATH,
				GrammarProbe.body(s.modelId(), LedgerNotes.question(notes, words), 1).toString());
		if (a.status() != 200)
			return 0;
		JsonNode u = a.json().path("usage");
		return u.path("prompt_tokens").asInt(0);
	}

	record Streamed(String text, double ttftMs, String error) {
	}

	private static Streamed stream(JunoHttpClient client, String modelId, String prompt, int maxTokens)
			throws InterruptedException {
		StringBuilder text = new StringBuilder();
		double[] ttft = { -1 };
		String[] error = { null };
		CountDownLatch done = new CountDownLatch(1);
		long t0 = System.nanoTime();
		client.streamingOpenAiChat(modelId, List.of(ChatMessage.user(prompt)), maxTokens, 0f)
				.subscribe(new Flow.Subscriber<String>() {
					@Override
					public void onSubscribe(Flow.Subscription sub) {
						sub.request(Long.MAX_VALUE);
					}

					@Override
					public void onNext(String piece) {
						if (ttft[0] < 0)
							ttft[0] = (System.nanoTime() - t0) / 1e6;
						text.append(piece);
					}

					@Override
					public void onError(Throwable t) {
						error[0] = String.valueOf(t.getMessage());
						done.countDown();
					}

					@Override
					public void onComplete() {
						done.countDown();
					}
				});
		if (!done.await(20, TimeUnit.MINUTES))
			return new Streamed(text.toString(), ttft[0], "the stream did not finish within 20 minutes");
		return new Streamed(text.toString(), ttft[0], error[0]);
	}
}
