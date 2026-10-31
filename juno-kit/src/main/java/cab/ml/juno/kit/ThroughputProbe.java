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
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.management.ManagementFactory;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import com.fasterxml.jackson.databind.JsonNode;

import cab.ml.juno.coordinator.GenerationResult;
import cab.ml.juno.coordinator.InferenceRequest;
import cab.ml.juno.coordinator.RequestPriority;
import cab.ml.juno.metrics.JfrMetricsCli;
import cab.ml.juno.player.JunoPlayer;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import jdk.jfr.Configuration;
import jdk.jfr.Recording;

/**
 * The throughput harness's two Juno lanes, in-process through {@link JunoPlayer}: a prefill lane (a prompt
 * of about {@code nPrompt} tokens, one generated token) and a generation lane (a two-token prompt,
 * {@code nGen} tokens forced). Each request is recorded with Juno's JFR settings and read off the same
 * spans the harness reads: prefill t/s is prompt tokens over the prefill spans, generation t/s is tokens
 * over the decode spans. A repetition whose spans do not fit inside its own request is withheld, as the
 * harness does. Allocation inside the recording and GC pause inside the token span are published beside
 * the throughput, not scored.
 */
final class ThroughputProbe implements KitProbe {

	static final String GENERATE_PROMPT = "Hi";
	private static final String PREFILL_MS = "juno.ForwardPass.prefill.total_ms";
	private static final String DECODE_MS = "juno.ForwardPass.decode.total_ms";
	private static final String ENCODE_MS = "juno.PromptEncode.total_ms";
	private static final String ALLOC = "jdk.ThreadAllocationStatistics.bytes_in_recording";
	private static final String GC_IN_SPAN = "jdk.GCPhasePause.in_token_span.total_ms";

	/** One measured request. */
	record Rep(int promptTokens, int generatedTokens, double latencyMs, double prefillMs, double decodeMs,
			double encodeMs, double allocBytes, double gcPauseMs) {

		/** The spans fit inside the request, with the harness's allowance for sampling and detokenization. */
		boolean consistent() {
			double resid = latencyMs - prefillMs - decodeMs - encodeMs;
			return prefillMs + decodeMs > 0 && resid >= -1 && resid <= 300 + 3.0 * generatedTokens;
		}

		/** Prompt tokens per second off the prefill spans, else off the request minus decode. */
		double prefillTps() {
			double ms = prefillMs > 0 ? prefillMs : latencyMs - decodeMs - encodeMs;
			return ms > 0 ? promptTokens / (ms / 1000.0) : Double.NaN;
		}

		double decodeTps() {
			return decodeMs > 0 ? generatedTokens / (decodeMs / 1000.0) : Double.NaN;
		}
	}

	@Override
	public ProbeReport run(String id, Map<String, String> args) throws Exception {
		Path model = Path.of(args.get("model"));
		int nPrompt = KitProbe.intArg(args, "nPrompt", 128);
		int nGen = KitProbe.intArg(args, "nGen", 64);
		int warmups = KitProbe.intArg(args, "warmups", 2);
		int reps = KitProbe.intArg(args, "reps", 3);
		int threads = KitProbe.intArg(args, "threads", Runtime.getRuntime().availableProcessors());
		boolean gpu = Boolean.parseBoolean(args.getOrDefault("gpu", "false"));
		Configuration jfc = jfrConfiguration();
		Path work = Files.createTempDirectory("juno-kit-jfr");
		Map<String, Object> facts = new LinkedHashMap<>();
		try (JunoPlayer player = JunoPlayer.builder(model).useGpu(gpu).threads(threads).build()) {
			// The chat template adds a fixed number of tokens; ask the engine and correct the word count.
			int overhead = request(player, words(nPrompt), 1, 0).promptTokens() - nPrompt;
			int words = Math.max(1, nPrompt - overhead);
			List<Rep> prefill = lane(player, jfc, work, words(words), 1, 0, warmups, reps, model);
			List<Rep> generate = lane(player, jfc, work, GENERATE_PROMPT, nGen, nGen, warmups, reps, model);

			int promptTokens = prefill.get(0).promptTokens();
			facts.put("model", model.getFileName().toString());
			facts.put("backend", gpu ? "gpu" : "cpu");
			facts.put("threads", threads);
			facts.put("nPrompt", nPrompt);
			facts.put("prefillPromptTokens", promptTokens);
			facts.put("promptParity", Math.abs(promptTokens - nPrompt) <= 0.10 * nPrompt);
			facts.put("generatePromptTokens", generate.get(0).promptTokens());
			facts.put("generatedTokens", generate.get(0).generatedTokens());
			facts.put("jfrClock", ManagementFactory.getRuntimeMXBean().getInputArguments()
					.contains("-XX:-UseFastUnorderedTimeStamps") ? "os" : "cpu-tsc");
			facts.put("prefillRepsWithheld", prefill.stream().filter(r -> !r.consistent()).count());
			facts.put("generateRepsWithheld", generate.stream().filter(r -> !r.consistent()).count());
			facts.put("prefillReps", prefill);
			facts.put("generateReps", generate);

			List<ProbeReport.Reading> readings = List.of(
					reading("prefill t/s", "t/s", prefill, Rep::prefillTps, true, true),
					reading("generation t/s", "t/s", generate, Rep::decodeTps, true, true),
					reading("generation allocation per token in recording", "B/token", generate,
							r -> r.allocBytes() / Math.max(1, r.generatedTokens()), false, false),
					reading("prefill GC pause in token span", "ms", prefill, Rep::gcPauseMs, false, false),
					reading("generation GC pause in token span", "ms", generate, Rep::gcPauseMs, false, false));
			return new ProbeReport(id, ProbeKind.READING, null, List.of(), readings, facts);
		} finally {
			try (Stream<Path> s = Files.walk(work)) {
				s.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
			}
		}
	}

	private static ProbeReport.Reading reading(String name, String unit, List<Rep> reps,
			java.util.function.ToDoubleFunction<Rep> f, boolean higherIsBetter, boolean scored) {
		List<Double> v = new ArrayList<>();
		for (Rep r : reps)
			if (r.consistent() && !Double.isNaN(f.applyAsDouble(r)))
				v.add(f.applyAsDouble(r));
		return new ProbeReport.Reading(name, unit, v, higherIsBetter, scored);
	}

	private static List<Rep> lane(JunoPlayer player, Configuration jfc, Path work, String prompt, int maxTokens,
			int minTokens, int warmups, int reps, Path model) throws Exception {
		for (int i = 0; i < warmups; i++)
			request(player, prompt, maxTokens, minTokens);
		List<Rep> out = new ArrayList<>();
		for (int i = 0; i < reps; i++) {
			Path jfr = work.resolve("rep-" + System.nanoTime() + ".jfr");
			Path metrics = work.resolve("rep-" + System.nanoTime() + ".json");
			GenerationResult res;
			try (Recording rec = new Recording(jfc)) {
				rec.start();
				res = request(player, prompt, maxTokens, minTokens);
				rec.stop();
				rec.dump(jfr);
			}
			String stem = model.getFileName().toString().replaceFirst("\\.[^.]+$", "");
			JfrMetricsCli.main(new String[] { jfr.toString(), metrics.toString(), stem,
					model.getFileName().toString() });
			JsonNode m = Json.MAPPER.readTree(metrics.toFile()).path("models").path(0).path("metrics");
			out.add(new Rep(res.promptTokens(), res.generatedTokens(), res.latency().toNanos() / 1e6,
					m.path(PREFILL_MS).asDouble(0), m.path(DECODE_MS).asDouble(0), m.path(ENCODE_MS).asDouble(0),
					m.path(ALLOC).asDouble(0), m.path(GC_IN_SPAN).asDouble(0)));
		}
		return out;
	}

	private static GenerationResult request(JunoPlayer player, String prompt, int maxTokens, int minTokens) {
		SamplingParams params = SamplingParams.deterministic().withMaxTokens(maxTokens).withMinTokens(minTokens);
		return player.scheduler().submitAndWait(InferenceRequest.of(player.modelId(),
				List.of(ChatMessage.user(prompt)), params, RequestPriority.NORMAL));
	}

	/** The harness's raw prompt: {@code x x x ...}, about one token per word. */
	static String words(int n) {
		StringBuilder sb = new StringBuilder("x");
		for (int i = 1; i < n; i++)
			sb.append(" x");
		return sb.toString();
	}

	/** Juno's measurement settings, packaged from {@code scripts/performance-tests/juno-perf.jfc}. */
	static Configuration jfrConfiguration() throws IOException, ParseException {
		try (InputStream in = ThroughputProbe.class.getResourceAsStream("/juno-perf.jfc")) {
			if (in == null)
				throw new IOException("juno-perf.jfc is not on the class path");
			return Configuration.create(new InputStreamReader(in, StandardCharsets.UTF_8));
		}
	}
}
