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

/**
 * Image to text on moondream2 through the launcher's vision route, as the vision script and the CPU hot-path
 * smoke's vision case run it: the standard test image with the script's prompt (32 tokens; latency, decode t/s off
 * the server's JFR summary, and the script's prompt-token range for image and prompt), then with the smoke's
 * one-sentence prompt (24 tokens, within 600 s). On the CPU this runs at two {@code --threads} values, whose
 * captions must be identical; on the GPU it runs once, and gives readings only when matmuls ran on the device.
 * Prefill is single-token, the vision script's default.
 */
final class VisionProbe implements KitProbe {

	static final String PATH = "/v1/vision/chat";
	static final String PROMPT = "What is in this image?";
	static final String SMOKE_PROMPT = "Describe the image in one sentence.";
	private static final Duration START = Duration.ofMinutes(10);
	private static final long SMOKE_LIMIT_MS = 600_000;

	@Override
	public ProbeReport run(String id, Map<String, String> args) throws Exception {
		Path model = Path.of(args.get("model"));
		String heap = args.getOrDefault("serverHeap", "6g");
		Path work = Path.of(args.getOrDefault("logDir", ".")).resolve(id);
		boolean gpu = Boolean.parseBoolean(args.getOrDefault("gpu", "false"));
		Path image = Path.of(args.getOrDefault("image", Path.of(System.getProperty("juno.kit.root", "."))
				.resolve("scripts/performance-tests/fixtures/vision-bench.jpg").toString()));
		List<String> runs = gpu ? List.of("GPU")
				: List.of(args.getOrDefault("threads", String.join(",", CpuHotPathProbe.defaultThreads(Runtime
						.getRuntime().availableProcessors()))).split(","));
		List<ProbeReport.Check> checks = new ArrayList<>();
		List<ProbeReport.Reading> readings = new ArrayList<>();
		Map<String, Object> facts = new LinkedHashMap<>();
		List<String> captions = new ArrayList<>();
		List<String> sentences = new ArrayList<>();
		for (String run : runs) {
			String t = run.trim();
			String label = gpu ? "GPU: " : "threads " + t + ": ";
			List<String> a = JunoProcess.localArgs(model, 32);
			a.addAll(List.of(gpu ? "--gpu" : "--cpu", "--prefill", "single", "--jfr", "30m"));
			if (!gpu)
				a.addAll(List.of("--threads", t));
			JunoProcess s = JunoProcess.serve(work.resolve(gpu ? "gpu" : "threads" + t), heap, a, START);
			JunoProcess.Answer bench;
			JunoProcess.Answer smoke;
			double benchMs;
			double smokeMs;
			try {
				String m = s.modelId();
				long t0 = System.nanoTime();
				bench = s.postImage(PATH, image, GrammarProbe.body(m, PROMPT, 32).toString());
				benchMs = (System.nanoTime() - t0) / 1e6;
				t0 = System.nanoTime();
				smoke = s.postImage(PATH, image, GrammarProbe.body(m, SMOKE_PROMPT, 24).toString());
				smokeMs = (System.nanoTime() - t0) / 1e6;
			} finally {
				s.close();
			}
			String benchText = GrammarProbe.content(bench);
			String smokeText = GrammarProbe.content(smoke);
			int pt = usage(bench, "prompt_tokens");
			checks.add(new ProbeReport.Check(label + "a caption for the benchmark prompt",
					bench.status() == 200 && !benchText.isBlank(), GrammarProbe.describe(bench)));
			checks.add(new ProbeReport.Check(label + "prompt tokens between 700 and 800", pt >= 700 && pt <= 800,
					pt + " prompt tokens"));
			checks.add(new ProbeReport.Check(label + "a one-sentence caption within 600 s",
					smoke.status() == 200 && !smokeText.isBlank() && smokeMs <= SMOKE_LIMIT_MS,
					GrammarProbe.describe(smoke) + String.format(java.util.Locale.ROOT, " (%.0f s)", smokeMs / 1000)));
			JsonNode jfr = s.metrics();
			readings.addAll(readings(label, gpu, bench.status() == 200 ? benchMs : Double.NaN, jfr));
			if (gpu)
				facts.put("gpu.deviceMatVecs", deviceMatVecs(jfr));
			facts.put((gpu ? "gpu" : "threads" + t) + ".caption", benchText);
			facts.put((gpu ? "gpu" : "threads" + t) + ".sentence", smokeText);
			captions.add(benchText);
			sentences.add(smokeText);
		}
		if (captions.size() > 1) {
			boolean same = captions.stream().distinct().count() == 1 && sentences.stream().distinct().count() == 1
					&& !captions.get(0).isBlank() && !sentences.get(0).isBlank();
			checks.add(new ProbeReport.Check("captions identical at " + String.join(" and ", runs.stream()
					.map(String::trim).toList()) + " threads", same, same ? "identical"
							: "captions " + captions + "; sentences " + sentences));
		}
		facts.put("image", image.getFileName().toString());
		facts.put("backend", gpu ? "gpu" : "cpu");
		return new ProbeReport(id, ProbeKind.CHECK, checks.stream().allMatch(ProbeReport.Check::passed), checks,
				readings, facts);
	}

	/**
	 * One server's readings: the benchmark request's latency, and the decode rate as the mean single-token forward
	 * pass off the server's JFR summary. With single-token prefill the prompt runs through that same pass, so prompt
	 * and generated tokens both count; generated tokens over the recording's time would also span the second
	 * request's image and prompt. A GPU run whose matmuls all ran on the host (an architecture without a device
	 * path) gives none: its figures would be the CPU's under the GPU's name.
	 */
	static List<ProbeReport.Reading> readings(String label, boolean gpu, double latencyMs, JsonNode jfr) {
		if (gpu && deviceMatVecs(jfr) == 0)
			return List.of();
		double passes = jfr.path("juno.ForwardPass.decode.count").asDouble(0);
		double ms = jfr.path("juno.ForwardPass.decode.total_ms").asDouble(0);
		return List.of(new ProbeReport.Reading(label + "latency", "ms", KitProbe.one(latencyMs), false, false),
				new ProbeReport.Reading(label + "decode t/s", "t/s",
						KitProbe.one(passes > 0 && ms > 0 ? passes / (ms / 1000) : Double.NaN), true, false));
	}

	/** Matmuls the recording saw on any backend other than the host's. */
	static long deviceMatVecs(JsonNode jfr) {
		long n = 0;
		for (Map.Entry<String, JsonNode> e : jfr.properties()) {
			String k = e.getKey();
			if (k.startsWith("juno.MatVec.backend.") && k.endsWith(".count")
					&& !k.equals("juno.MatVec.backend.cpu.count"))
				n += e.getValue().asLong(0);
		}
		return n;
	}

	private static int usage(JunoProcess.Answer a, String field) {
		try {
			return a.status() == 200 ? a.json().path("usage").path(field).asInt(0) : 0;
		} catch (Exception e) {
			return 0;
		}
	}
}
