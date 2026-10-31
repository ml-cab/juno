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
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import com.fasterxml.jackson.core.type.TypeReference;

/**
 * The GPU residency smoke's local checks on one model: greedy output with the device-resident decode region
 * off, then on, each mode in a JVM of its own. Request 1 and the last request generate {@code nGen} tokens and
 * must be identical; the requests between generate {@code memNGen} and must reproduce request 1's opening.
 * The on run must report the region active (or why it declined). Over the second half of the requests each
 * mode's device memory may grow by at most {@value #LEAK_MIB_PER_REQUEST} MiB per request, and no more with
 * the region on than off. With {@code requireIdentical} the two modes' output must be identical. Then, unless
 * {@code cluster=false}, the launcher starts pipeline and tensor clusters with the region on: the forked nodes
 * must answer and none may be left after the cluster stops; whether their output equals the local on-run's is
 * recorded, not required, as the smoke reports it.
 */
final class GpuResidencyProbe implements KitProbe {

	static final int LEAK_MIB_PER_REQUEST = 8;

	record Run(List<String> texts, List<Long> memMiB, String notice) {
	}

	@Override
	public ProbeReport run(String id, Map<String, String> args) throws Exception {
		String model = args.get("model");
		int requests = KitProbe.intArg(args, "requests", 4);
		int nGen = KitProbe.intArg(args, "nGen", 32);
		int memNGen = KitProbe.intArg(args, "memNGen", 8);
		boolean gpu = Boolean.parseBoolean(args.getOrDefault("gpu", "true"));
		boolean requireIdentical = Boolean.parseBoolean(args.getOrDefault("requireIdentical", "true"));
		String heap = args.getOrDefault("childHeap", "4g");
		Run off = mode("off", model, requests, nGen, memNGen, gpu, heap);
		Run on = mode("on", model, requests, nGen, memNGen, gpu, heap);

		List<ProbeReport.Check> checks = new ArrayList<>();
		checks.add(consistency("region off", off));
		checks.add(consistency("region on", on));
		String n = on.notice();
		checks.add(new ProbeReport.Check("region on: activation or decline reported",
				n.contains("GPU-resident decode region active") || n.contains("requested, but"),
				n.isEmpty() ? "no activation or decline notice" : n));
		int diff = firstDiff(off.texts().get(0), on.texts().get(0));
		checks.add(new ProbeReport.Check("greedy output identical with the region on and off",
				diff < 0 || !requireIdentical, diff < 0 ? "identical over " + nGen + " tokens"
						: "first difference at word " + diff + (requireIdentical ? "" : " (reported, not required)")));
		int windowStart = Math.max(2, (requests + 1) / 2);
		long limit = (long) LEAK_MIB_PER_REQUEST * Math.max(1, requests - windowStart);
		long offGrowth = growth(off.memMiB(), windowStart);
		long onGrowth = growth(on.memMiB(), windowStart);
		boolean noReading = off.memMiB().stream().allMatch(m -> m == 0) && on.memMiB().stream().allMatch(m -> m == 0);
		boolean memOk = offGrowth <= limit && onGrowth <= limit && onGrowth <= offGrowth;
		checks.add(new ProbeReport.Check("device memory does not grow per request", memOk,
				(noReading ? "no device memory reading; " : "") + "growth over requests " + windowStart + ".."
						+ requests + ": on " + onGrowth + " MiB, off " + offGrowth + " MiB (limit " + limit
						+ " MiB); MiB after each request: off " + off.memMiB() + ", on " + on.memMiB()));
		Map<String, Object> facts = new LinkedHashMap<>();
		if (Boolean.parseBoolean(args.getOrDefault("cluster", "true")))
			clusters(Path.of(model), nGen, gpu, heap, on.texts().get(0),
					Path.of(args.getOrDefault("logDir", ".")).resolve(id), checks, facts);
		boolean all = checks.stream().allMatch(ProbeReport.Check::passed);
		facts.put("notice", n);
		facts.put("offMemMiB", off.memMiB());
		facts.put("onMemMiB", on.memMiB());
		facts.put("reply", on.texts().get(0));
		return new ProbeReport(id, ProbeKind.CHECK, all, checks, List.of(), facts);
	}

	private static void clusters(Path model, int nGen, boolean gpu, String heap, String localOn, Path work,
			List<ProbeReport.Check> checks, Map<String, Object> facts) throws Exception {
		for (String type : List.of("pipeline", "tensor")) {
			Instant t0 = Instant.now();
			List<String> args = List.of("--model-path", model.toString(), "--dtype", "FLOAT32", "--byteOrder", "BE",
					"--max-tokens", String.valueOf(nGen), "--temperature", "0", "--top-k", "0", "--top-p", "0",
					gpu ? "--gpu" : "--cpu", "--verbose", "--pType", type, "--gpu-residency", "on");
			String text = "";
			String error = "";
			try (JunoProcess s = JunoProcess.serve(work.resolve("cluster-" + type), heap, args, Duration.ofMinutes(30))) {
				JunoProcess.Answer a = s.post(GrammarProbe.PATH,
						GrammarProbe.body(s.modelId(), ResidencyRunMain.PROMPT, nGen).toString());
				text = GrammarProbe.content(a);
				if (a.status() != 200)
					error = GrammarProbe.describe(a);
			} catch (IOException e) {
				error = e.getMessage();
			}
			Thread.sleep(3000);
			List<ProcessHandle> left = JunoProcess.nodesStartedSince(t0);
			left.forEach(ProcessHandle::destroyForcibly);
			checks.add(new ProbeReport.Check(type + " cluster, region on: the forked nodes answer", !text.isBlank(),
					text.isBlank() ? "no text" + (error.isEmpty() ? "" : ": " + error) : text.length() + " characters"));
			checks.add(new ProbeReport.Check(type + " cluster: no node JVM left", left.isEmpty(),
					left.size() + " node JVMs left"));
			int diff = firstDiff(localOn, text);
			facts.put(type + ".equalsLocal", diff < 0);
			if (diff >= 0)
				facts.put(type + ".firstDifferentWord", diff);
		}
	}

	private static ProbeReport.Check consistency(String label, Run r) {
		List<String> t = r.texts();
		String first = t.get(0);
		boolean ok = !first.isBlank() && t.get(t.size() - 1).equals(first);
		for (int i = 1; i < t.size() - 1; i++)
			ok &= first.startsWith(t.get(i).strip());
		return new ProbeReport.Check(label + ": every request answers, the last equals the first, shorter ones are"
				+ " its opening", ok, ok ? t.size() + " requests" : "replies: " + t);
	}

	/** Index of the first differing whitespace-separated word, or -1. */
	static int firstDiff(String a, String b) {
		String[] x = a.strip().split("\\s+");
		String[] y = b.strip().split("\\s+");
		for (int i = 0; i < Math.max(x.length, y.length); i++)
			if (i >= x.length || i >= y.length || !x[i].equals(y[i]))
				return i;
		return -1;
	}

	/** Memory after the last request minus memory after request {@code windowStart} (1-based). */
	static long growth(List<Long> mem, int windowStart) {
		return mem.isEmpty() ? 0 : mem.get(mem.size() - 1) - mem.get(Math.min(windowStart, mem.size()) - 1);
	}

	private static Run mode(String mode, String model, int requests, int nGen, int memNGen, boolean gpu, String heap)
			throws IOException, InterruptedException {
		List<String> cmd = JvmLauncher.command(heap, false, List.of("-DJUNO_GPU_RESIDENCY=" + mode),
				ResidencyRunMain.class.getName(), List.of(model, String.valueOf(requests), String.valueOf(nGen),
						String.valueOf(memNGen), String.valueOf(gpu)));
		Process p = new ProcessBuilder(cmd).redirectErrorStream(true).start();
		p.getOutputStream().close();
		String out = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
		if (!p.waitFor(30, TimeUnit.MINUTES)) {
			p.destroyForcibly();
			throw new IOException("residency " + mode + " run timed out");
		}
		System.out.println("== region " + mode + " (exit " + p.exitValue() + ")\n" + out);
		for (String line : out.split("\n"))
			if (line.startsWith(ResidencyRunMain.PREFIX)) {
				Map<String, Object> m = Json.MAPPER.readValue(line.substring(ResidencyRunMain.PREFIX.length()),
						new TypeReference<Map<String, Object>>() {
						});
				List<Long> mem = new ArrayList<>();
				for (Object o : (List<?>) m.get("memMiB"))
					mem.add(((Number) o).longValue());
				@SuppressWarnings("unchecked")
				List<String> texts = (List<String>) m.get("texts");
				return new Run(texts, mem, String.valueOf(m.get("notice")));
			}
		throw new IOException("residency " + mode + " run printed no result (exit " + p.exitValue() + ")");
	}
}
