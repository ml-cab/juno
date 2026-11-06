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
import java.math.BigDecimal;
import java.math.MathContext;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;

/** The markdown {@code ingest} drafts: the entry's {@code INDEX.md} and its row for the perf-compare README. */
final class IngestReport {

	private static final int REPLY_CHARS = 60;
	private static final int NOTE_CHARS = 120;
	private static final int BUILD_LOG_LINES = 60;

	private IngestReport() {
	}

	static String index(String name, RunResults run, RunResults ref, String refName, List<ReferenceComparison.Row> rows,
			Coverage coverage, boolean asReference) throws IOException {
		Manifest m = run.manifest();
		StringBuilder b = new StringBuilder();
		b.append("# ").append(asReference ? "Kit reference run" : "Remote kit run").append(": ")
				.append(Ingest.hostLabel(m.host())).append(" (").append(m.finishedUtc()).append(")\n\n");
		b.append("**Draft written by `juno-kit ingest`; review it before publishing.**\n\n");
		b.append("**Host.** ").append(host(m.host())).append("\n\n");
		b.append("**Build.** Juno `").append(m.junoCommit()).append('`')
				.append(m.junoTreeDirty() ? " (modified tree)" : "").append("; kit ").append(m.kitVersion())
				.append(", kit jar sha256 ").append(m.kitJarSha256() == null ? "none" : "`" + m.kitJarSha256() + "`")
				.append(".\n\n");
		if (run.buildFailed()) {
			b.append("**The build failed** before any probe ran. The end of `build.log`:\n\n```\n")
					.append(tail(run.dir().resolve("build.log"))).append("```\n");
			return b.toString();
		}
		b.append("**Run.** Suite `").append(m.suite()).append("`, ").append(m.startedUtc()).append(" to ")
				.append(m.finishedUtc()).append("; ").append(run.checkpoint().done().size()).append(" probes: ")
				.append(statusCounts(run.checkpoint().done())).append(".\n\n");

		b.append("## Probes\n\n| Probe | Status | Time | Attempts | Other CPU | Note |\n|---|---|---|---|---|---|\n");
		for (ProbeOutcome o : run.checkpoint().done())
			b.append("| `").append(o.id()).append("` | ").append(o.status()).append(" | ")
					.append(Progress.clock(o.wallMillis() / 1000.0)).append(" | ").append(o.attempts()).append(" | ")
					.append(Math.round(o.otherCpuShare() * 100)).append("% | ").append(cell(o.note(), NOTE_CHARS))
					.append(" |\n");

		b.append("\n## Readings\n\nEach reading's repetitions on this host. They are not compared with the reference"
				+ " host.\n\n| Probe | Reading | Unit | Median | Min | Max | n | Status |\n"
				+ "|---|---|---|---|---|---|---|---|\n");
		Map<String, String> status = new LinkedHashMap<>();
		for (ProbeOutcome o : run.checkpoint().done())
			status.put(o.id(), o.status());
		for (Map.Entry<String, ProbeReport> e : run.reports().entrySet())
			for (ProbeReport.Reading r : e.getValue().readings())
				b.append("| `").append(e.getKey()).append("` | ").append(cell(r.name(), NOTE_CHARS)).append(" | ")
						.append(r.unit()).append(" | ").append(number(r.median())).append(" | ")
						.append(number(r.min())).append(" | ").append(number(r.max())).append(" | ")
						.append(r.values().size()).append(" | ").append(status.get(e.getKey())).append(" |\n");

		b.append("\n## Correctness against the reference run\n\n");
		if (asReference)
			b.append("This is a reference run: later ingests at Juno `").append(shortCommit(m))
					.append("` compare with it.\n");
		else if (ref == null)
			b.append("No reference run at commit ").append(shortCommit(m)).append(": correctness is not compared."
					+ " Ingest the owner's run at this commit with `--reference-run`, then ingest this zip again.\n");
		else {
			long listed = rows.stream().filter(ReferenceComparison.Row::listed).count();
			b.append("Reference: `").append(refName).append("` (Juno `").append(shortCommit(ref.manifest()))
					.append("`). ").append(rows.size()).append(" items compared: ").append(rows.size() - listed)
					.append(" equal or within the bound").append(listed == 0 ? ".\n" : "; listed below.\n");
			if (listed > 0) {
				b.append("\n| Probe | Item | Here | Reference | Verdict |\n|---|---|---|---|---|\n");
				for (ReferenceComparison.Row r : rows)
					if (r.listed())
						b.append("| `").append(r.probe()).append("` | ").append(cell(r.item(), NOTE_CHARS))
								.append(" | ").append(cell(r.remote(), REPLY_CHARS)).append(" | ")
								.append(cell(r.reference(), REPLY_CHARS)).append(" | ").append(r.verdict())
								.append(" |\n");
			}
		}

		b.append("\n## Coverage on this host\n\n").append(gpuBackend(m.host()))
				.append("\n\n| Item | Title | Probes run here |\n|---|---|---|\n");
		for (Coverage.Entry e : coverage.entries()) {
			List<ProbeOutcome> ran = new ArrayList<>();
			for (String p : e.probes())
				for (String id : Coverage.matching(p, status.keySet()))
					run.checkpoint().done().stream().filter(o -> o.id().equals(id) && !ran.contains(o))
							.forEach(ran::add);
			b.append("| ").append(e.key()).append(" | ").append(cell(e.title(), NOTE_CHARS)).append(" | ")
					.append(ran.isEmpty() ? "none" : statusCounts(ran)).append(" |\n");
		}

		b.append("\n## Files\n\n- `manifest.json`, `checkpoint.json`, `host.json`, `SUMMARY.txt`: as returned.\n"
				+ "- `probes/`: each probe's report.\n");
		return b.toString();
	}

	/**
	 * The row for the perf-compare README's table, in its four columns: the entry, what ran on which host, what the
	 * entry records, its INDEX. A kit run is never clock-pinned, since pinning needs admin rights.
	 */
	static String readmeRow(String name, RunResults run, RunResults ref, List<ReferenceComparison.Row> rows,
			boolean asReference) {
		Manifest m = run.manifest();
		StringBuilder b = new StringBuilder("| [`").append(name).append("`](").append(name).append("/) | ");
		b.append(asReference ? "Hardware validation kit reference run: " : "Hardware validation kit, remote run: ")
				.append(host(m.host())).append("; Juno `").append(shortCommit(m)).append('`')
				.append(m.junoTreeDirty() ? " (modified tree)" : "");
		String index = " | [INDEX](" + name + "/INDEX.md) |";
		if (run.buildFailed())
			return b.append(", the build failed | the end of the build log").append(index).toString();
		String jar = m.kitJarSha256();
		if (jar != null)
			b.append(", kit jar `").append(jar, 0, Math.min(16, jar.length())).append('`');
		Map<String, Long> counts = run.checkpoint().done().stream()
				.collect(Collectors.groupingBy(ProbeOutcome::status, TreeMap::new, Collectors.counting()));
		long good = counts.getOrDefault(ProbeOutcome.PASS, 0L) + counts.getOrDefault(ProbeOutcome.DONE, 0L);
		b.append(", suite `").append(m.suite()).append("`, unpinned: ").append(good).append(" PASS or DONE");
		for (String s : List.of(ProbeOutcome.FAIL, ProbeOutcome.NOISY, ProbeOutcome.ERROR, ProbeOutcome.TIMEOUT))
			if (counts.containsKey(s))
				b.append(", ").append(counts.get(s)).append(' ').append(s);
		if (ref != null)
			b.append("; against the reference run, ").append(rows.stream().filter(ReferenceComparison.Row::listed).count())
					.append(" of ").append(rows.size()).append(" items differ");
		b.append(" | probe status and time, readings per probe, coverage");
		if (ref != null)
			b.append(", correctness against the reference run");
		return b.append(index).toString();
	}

	/** Six significant digits, trailing zeros dropped; {@code n/a} for a missing value. */
	static String number(double v) {
		if (Double.isNaN(v) || Double.isInfinite(v))
			return "n/a";
		return new BigDecimal(v).round(new MathContext(6)).stripTrailingZeros().toPlainString();
	}

	private static String host(Map<String, Object> h) {
		StringBuilder b = new StringBuilder();
		b.append(h.getOrDefault("os", "unknown OS"));
		if (h.get("osVersion") != null)
			b.append(' ').append(h.get("osVersion"));
		b.append(' ').append(h.getOrDefault("arch", ""));
		if (h.get("cpuModel") != null)
			b.append("; ").append(h.get("cpuModel"));
		if (h.get("cores") != null)
			b.append(", ").append(h.get("cores")).append(" cores");
		if (h.get("threads") != null)
			b.append(", ").append(h.get("threads")).append(" threads");
		if (h.get("ramBytes") instanceof Number ram)
			b.append(", ").append(ram.longValue() >> 30).append(" GiB RAM");
		if (h.get("intVectorBits") != null)
			b.append("; JVM vectors: integer ").append(h.get("intVectorBits")).append("-bit, float ")
					.append(h.get("floatVectorBits")).append("-bit");
		if (h.get("gpus") instanceof List<?> gpus)
			b.append("; GPU: ").append(gpus.isEmpty() ? "none"
					: gpus.stream().map(g -> g instanceof Map<?, ?> gm ? String.valueOf(gm.get("name")) : "?")
							.collect(Collectors.joining(", ")));
		return cell(b.toString(), 400);
	}

	/**
	 * Which backend the GPU probes ran on, from Juno's own backend probe in the host map. A probe that needs no
	 * particular backend runs on the one Juno selects, CUDA before ROCm, whichever GPU row lists it.
	 */
	private static String gpuBackend(Map<String, Object> h) {
		if (!h.containsKey("cuda") && !h.containsKey("rocm"))
			return "**GPU backend:** not recorded in the manifest.";
		boolean cuda = Boolean.TRUE.equals(h.get("cuda"));
		boolean rocm = Boolean.TRUE.equals(h.get("rocm"));
		if (cuda && rocm)
			return "**GPU backend:** CUDA and ROCm both load. A GPU probe that needs no particular backend ran on CUDA,"
					+ " which Juno selects first, so the ROCm row counts CUDA runs except for the probes that need ROCm.";
		if (cuda)
			return "**GPU backend:** CUDA. Every GPU probe ran on it, so the ROCm row counts CUDA runs and says nothing"
					+ " about ROCm here.";
		if (rocm)
			return "**GPU backend:** ROCm. Every GPU probe ran on it, so the CUDA row counts ROCm runs and says nothing"
					+ " about CUDA here.";
		return "**GPU backend:** none. Juno found no usable GPU, so no GPU probe ran.";
	}

	private static String statusCounts(List<ProbeOutcome> outcomes) {
		Map<String, Long> counts = outcomes.stream()
				.collect(Collectors.groupingBy(ProbeOutcome::status, TreeMap::new, Collectors.counting()));
		return counts.entrySet().stream().map(e -> e.getValue() + " " + e.getKey()).collect(Collectors.joining(", "));
	}

	private static String shortCommit(Manifest m) {
		String c = m.junoCommit();
		return c == null ? "unknown" : c.substring(0, Math.min(8, c.length()));
	}

	/** One table cell: pipes escaped, line breaks as spaces, cut to {@code max} characters. */
	private static String cell(String s, int max) {
		if (s == null)
			return "";
		String t = s.replace("|", "\\|").replaceAll("\\s+", " ").strip();
		return t.length() <= max ? t : t.substring(0, max) + "...";
	}

	private static String tail(Path log) throws IOException {
		if (!Files.isRegularFile(log))
			return "(no build.log in the zip)\n";
		List<String> lines = Files.readAllLines(log);
		StringBuilder b = new StringBuilder();
		for (String l : lines.subList(Math.max(0, lines.size() - BUILD_LOG_LINES), lines.size()))
			b.append(l).append('\n');
		return b.toString();
	}
}
