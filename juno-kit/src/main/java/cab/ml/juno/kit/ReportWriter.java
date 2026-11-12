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
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;

/** Writes {@code SUMMARY.txt} and {@code report.html} for a run directory from its probe reports. */
final class ReportWriter {

	private ReportWriter() {
	}

	static String summary(Manifest m, HostInfo host, Checkpoint cp, List<Selection.Skip> skipped, Path probesDir)
			throws IOException {
		StringBuilder sb = new StringBuilder();
		sb.append("Juno hardware validation kit ").append(m.kitVersion()).append(", Juno ").append(m.junoCommit())
				.append(m.junoTreeDirty() ? " (modified tree)" : "").append('\n');
		sb.append("Suite ").append(m.suite()).append(", started ").append(m.startedUtc()).append(", finished ")
				.append(m.finishedUtc()).append('\n');
		sb.append(String.format(Locale.ROOT, "Host: %s %s %s; %s, %d cores, %d threads, %d GiB RAM%n", host.os(),
				host.osVersion(), host.arch(), host.cpuModel(), host.cores(), host.threads(), host.ramBytes() >> 30));
		sb.append(String.format(Locale.ROOT, "JVM vectors: integer %d-bit, float %d-bit; UseAVX %s; Juno CUDA %s, ROCm %s%n",
				host.intVectorBits(), host.floatVectorBits(), host.useAvx(), host.cuda() ? "yes" : "no",
				host.rocm() ? "yes" : "no"));
		for (HostInfo.Gpu g : host.gpus())
			sb.append(String.format(Locale.ROOT, "GPU: %s %s, %d MiB, driver %s%n", g.vendor(), g.name(),
					g.memoryBytes() >> 20, g.driver()));
		sb.append(JfrClockCheck.describe(host.clockSource(), JfrClockCheck.ratioOf(host.backend()))).append('\n');
		sb.append('\n');
		for (String id : cp.selected()) {
			ProbeOutcome o = cp.done().stream().filter(d -> d.id().equals(id)).findFirst().orElse(null);
			sb.append(String.format(Locale.ROOT, "%-30s %-8s %s%n", id, o == null ? "NOT RUN" : o.status(),
					o == null ? "" : Progress.clock(o.wallMillis() / 1000.0)
							+ (o.attempts() > 1 ? ", " + o.attempts() + " attempts" : "")
							+ (o.ownGpuPeakMiB() > 0 ? ", GPU " + o.ownGpuPeakMiB() + " MiB" : "")
							+ (o.gpuBusy() ? ", GPU busy" : "")
							+ (o.note().isEmpty() ? "" : ", " + o.note())));
			Path f = probesDir.resolve(id + ".json");
			if (!Files.isRegularFile(f))
				continue;
			ProbeReport r = Json.read(f, ProbeReport.class);
			for (ProbeReport.Reading rd : r.readings())
				sb.append(String.format(Locale.ROOT, "    %-52s %s%n", rd.name(), numbers(rd)));
			for (ProbeReport.Check c : r.checks())
				sb.append(String.format(Locale.ROOT, "    %-4s %s%s%n", c.passed() ? "ok" : "FAIL", c.name(),
						c.detail() == null || c.detail().isEmpty() || (c.passed() && c.detail().length() > 120) ? ""
								: ": " + c.detail()));
		}
		if (!skipped.isEmpty()) {
			sb.append("\nNot run on this machine:\n");
			for (Selection.Skip s : skipped)
				sb.append(String.format(Locale.ROOT, "  %-30s %s%n", s.probe().id(), s.reason()));
		}
		return sb.toString();
	}

	static String numbers(ProbeReport.Reading r) {
		if (r.values().isEmpty())
			return "no valid repetition (" + r.unit() + ")";
		return String.format(Locale.ROOT, "median %s, min %s, max %s %s (n=%d%s)", fmt(r.median()), fmt(r.min()),
				fmt(r.max()), r.unit(), r.values().size(),
				r.scored() ? String.format(Locale.ROOT, ", spread %.1f%%", NoiseRule.spread(r.values()) * 100) : "");
	}

	private static String fmt(double v) {
		double a = Math.abs(v);
		return a >= 1000 ? String.format(Locale.ROOT, "%.0f", v)
				: a >= 10 ? String.format(Locale.ROOT, "%.2f", v) : String.format(Locale.ROOT, "%.4f", v);
	}

	/** The summary as an HTML page: one escaped preformatted block, readable in any browser, nothing external. */
	static String html(String summary) {
		String body = summary.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;");
		return "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"><title>Juno kit results</title>"
				+ "<style>body{font:14px/1.45 ui-monospace,monospace;margin:24px;background:#fff;color:#111}"
				+ "pre{white-space:pre-wrap}</style></head><body><pre>" + body + "</pre></body></html>\n";
	}

	static void write(Path runDir, Manifest m, HostInfo host, Checkpoint cp, List<Selection.Skip> skipped)
			throws IOException {
		String s = summary(m, host, cp, skipped, runDir.resolve(RunSession.PROBES_DIR));
		Files.writeString(runDir.resolve("SUMMARY.txt"), s);
		Files.writeString(runDir.resolve("report.html"), html(s));
	}
}
