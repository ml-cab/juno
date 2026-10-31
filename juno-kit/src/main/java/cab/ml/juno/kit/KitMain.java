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


import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Duration;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

import com.fasterxml.jackson.core.type.TypeReference;

import cab.ml.juno.player.HfGgufFetcher;

/**
 * The kit itself, started by {@link KitBootstrap} once the build is done: detects the machine, lets the
 * tester choose, calibrates and estimates, warns, runs the probes with a checkpoint after each, and packs
 * the scrubbed results into one zip. Nothing is uploaded.
 */
public final class KitMain {

	private KitMain() {
	}

	public static void main(String[] args) {
		int rc;
		try {
			rc = run(List.of(args), System.out, System.in);
		} catch (IllegalArgumentException e) {
			System.err.println(e.getMessage());
			rc = 2;
		} catch (Exception e) {
			e.printStackTrace();
			rc = 1;
		}
		System.exit(rc);
	}

	static int run(List<String> args, PrintStream console, InputStream stdin) throws Exception {
		Path root = Path.of(System.getProperty("juno.kit.root", ".")).toAbsolutePath().normalize();
		KitOptions o = KitOptions.parse(args, root);
		if (o.help()) {
			console.print(KitOptions.USAGE);
			return 0;
		}
		KitBootstrap.Privacy privacy = KitBootstrap.Privacy.fromSystem();
		return switch (o.command()) {
		case "pack" -> pack(o, root, console, privacy);
		case "resume" -> resume(o, root, console, privacy);
		case "ingest" -> ingest(o, console);
		default -> fresh(o, root, console, stdin, privacy);
		};
	}

	// ── a new run ────────────────────────────────────────────────────────────

	private static int fresh(KitOptions o, Path root, PrintStream console, InputStream stdin,
			KitBootstrap.Privacy privacy) throws Exception {
		String[] commit = KitBootstrap.junoCommit(root);
		console.println("juno-kit " + KitBootstrap.kitVersion(root) + ", Juno " + commit[0]
				+ (Boolean.parseBoolean(commit[1]) ? " (modified tree)" : ""));
		console.println("Detecting this machine...");
		HostInfo host = HostDetector.detect(existingAncestor(o.resultsRoot()));
		printHost(console, host);

		ModelStore store = new ModelStore(o.modelsDir(), o.cacheDir(), new HfGgufFetcher());
		List<Probe> catalog = ProbeCatalog.all();
		Map<String, Path> present = new LinkedHashMap<>();
		Set<String> missing = new LinkedHashSet<>();
		for (Probe p : catalog)
			for (String id : p.models()) {
				Path f = store.present(CatalogModel.byId(id));
				if (f != null)
					present.put(id, f);
				else
					missing.add(id);
			}
		boolean hub = !missing.isEmpty() && reachable("https://huggingface.co/");
		Set<String> available = new LinkedHashSet<>(present.keySet());
		if (hub)
			available.addAll(missing);

		if (o.list()) {
			printList(console, catalog, host, available, present);
			return 0;
		}
		BufferedReader in = new BufferedReader(new InputStreamReader(stdin, StandardCharsets.UTF_8));
		boolean ask = System.console() != null && !o.yes() && !o.plan();
		String suite = o.suite() != null ? o.suite() : ask ? menu(console, in, catalog, host, available) : "quick";
		List<String> probeIds = o.probes();
		if (suite.equals("custom") && probeIds.isEmpty())
			probeIds = askProbes(console, in, catalog);

		Selection.Result planned = Selection.select(catalog, suite, probeIds, host, available, o.budget(), 1.0);
		if (o.plan()) {
			printPlan(console, suite, planned, present, o);
			return 0;
		}
		if (planned.selected().isEmpty()) {
			console.println("Nothing to run on this machine for suite " + suite + ".");
			printSkipped(console, planned.skipped());
			return 0;
		}

		String utc = nowUtc();
		Path runDir = o.resultsRoot().resolve(utc + "-" + suite);
		Files.createDirectories(runDir.resolve(RunSession.PROBES_DIR));
		try (PrintStream out = tee(console, runDir.resolve("kit.log"))) {
			Json.write(runDir.resolve("host.json"), host);
			out.println("Calibrating (about a minute). Leave the computer idle.");
			ProbeOutcome calibration = new ProbeRunner(Map.of()).execute(ProbeCatalog.byId("calibration"), runDir);
			double scale = scale(runDir);
			Selection.Result sel = Selection.select(catalog, suite, probeIds, host, available, o.budget(), scale);
			long downloadBytes = downloadBytes(sel.selected(), present);
			long free = Files.getFileStore(existingAncestor(o.cacheDir())).getUsableSpace();
			if (downloadBytes > free) {
				out.printf(Locale.ROOT, "The downloads need %d MiB in %s and %d MiB are free. Free some space or pass"
						+ " --cache-dir.%n", downloadBytes >> 20, o.cacheDir(), free >> 20);
				return 2;
			}
			double rate = downloadBytes > 0 ? measureRate(sel.selected(), present) : 0;
			List<Probe> left = new ArrayList<>(sel.selected());
			left.removeIf(p -> p.id().equals("calibration"));
			Estimator.Range est = Estimator.estimate(left, scale, downloadBytes, rate);
			out.printf(Locale.ROOT, "Calibration: %s (this machine takes %.2fx the reference host's time).%n",
					calibration.status(), scale);
			out.println("Will run:");
			for (Probe p : sel.selected())
				out.printf(Locale.ROOT, "  %-30s %s%n", p.id(), p.title());
			printSkipped(out, sel.skipped());
			warnings(out, host, sel.selected(), downloadBytes);
			out.printf(Locale.ROOT, "Estimated time: %s to %s.%n", Progress.minutes(est.low().toSeconds()),
					Progress.minutes(est.high().toSeconds()));
			if (!o.yes() && !confirm(out, in)) {
				out.println("Nothing run. The calibration reading is in " + runDir);
				return 0;
			}
			Json.write(runDir.resolve("skipped.json"), skipsToMaps(sel.skipped()));
			Json.write(runDir.resolve("estimate.json"), Map.of("lowSeconds", est.low().toSeconds(), "pointSeconds",
					est.point().toSeconds(), "highSeconds", est.high().toSeconds(), "scale", scale));
			Checkpoint cp = Checkpoint.start(suite, sel.selected().stream().map(Probe::id).toList(), o.raw(), utc)
					.with(calibration);
			cp.save(runDir);
			return execute(runDir, sel.selected(), cp, host, store, scale, out, privacy, root, sel.skipped());
		}
	}

	// ── ingest ───────────────────────────────────────────────────────────────

	private static int ingest(KitOptions o, PrintStream console) throws IOException {
		Path scratch = Files.createTempDirectory("juno-kit-ingest");
		try {
			Path draft = Ingest.run(o.runDir(), o.reference(), o.resultsRoot(), nowUtc(), scratch, o.referenceRun());
			console.println("Draft written: " + draft);
			console.println("Review its INDEX.md, then add README-row.md to the perf-compare README.");
			return 0;
		} finally {
			try (Stream<Path> s = Files.walk(scratch)) {
				for (Path p : s.sorted(Comparator.reverseOrder()).toList())
					Files.deleteIfExists(p);
			}
		}
	}

	// ── resume and pack ──────────────────────────────────────────────────────

	private static int resume(KitOptions o, Path root, PrintStream console, KitBootstrap.Privacy privacy)
			throws Exception {
		Path runDir = o.runDir();
		Checkpoint cp = Checkpoint.load(runDir);
		HostInfo host = Json.read(runDir.resolve("host.json"), HostInfo.class);
		KitOptions started = KitOptions.parse(cp.args(), root);
		List<Probe> selected = cp.selected().stream().map(ProbeCatalog::byId).toList();
		ModelStore store = new ModelStore(started.modelsDir(), started.cacheDir(), new HfGgufFetcher());
		try (PrintStream out = tee(console, runDir.resolve("kit.log"))) {
			out.println("Resuming " + runDir.getFileName() + ": " + cp.remaining().size() + " of " + selected.size()
					+ " probes left.");
			return execute(runDir, selected, cp, host, store, scale(runDir), out, privacy, root, readSkipped(runDir));
		}
	}

	private static int pack(KitOptions o, Path root, PrintStream console, KitBootstrap.Privacy privacy)
			throws IOException {
		Path runDir = o.runDir();
		return finish(runDir, Checkpoint.load(runDir), Json.read(runDir.resolve("host.json"), HostInfo.class), console,
				privacy, root, readSkipped(runDir));
	}

	// ── shared ───────────────────────────────────────────────────────────────

	private static int execute(Path runDir, List<Probe> selected, Checkpoint cp, HostInfo host, ModelStore store,
			double scale, PrintStream out, KitBootstrap.Privacy privacy, Path root, List<Selection.Skip> skipped)
			throws Exception {
		try (KeepAwake awake = KeepAwake.start(out)) {
			Map<String, Path> modelPaths = new LinkedHashMap<>();
			Set<String> needed = new LinkedHashSet<>();
			for (Probe p : selected)
				if (cp.remaining().contains(p.id()))
					needed.addAll(p.models());
			if (!needed.isEmpty())
				out.println("Checking and downloading models. You may use the computer meanwhile.");
			for (String id : needed) {
				try {
					modelPaths.put(id, store.ensure(CatalogModel.byId(id), out));
				} catch (IOException e) {
					out.println("  " + id + ": " + e.getMessage());
					for (Probe p : selected)
						if (p.models().contains(id) && cp.remaining().contains(p.id()))
							cp = cp.with(new ProbeOutcome(p.id(), ProbeOutcome.ERROR, 0, 0,
									"model " + id + ": " + e.getMessage(), 0, 0, false));
					cp.save(runDir);
				}
			}
			out.println("Running " + cp.remaining().size() + " probes.");
			List<String> remaining = cp.remaining();
			List<Probe> toRun = selected.stream().filter(p -> remaining.contains(p.id())).toList();
			cp = new RunSession(runDir, selected, cp, new ProbeRunner(modelPaths),
					new Progress(out, toRun, scale, System.console() != null)).run();
		}
		return finish(runDir, cp, host, out, privacy, root, skipped);
	}

	private static int finish(Path runDir, Checkpoint cp, HostInfo host, PrintStream out,
			KitBootstrap.Privacy privacy, Path root, List<Selection.Skip> skipped) throws IOException {
		String[] commit = KitBootstrap.junoCommit(root);
		@SuppressWarnings("unchecked")
		Map<String, Object> hostMap = Json.MAPPER.convertValue(host, Map.class);
		Manifest m = new Manifest(Manifest.SCHEMA_VERSION, "run", KitBootstrap.kitVersion(root), commit[0],
				Boolean.parseBoolean(commit[1]), kitJarSha256(), cp.suite(), cp.selected(), cp.startedUtc(), nowUtc(),
				hostMap);
		m.write(runDir.resolve(Manifest.FILE));
		String buildLog = System.getProperty("juno.kit.buildLog");
		if (buildLog != null && Files.isRegularFile(Path.of(buildLog)) && !Files.exists(runDir.resolve("build.log")))
			Files.copy(Path.of(buildLog), runDir.resolve("build.log"));
		ReportWriter.write(runDir, m, host, cp, skipped);
		Path zip = runDir.toAbsolutePath().getParent().resolve(KitBootstrap.zipName(cp.startedUtc()));
		ResultsPacker.pack(runDir, zip, privacy);
		out.println();
		out.print(Files.readString(runDir.resolve("SUMMARY.txt")));
		out.println();
		out.println("Results: " + zip);
		out.println("Please send this file back. The kit uploaded nothing.");
		return 0;
	}

	private static double scale(Path runDir) throws IOException {
		Path f = runDir.resolve(RunSession.PROBES_DIR).resolve("calibration.json");
		if (!Files.isRegularFile(f))
			return 1.0;
		for (ProbeReport.Reading r : Json.read(f, ProbeReport.class).readings())
			if (r.name().equals(CalibrationProbe.READING) && !r.values().isEmpty())
				return Estimator.scale(r.median(), ProbeCatalog.REFERENCE_CALIBRATION_MS);
		return 1.0;
	}

	private static long downloadBytes(List<Probe> probes, Map<String, Path> present) {
		Set<String> ids = new LinkedHashSet<>();
		for (Probe p : probes)
			for (String id : p.models())
				if (!present.containsKey(id))
					ids.add(id);
		long b = 0;
		for (String id : ids)
			b += CatalogModel.byId(id).bytes();
		return b;
	}

	/** Bytes per second over the first 8 MiB of the first model to download; 0 when it cannot be measured. */
	private static double measureRate(List<Probe> probes, Map<String, Path> present) {
		for (Probe p : probes)
			for (String id : p.models()) {
				if (present.containsKey(id))
					continue;
				CatalogModel m = CatalogModel.byId(id);
				try {
					HttpClient http = HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL)
							.connectTimeout(Duration.ofSeconds(10)).build();
					long t0 = System.nanoTime();
					HttpResponse<byte[]> r = http.send(HttpRequest
							.newBuilder(URI.create("https://huggingface.co/" + m.repo() + "/resolve/" + m.revision() + "/"
									+ m.file()))
							.header("Range", "bytes=0-8388607").timeout(Duration.ofSeconds(60)).build(),
							HttpResponse.BodyHandlers.ofByteArray());
					double s = (System.nanoTime() - t0) / 1e9;
					return s > 0 ? r.body().length / s : 0;
				} catch (IOException e) {
					return 0;
				} catch (InterruptedException e) {
					Thread.currentThread().interrupt();
					return 0;
				}
			}
		return 0;
	}

	private static boolean reachable(String url) {
		try {
			HttpClient http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
			int code = http.send(HttpRequest.newBuilder(URI.create(url)).timeout(Duration.ofSeconds(10))
					.method("HEAD", HttpRequest.BodyPublishers.noBody()).build(), HttpResponse.BodyHandlers.discarding())
					.statusCode();
			return code > 0 && code < 500;
		} catch (IOException e) {
			return false;
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
			return false;
		}
	}

	private static String kitJarSha256() {
		try {
			Path p = Path.of(KitMain.class.getProtectionDomain().getCodeSource().getLocation().toURI());
			return Files.isRegularFile(p) ? ModelStore.sha256(p) : "not-a-jar";
		} catch (Exception e) {
			return "unknown";
		}
	}

	static String nowUtc() {
		return ZonedDateTime.now(ZoneOffset.UTC).format(KitBootstrap.UTC_STAMP);
	}

	private static Path existingAncestor(Path p) {
		Path c = p.toAbsolutePath();
		while (c != null && !Files.exists(c))
			c = c.getParent();
		return c != null ? c : Path.of(".").toAbsolutePath();
	}

	private static List<Map<String, String>> skipsToMaps(List<Selection.Skip> skips) {
		List<Map<String, String>> out = new ArrayList<>();
		for (Selection.Skip s : skips)
			out.add(Map.of("id", s.probe().id(), "reason", s.reason()));
		return out;
	}

	private static List<Selection.Skip> readSkipped(Path runDir) throws IOException {
		Path f = runDir.resolve("skipped.json");
		if (!Files.isRegularFile(f))
			return List.of();
		List<Selection.Skip> out = new ArrayList<>();
		for (Map<String, String> m : Json.MAPPER.readValue(f.toFile(), new TypeReference<List<Map<String, String>>>() {
		}))
			out.add(new Selection.Skip(ProbeCatalog.byId(m.get("id")), m.get("reason")));
		return out;
	}

	// ── console ──────────────────────────────────────────────────────────────

	private static void printHost(PrintStream out, HostInfo h) {
		out.printf(Locale.ROOT, "  %s %s %s; %s, %d cores, %d threads, %d GiB RAM%n", h.os(), h.osVersion(), h.arch(),
				h.cpuModel(), h.cores(), h.threads(), h.ramBytes() >> 30);
		out.printf(Locale.ROOT, "  JVM vectors: integer %d-bit, float %d-bit; Juno CUDA: %s; Juno ROCm: %s%n",
				h.intVectorBits(), h.floatVectorBits(), h.cuda() ? "yes" : "no", h.rocm() ? "yes" : "no");
		for (HostInfo.Gpu g : h.gpus())
			out.printf(Locale.ROOT, "  GPU: %s %s, %d MiB%n", g.vendor(), g.name(), g.memoryBytes() >> 20);
		if (h.power().hasBattery())
			out.printf(Locale.ROOT, "  Battery: %s%s%n", h.power().onBattery() ? "discharging" : "on mains",
					h.power().percent() >= 0 ? ", " + h.power().percent() + "%" : "");
	}

	private static void printList(PrintStream out, List<Probe> catalog, HostInfo host, Set<String> available,
			Map<String, Path> present) {
		out.println("Probes:");
		for (Probe p : catalog) {
			String why = p.requirements().unmetBy(host);
			for (String m : p.models())
				if (why == null && !available.contains(m))
					why = "model " + m + " is neither present nor downloadable";
			out.printf(Locale.ROOT, "  %-30s %-7s %-17s %s%n", p.id(), p.kind(), String.join(",", p.suites()),
					why == null ? "runnable" : "not runnable: " + why);
		}
		out.println("Models:");
		for (CatalogModel m : CatalogModel.ALL)
			out.printf(Locale.ROOT, "  %-15s %6d MiB  %s%n", m.id(), m.bytes() >> 20,
					present.containsKey(m.id()) ? "present: " + present.get(m.id())
							: "downloads from huggingface.co/" + m.repo());
	}

	private static void printPlan(PrintStream out, String suite, Selection.Result plan, Map<String, Path> present,
			KitOptions o) throws IOException {
		out.println("Plan for suite " + suite + " (nothing is written, downloaded or built):");
		for (Probe p : plan.selected())
			out.printf(Locale.ROOT, "  %-30s about %s on the reference host%n", p.id(),
					Progress.minutes(p.refCostSeconds()));
		printSkipped(out, plan.skipped());
		long bytes = downloadBytes(plan.selected(), present);
		Set<String> models = new LinkedHashSet<>();
		for (Probe p : plan.selected())
			models.addAll(p.models());
		for (String id : models)
			out.printf(Locale.ROOT, "  model %-15s %s%n", id, present.containsKey(id) ? "present: " + present.get(id)
					: "download " + (CatalogModel.byId(id).bytes() >> 20) + " MiB");
		out.printf(Locale.ROOT, "Downloads: %d MiB into %s (%d MiB free).%n", bytes >> 20, o.cacheDir(),
				Files.getFileStore(existingAncestor(o.cacheDir())).getUsableSpace() >> 20);
		Estimator.Range r = Estimator.estimate(plan.selected(), 1.0, 0, 0);
		out.printf(Locale.ROOT, "Time: %s to %s at the reference host's speed, plus downloads; a run calibrates"
				+ " this to the machine before it asks to start.%n", Progress.minutes(r.low().toSeconds()),
				Progress.minutes(r.high().toSeconds()));
	}

	private static void printSkipped(PrintStream out, List<Selection.Skip> skipped) {
		if (skipped.isEmpty())
			return;
		out.println("Not run on this machine:");
		for (Selection.Skip s : skipped)
			out.printf(Locale.ROOT, "  %-30s %s%n", s.probe().id(), s.reason());
	}

	static void warnings(PrintStream out, HostInfo host, List<Probe> probes, long downloadBytes) {
		if (host.power().onBattery())
			out.println("  WARNING: the machine is on battery" + (host.power().percent() >= 0
					? " (" + host.power().percent() + "%)" : "")
					+ ". Plug it in: readings on battery are slower and noisier, and a long run can drain it.");
		if (downloadBytes > 0)
			out.printf(Locale.ROOT, "  Downloads: %d MiB%s. You may use the computer while they run.%n",
					downloadBytes >> 20, "yes".equals(host.metered()) ? " over a METERED connection" : "");
		if (probes.stream().anyMatch(p -> p.requirements().cuda() || p.requirements().rocm()))
			out.println("  Close games, 3D and video programs before the GPU probes: other GPU work changes GPU"
					+ " readings.");
		out.println("  During measurements leave the computer idle: other programs' CPU use is sampled, and a"
				+ " reading taken under load is re-run.");
	}

	private static String menu(PrintStream out, BufferedReader in, List<Probe> catalog, HostInfo host,
			Set<String> available) throws IOException {
		out.println("What should the kit run? (times are at the reference host's speed, before calibration)");
		List<String> suites = List.of("quick", "cpu", "gpu", "all", "custom");
		for (int i = 0; i < suites.size(); i++) {
			String s = suites.get(i);
			if (s.equals("custom")) {
				out.printf(Locale.ROOT, "  %d) custom  choose probes by id%n", i + 1);
				continue;
			}
			Selection.Result r = Selection.select(catalog, s, List.of(), host, available, null, 1.0);
			Estimator.Range e = Estimator.estimate(r.selected(), 1.0, 0, 0);
			out.printf(Locale.ROOT, "  %d) %-7s %d probes runnable, %d not, about %s to %s%n", i + 1, s,
					r.selected().size(), r.skipped().size(), Progress.minutes(e.low().toSeconds()),
					Progress.minutes(e.high().toSeconds()));
		}
		out.print("Choose 1 to 5 [1]: ");
		out.flush();
		String line = in.readLine();
		if (line == null || line.isBlank())
			return "quick";
		try {
			int n = Integer.parseInt(line.trim());
			if (n >= 1 && n <= suites.size())
				return suites.get(n - 1);
		} catch (NumberFormatException ignored) {
		}
		if (suites.contains(line.trim()))
			return line.trim();
		throw new IllegalArgumentException("no such choice: " + line.trim());
	}

	private static List<String> askProbes(PrintStream out, BufferedReader in, List<Probe> catalog) throws IOException {
		for (Probe p : catalog)
			out.printf(Locale.ROOT, "  %-30s %s%n", p.id(), p.title());
		out.print("Probe ids, comma-separated: ");
		out.flush();
		String line = in.readLine();
		List<String> ids = new ArrayList<>();
		if (line != null)
			for (String s : line.split(","))
				if (!s.isBlank())
					ids.add(s.trim());
		return ids;
	}

	private static boolean confirm(PrintStream out, BufferedReader in) throws IOException {
		out.print("Start? [Y/n] ");
		out.flush();
		String line = in.readLine();
		return line == null || line.isBlank() || line.trim().toLowerCase(Locale.ROOT).startsWith("y");
	}

	/** Console output also written to the run's {@code kit.log}; closing it leaves the console open. */
	private static PrintStream tee(PrintStream console, Path log) throws IOException {
		OutputStream file = Files.newOutputStream(log, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
		return new PrintStream(new OutputStream() {
			@Override
			public void write(int b) throws IOException {
				console.write(b);
				file.write(b);
			}

			@Override
			public void write(byte[] b, int off, int len) throws IOException {
				console.write(b, off, len);
				file.write(b, off, len);
			}

			@Override
			public void flush() throws IOException {
				console.flush();
				file.flush();
			}

			@Override
			public void close() throws IOException {
				console.flush();
				file.close();
			}
		}, true, StandardCharsets.UTF_8);
	}
}
