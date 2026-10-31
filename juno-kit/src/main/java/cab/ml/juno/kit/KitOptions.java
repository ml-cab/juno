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
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import cab.ml.juno.player.HfGgufFetcher;

/**
 * The kit's command line.
 *
 * @param command {@code run} (default), {@code resume <run dir>}, {@code pack <run dir>} or {@code ingest <zip>}
 * @param runDir  the run directory of {@code resume} and {@code pack}; the results zip, or run directory, of
 *                {@code ingest}
 * @param suite   {@code null} when none was given (the menu asks, or {@code quick} with {@code --yes})
 */
record KitOptions(String command, Path runDir, String suite, List<String> probes, Duration budget, boolean yes,
		boolean plan, boolean list, boolean help, Path modelsDir, Path cacheDir, Path resultsRoot, List<String> raw,
		Path reference, boolean referenceRun) {

	static final String USAGE = """
			Usage: juno-kit.sh [options]            run probes (asks what to run when a console is attached)
			       juno-kit.sh resume <run dir>     continue an interrupted run
			       juno-kit.sh pack <run dir>       pack a run, finished or not, into the results zip
			       juno-kit.sh ingest <zip>         draft a returned zip's perf-compare entry (owner side)
			On Windows the command is juno-kit.bat.

			Options:
			  --suite quick|cpu|gpu|all|custom  what to run (quick: TinyLlama only)
			  --probes id,id,...                 run these probes (implies --suite custom)
			  --budget 45m|2h|1h30m              run the highest-priority probes that fit in this time
			  --yes                              no questions; --suite defaults to quick
			  --plan                             show what would run, the time, the downloads and the disk
			                                     needed, and stop; writes, downloads and builds nothing
			  --list                             list the probes and models, with what this machine can run
			  --models-dir <dir>                 look for model files here first (never written)
			  --cache-dir <dir>                  where downloaded models go (default ~/.cache/juno/models/kit)
			  --out <dir>                        where results go (default juno-kit-results/)
			  --skip-build                       start the kit already built (launcher only)
			  --reference <dir or zip>           ingest: compare with this reference run instead of the newest
			                                     *-kit-reference entry at the zip's commit
			  --reference-run                    ingest: the zip is the reference host's own run; draft it as
			                                     <UTC>-kit-reference (--out defaults to docs/perf-compare)
			  -h, --help                         this text
			""";

	static KitOptions parse(List<String> args, Path root) {
		String command = "run";
		Path runDir = null;
		String suite = null;
		List<String> probes = new ArrayList<>();
		Duration budget = null;
		boolean yes = false, plan = false, list = false, help = false;
		Path modelsDir = null;
		Path cacheDir = HfGgufFetcher.defaultCacheDir().resolve("kit");
		Path out = null;
		Path reference = null;
		boolean referenceRun = false;
		int i = 0;
		if (!args.isEmpty() && (args.get(0).equals("resume") || args.get(0).equals("pack"))) {
			command = args.get(0);
			if (args.size() < 2)
				throw new IllegalArgumentException(command + " needs the run directory, for example juno-kit.sh "
						+ command + " juno-kit-results/20261028T101500Z-quick");
			runDir = Path.of(args.get(1)).toAbsolutePath().normalize();
			i = 2;
		} else if (!args.isEmpty() && args.get(0).equals("ingest")) {
			command = "ingest";
			if (args.size() < 2 || args.get(1).startsWith("--"))
				throw new IllegalArgumentException("ingest needs the results zip, for example juno-kit.sh ingest "
						+ "juno-kit-results-linux-amd64-20261030T220016Z.zip");
			runDir = Path.of(args.get(1)).toAbsolutePath().normalize();
			i = 2;
		}
		for (; i < args.size(); i++) {
			String a = args.get(i);
			switch (a) {
			case "--suite" -> suite = value(args, ++i, a);
			case "--probes" -> {
				for (String p : value(args, ++i, a).split(","))
					if (!p.isBlank())
						probes.add(p.trim());
			}
			case "--budget" -> budget = duration(value(args, ++i, a));
			case "--yes", "-y" -> yes = true;
			case "--plan" -> plan = true;
			case "--list" -> list = true;
			case "--models-dir" -> modelsDir = Path.of(value(args, ++i, a)).toAbsolutePath().normalize();
			case "--cache-dir" -> cacheDir = Path.of(value(args, ++i, a)).toAbsolutePath().normalize();
			case "--out" -> out = Path.of(value(args, ++i, a)).toAbsolutePath().normalize();
			case "--skip-build" -> {
			}
			case "--reference" -> reference = Path.of(value(args, ++i, a)).toAbsolutePath().normalize();
			case "--reference-run" -> referenceRun = true;
			case "-h", "--help" -> help = true;
			default -> throw new IllegalArgumentException("unknown option " + a + "\n\n" + USAGE);
			}
		}
		if ((reference != null || referenceRun) && !command.equals("ingest"))
			throw new IllegalArgumentException("--reference and --reference-run belong to ingest");
		if (reference != null && referenceRun)
			throw new IllegalArgumentException("a reference run is not compared with another; give --reference or"
					+ " --reference-run, not both");
		if (out == null)
			out = command.equals("ingest") ? root.resolve("docs").resolve("perf-compare")
					: root.resolve(KitBootstrap.RESULTS_DIR);
		if (suite == null && !probes.isEmpty())
			suite = "custom";
		if (suite != null && !Selection.SUITES.contains(suite))
			throw new IllegalArgumentException("unknown suite '" + suite + "'; choose one of " + Selection.SUITES);
		return new KitOptions(command, runDir, suite, List.copyOf(probes), budget, yes, plan, list, help, modelsDir,
				cacheDir, out, List.copyOf(args), reference, referenceRun);
	}

	private static String value(List<String> args, int i, String flag) {
		if (i >= args.size())
			throw new IllegalArgumentException(flag + " needs a value");
		return args.get(i);
	}

	/** {@code 45m}, {@code 2h}, {@code 1h30m}, {@code 90s}; a bare number is minutes. */
	static Duration duration(String s) {
		String t = s.trim().toLowerCase(java.util.Locale.ROOT);
		if (t.matches("\\d+"))
			return Duration.ofMinutes(Long.parseLong(t));
		Matcher m = Pattern.compile("(?:(\\d+)h)?(?:(\\d+)m)?(?:(\\d+)s)?").matcher(t);
		if (t.isEmpty() || !m.matches())
			throw new IllegalArgumentException("--budget takes a duration such as 45m, 2h or 1h30m, not " + s);
		long sec = 0;
		if (m.group(1) != null)
			sec += Long.parseLong(m.group(1)) * 3600;
		if (m.group(2) != null)
			sec += Long.parseLong(m.group(2)) * 60;
		if (m.group(3) != null)
			sec += Long.parseLong(m.group(3));
		return Duration.ofSeconds(sec);
	}
}
