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
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;

/**
 * {@code juno-kit ingest <zip>}, on the owner's side: reads a returned results zip and drafts its perf-compare entry,
 * {@code <out>/<UTC>-remote-<host label>/}, holding {@code INDEX.md}, {@code README-row.md} (the row for the
 * perf-compare README) and the zip's manifest, checkpoint, summary and probe reports. Correctness is compared with the
 * newest reference run at the same Juno commit, {@code <out>/<UTC>-kit-reference/}, or the run {@code --reference}
 * names. With {@code --reference-run} the zip is the owner's own run on the reference host, drafted under that name
 * for later ingests to compare with. The owner reviews a draft before publishing it.
 */
final class Ingest {

	static final String REFERENCE_SUFFIX = "-kit-reference";
	private static final Set<String> DATA_FILES = Set.of(Manifest.FILE, Checkpoint.FILE, "SUMMARY.txt", "host.json",
			"skipped.json", "estimate.json");
	/** Words of a CPU or GPU name that say nothing about the part. */
	private static final Set<String> NOISE_WORDS = Set.of("intel", "amd", "nvidia", "cpu", "processor", "graphics");

	private Ingest() {
	}

	/**
	 * @param reference {@code null} to look for the reference run in {@code outRoot}
	 * @param scratch   where zips are extracted
	 * @return the draft directory
	 */
	static Path run(Path input, Path reference, Path outRoot, String utc, Path scratch) throws IOException {
		return run(input, reference, outRoot, utc, scratch, false);
	}

	static Path run(Path input, Path reference, Path outRoot, String utc, Path scratch, boolean asReference)
			throws IOException {
		RunResults remote = RunResults.open(input, scratch.resolve("remote"));
		RunResults ref = null;
		String refName = null;
		if (!asReference && !remote.buildFailed()) {
			Path refPath = reference != null ? reference
					: findReference(outRoot, remote.manifest().junoCommit()).orElse(null);
			if (refPath != null) {
				ref = RunResults.open(refPath, scratch.resolve("reference"));
				refName = refPath.getFileName().toString();
			}
		}
		String name = utc + (asReference ? REFERENCE_SUFFIX : "-remote-" + hostLabel(remote.manifest().host()));
		Path draft = outRoot.resolve(name);
		if (Files.exists(draft))
			throw new IllegalArgumentException(draft + " exists already");
		Files.createDirectories(draft);
		copyData(remote.dir(), draft);
		List<ReferenceComparison.Row> rows = ref == null ? List.of() : ReferenceComparison.compare(remote, ref);
		Files.writeString(draft.resolve("INDEX.md"),
				IngestReport.index(name, remote, ref, refName, rows, Coverage.load(), asReference));
		Files.writeString(draft.resolve("README-row.md"), IngestReport.readmeRow(name, remote, ref, rows) + "\n");
		return draft;
	}

	/** The newest {@code *-kit-reference} directory in {@code outRoot} whose manifest names {@code commit}. */
	static Optional<Path> findReference(Path outRoot, String commit) throws IOException {
		if (commit == null || !Files.isDirectory(outRoot))
			return Optional.empty();
		List<Path> candidates;
		try (Stream<Path> s = Files.list(outRoot)) {
			candidates = s.filter(p -> p.getFileName().toString().endsWith(REFERENCE_SUFFIX))
					.filter(p -> Files.isRegularFile(p.resolve(Manifest.FILE)))
					.sorted(Comparator.comparing((Path p) -> p.getFileName().toString()).reversed()).toList();
		}
		for (Path p : candidates)
			if (Objects.equals(Manifest.read(p.resolve(Manifest.FILE)).junoCommit(), commit))
				return Optional.of(p);
		return Optional.empty();
	}

	/**
	 * The hardware as a folder-name label: OS, architecture, CPU and first GPU, lower case. It names parts, never the
	 * person or the machine's name.
	 */
	static String hostLabel(Map<String, Object> host) {
		List<String> parts = new ArrayList<>();
		String os = String.valueOf(host.getOrDefault("os", "unknown")).toLowerCase(Locale.ROOT);
		parts.add(os.startsWith("mac") ? "macos" : os.startsWith("windows") ? "windows" : os.split("\\s+")[0]);
		parts.add(String.valueOf(host.getOrDefault("arch", "unknown")).toLowerCase(Locale.ROOT));
		if (host.get("cpuModel") instanceof String cpu)
			parts.add(part(cpu));
		if (host.get("gpus") instanceof List<?> gpus && !gpus.isEmpty() && gpus.get(0) instanceof Map<?, ?> g
				&& g.get("name") instanceof String gpu)
			parts.add(part(gpu));
		return String.join("-", parts.stream().filter(s -> !s.isEmpty()).toList());
	}

	private static String part(String name) {
		String n = name.toLowerCase(Locale.ROOT).replace("(r)", "").replace("(tm)", "");
		int at = n.indexOf('@');
		if (at >= 0)
			n = n.substring(0, at);
		List<String> words = new ArrayList<>();
		for (String w : n.split("[^a-z0-9]+"))
			if (!w.isEmpty() && !NOISE_WORDS.contains(w))
				words.add(w);
		return String.join("-", words);
	}

	private static void copyData(Path from, Path draft) throws IOException {
		for (String f : DATA_FILES)
			if (Files.isRegularFile(from.resolve(f)))
				Files.copy(from.resolve(f), draft.resolve(f));
		Path probes = from.resolve(RunSession.PROBES_DIR);
		if (!Files.isDirectory(probes))
			return;
		Path to = Files.createDirectories(draft.resolve(RunSession.PROBES_DIR));
		try (Stream<Path> s = Files.list(probes)) {
			for (Path p : s.filter(p -> p.getFileName().toString().endsWith(".json")).toList())
				Files.copy(p, to.resolve(p.getFileName().toString()));
		}
	}
}
