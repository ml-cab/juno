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
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

/**
 * The coverage file, {@code coverage.tsv}: every cross-surface row, model architecture, smoke and compare script
 * and feature, each with the probes that cover it. A line is {@code key}, a title, the probes (comma-separated; a
 * trailing {@code *} names every catalog probe with that prefix) and an optional note, separated by tabs. A key is
 * {@code surface:<row>}, {@code family:<general.architecture>}, {@code script:<file>} or {@code feature:<name>}.
 */
record Coverage(List<Entry> entries) {

	static final String RESOURCE = "coverage.tsv";
	/** The rows of the cross-surface compatibility checklist. */
	static final int SURFACE_ROWS = 15;

	Coverage {
		entries = List.copyOf(entries);
	}

	record Entry(String key, String title, List<String> probes, String note) {

		Entry {
			probes = List.copyOf(probes);
		}

		String kind() {
			int colon = key.indexOf(':');
			return colon < 0 ? "" : key.substring(0, colon);
		}

		String name() {
			return key.substring(key.indexOf(':') + 1);
		}
	}

	static Coverage load() throws IOException {
		try (InputStream in = Coverage.class.getResourceAsStream(RESOURCE)) {
			if (in == null)
				throw new IOException(RESOURCE + " is not on the class path");
			return parse(new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8)).lines().toList());
		}
	}

	/** Blank lines and lines starting with {@code #} are skipped. */
	static Coverage parse(List<String> lines) {
		List<Entry> entries = new ArrayList<>();
		for (String line : lines) {
			if (line.isBlank() || line.startsWith("#"))
				continue;
			String[] f = line.split("\t", -1);
			List<String> probes = f.length > 2
					? Arrays.stream(f[2].split(",")).map(String::strip).filter(p -> !p.isEmpty()).toList()
					: List.of();
			entries.add(new Entry(f[0].strip(), f.length > 1 ? f[1].strip() : "", probes,
					f.length > 3 ? f[3].strip() : ""));
		}
		return new Coverage(entries);
	}

	static boolean isSmokeOrCompareScript(String fileName) {
		return (fileName.startsWith("smoke-") || fileName.startsWith("compare-")) && fileName.endsWith(".sh");
	}

	/**
	 * What does not hold, in file order and then probes, scripts, architectures and rows without a line.
	 *
	 * @param probeIds      the catalog's probe ids
	 * @param scripts       the smoke and compare scripts' file names
	 * @param architectures every {@code general.architecture} value Juno loads
	 */
	List<String> problems(Set<String> probeIds, Set<String> scripts, Set<String> architectures) {
		List<String> problems = new ArrayList<>();
		Set<String> keys = new HashSet<>();
		Set<String> covered = new HashSet<>();
		for (Entry e : entries) {
			if (!keys.add(e.key()))
				problems.add(e.key() + ": listed twice");
			switch (e.kind()) {
			case "surface" -> {
				if (surfaceRow(e.name()) < 0)
					problems.add(e.key() + ": no such cross-surface row");
			}
			case "family" -> {
				if (!architectures.contains(e.name()))
					problems.add(e.key() + ": no such architecture");
			}
			case "script" -> {
				if (!scripts.contains(e.name()))
					problems.add(e.key() + ": no such smoke or compare script");
			}
			case "feature" -> {
			}
			default -> problems.add(e.key() + ": unknown kind");
			}
			if (e.probes().isEmpty())
				problems.add(e.key() + ": no probe");
			for (String p : e.probes()) {
				List<String> matched = matching(p, probeIds);
				if (matched.isEmpty())
					problems.add(e.key() + ": " + (p.endsWith("*") ? p + " matches no catalog probe"
							: "names " + p + ", which is not a catalog probe"));
				covered.addAll(matched);
			}
		}
		new TreeSet<>(probeIds).stream().filter(id -> !covered.contains(id))
				.forEach(id -> problems.add("probe " + id + ": covers no line"));
		new TreeSet<>(scripts).stream().filter(s -> !keys.contains("script:" + s))
				.forEach(s -> problems.add("script " + s + ": no line"));
		new TreeSet<>(architectures).stream().filter(a -> !keys.contains("family:" + a))
				.forEach(a -> problems.add("architecture " + a + ": no line"));
		for (int row = 1; row <= SURFACE_ROWS; row++)
			if (!keys.contains("surface:" + row))
				problems.add("cross-surface row " + row + ": no line");
		return problems;
	}

	/** The probe ids {@code probe} names: itself, or with a trailing {@code *} every id with that prefix. */
	static List<String> matching(String probe, Set<String> probeIds) {
		if (!probe.endsWith("*"))
			return probeIds.contains(probe) ? List.of(probe) : List.of();
		String prefix = probe.substring(0, probe.length() - 1);
		return probeIds.stream().filter(id -> id.startsWith(prefix)).sorted().toList();
	}

	private static int surfaceRow(String name) {
		try {
			int row = Integer.parseInt(name);
			return row >= 1 && row <= SURFACE_ROWS ? row : -1;
		} catch (NumberFormatException e) {
			return -1;
		}
	}
}
