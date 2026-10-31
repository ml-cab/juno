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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class IngestTest {

	private static final String COMMIT = "06d6d22d01f195882b72a67bb22acf55605ff1bd";
	private static final String UTC = "20261101T120000Z";

	@TempDir
	Path tmp;

	@Test
	void aReturnedZipBecomesADraftWithItsProbesReadingsAndAReadmeRow() throws Exception {
		Path zip = zip(run("remote", COMMIT, 1.20, "Sure, three colours", true));
		Path out = Files.createDirectories(tmp.resolve("perf-compare"));

		Path draft = Ingest.run(zip, null, out, UTC, tmp.resolve("scratch"));

		assertThat(draft.getFileName().toString())
				.isEqualTo(UTC + "-remote-linux-amd64-xeon-e5-1650-v2-geforce-gtx-1080");
		String index = Files.readString(draft.resolve("INDEX.md"));
		assertThat(index).contains("| `cpu.throughput.tinyllama.128` | DONE |", "| `grammar.tinyllama` | PASS |",
				"| `cpu.throughput.tinyllama.128` | prefill t/s | t/s | 11 | 10 | 12 | 3 |", COMMIT,
				"No reference run at commit " + COMMIT.substring(0, 8));
		assertThat(Files.readString(draft.resolve("README-row.md"))).startsWith("| [`" + draft.getFileName() + "`]")
				.contains("4 PASS or DONE");
		assertThat(draft.resolve("probes").resolve("grammar.tinyllama.json")).exists();
		assertThat(draft.resolve("manifest.json")).exists();
	}

	@Test
	void correctnessIsComparedWithTheReferenceRunAtTheSameCommit() throws Exception {
		Path out = Files.createDirectories(tmp.resolve("perf-compare"));
		Path ref = run("ref", COMMIT, 1.20, "Sure, three colours", true);
		copyTree(ref, out.resolve("20261031T000000Z-kit-reference"));
		Path otherCommit = run("old", "1111111111111111111111111111111111111111", 1.20, "x", true);
		copyTree(otherCommit, out.resolve("20261101T000000Z-kit-reference"));

		Path draft = Ingest.run(zip(run("remote", COMMIT, 1.25, "Sure, three colors", false)), null, out, UTC,
				tmp.resolve("scratch"));

		String index = Files.readString(draft.resolve("INDEX.md"));
		assertThat(index).contains("20261031T000000Z-kit-reference")
				.doesNotContain("20261101T000000Z-kit-reference");
		assertThat(index).contains("| `quality.tinyllama-1.1b` | mean NLL, integer kernels | 1.25 | 1.2 | "
				+ ReferenceComparison.ABOVE_BOUND + " |");
		assertThat(index).contains("| `cpu.hot-path.tinyllama-1.1b` | threads12.short | ");
		assertThat(index).contains(ReferenceComparison.DIFFERS);
		assertThat(index).contains("| `grammar.tinyllama` | json object | FAIL | PASS | "
				+ ReferenceComparison.FAILS_HERE + " |");
	}

	@Test
	void equalRepliesAndNllWithinTheBoundAreCountedNotListed() throws Exception {
		RunResults ref = RunResults.open(run("ref", COMMIT, 1.20, "Sure", true), tmp.resolve("s1"));
		RunResults remote = RunResults.open(run("remote", COMMIT, 1.205, "Sure", true), tmp.resolve("s2"));

		List<ReferenceComparison.Row> rows = ReferenceComparison.compare(remote, ref);

		assertThat(rows).extracting(ReferenceComparison.Row::verdict).containsOnly(ReferenceComparison.EQUAL,
				ReferenceComparison.WITHIN_BOUND);
		assertThat(rows).hasSize(4);
	}

	@Test
	void aZipFromANewerKitIsRefusedWithAnUpdateMessage() throws Exception {
		Path dir = run("remote", COMMIT, 1.2, "x", true);
		String m = Files.readString(dir.resolve(Manifest.FILE)).replace("\"schemaVersion\" : 1",
				"\"schemaVersion\" : " + (Manifest.SCHEMA_VERSION + 1));
		Files.writeString(dir.resolve(Manifest.FILE), m);

		assertThatThrownBy(() -> RunResults.open(zip(dir), tmp.resolve("scratch")))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("newer kit")
				.hasMessageContaining("update");
	}

	/** There is no older schema yet; a reader ignores fields it does not know and tolerates ones it lacks. */
	@Test
	void fieldsAnotherKitVersionWritesOrLeavesOutDoNotStopTheRead() throws Exception {
		Path dir = run("remote", COMMIT, 1.2, "x", true);
		String m = Files.readString(dir.resolve(Manifest.FILE)).replace("\"kitJarSha256\"", "\"futureField\" : 7,\n"
				+ "  \"kitJarSha256Renamed\"");
		Files.writeString(dir.resolve(Manifest.FILE), m);

		RunResults r = RunResults.open(dir, tmp.resolve("scratch"));
		assertThat(r.manifest().kitJarSha256()).isNull();
		assertThat(r.reports()).containsKey("grammar.tinyllama");
	}

	@Test
	void aZipEntryThatLeavesItsFolderIsRefused() throws Exception {
		Path zip = tmp.resolve("evil.zip");
		try (ZipOutputStream z = new ZipOutputStream(Files.newOutputStream(zip))) {
			z.putNextEntry(new ZipEntry("../escaped.txt"));
			z.write("x".getBytes(StandardCharsets.UTF_8));
			z.closeEntry();
		}

		assertThatThrownBy(() -> RunResults.open(zip, tmp.resolve("scratch").resolve("a")))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("escaped.txt");
		assertThat(tmp.resolve("scratch").resolve("escaped.txt")).doesNotExist();
	}

	@Test
	void aBuildFailureZipDraftsTheEndOfTheBuildLog() throws Exception {
		Path dir = Files.createDirectories(tmp.resolve("failed"));
		Files.writeString(dir.resolve(Manifest.FILE),
				KitBootstrap.buildFailureManifest("0.1.2", COMMIT, false, "20261101T100000Z", "Windows", "amd64"));
		Files.writeString(dir.resolve("build.log"), "[INFO] Building node\n[ERROR] cannot find symbol Foo\n");
		Path out = Files.createDirectories(tmp.resolve("perf-compare"));

		Path draft = Ingest.run(zip(dir), null, out, UTC, tmp.resolve("scratch"));

		assertThat(draft.getFileName().toString()).startsWith(UTC + "-remote-windows-amd64");
		assertThat(Files.readString(draft.resolve("INDEX.md"))).contains("The build failed")
				.contains("[ERROR] cannot find symbol Foo");
	}

	@Test
	void theHostLabelNamesTheHardwareNotThePerson() {
		assertThat(Ingest.hostLabel(Map.of("os", "Linux", "arch", "amd64", "cpuModel",
				"Intel(R) Xeon(R) CPU E5-1650 v2 @ 3.50GHz", "gpus",
				List.of(Map.of("vendor", "NVIDIA", "name", "NVIDIA GeForce GTX 1080")))))
				.isEqualTo("linux-amd64-xeon-e5-1650-v2-geforce-gtx-1080");
		assertThat(Ingest.hostLabel(Map.of("os", "Mac OS X", "arch", "aarch64", "cpuModel", "Apple M2", "gpus",
				List.of()))).isEqualTo("macos-aarch64-apple-m2");
		assertThat(Ingest.hostLabel(Map.of("os", "Windows 11", "arch", "amd64", "cpuModel",
				"AMD Ryzen 7 7840HS w/ Radeon 780M Graphics"))).isEqualTo("windows-amd64-ryzen-7-7840hs-w-radeon-780m");
	}

	// ── fixtures ─────────────────────────────────────────────────────────────

	/** A finished run directory: four probes, one throughput reading, one NLL, one reply, one check. */
	private Path run(String name, String commit, double nll, String reply, boolean jsonCheckPasses)
			throws IOException {
		Path dir = Files.createDirectories(tmp.resolve(name + "-" + commit.substring(0, 4)));
		Map<String, Object> host = new LinkedHashMap<>();
		host.put("os", "Linux");
		host.put("arch", "amd64");
		host.put("cpuModel", "Intel(R) Xeon(R) CPU E5-1650 v2 @ 3.50GHz");
		host.put("threads", 12);
		host.put("gpus", List.of(Map.of("vendor", "NVIDIA", "name", "NVIDIA GeForce GTX 1080")));
		List<String> ids = List.of("cpu.throughput.tinyllama.128", "grammar.tinyllama", "quality.tinyllama-1.1b",
				"cpu.hot-path.tinyllama-1.1b");
		new Manifest(Manifest.SCHEMA_VERSION, "run", "0.1.2", commit, false, "ab12", "custom", ids,
				"20261101T100000Z", "20261101T110000Z", host).write(dir.resolve(Manifest.FILE));
		Checkpoint cp = Checkpoint.start("custom", ids, List.of("--suite", "custom"), "20261101T100000Z");
		cp = cp.with(new ProbeOutcome(ids.get(0), ProbeOutcome.DONE, 60_000, 1, "", 0.01, 0, false));
		cp = cp.with(new ProbeOutcome(ids.get(1), jsonCheckPasses ? ProbeOutcome.PASS : ProbeOutcome.FAIL, 300_000, 1,
				"", 0.02, 0, false));
		cp = cp.with(new ProbeOutcome(ids.get(2), ProbeOutcome.DONE, 120_000, 1, "", 0.01, 0, false));
		cp = cp.with(new ProbeOutcome(ids.get(3), ProbeOutcome.PASS, 120_000, 1, "", 0.01, 0, false));
		cp.save(dir);
		Path probes = Files.createDirectories(dir.resolve("probes"));
		write(probes, new ProbeReport(ids.get(0), ProbeKind.READING, null, List.of(),
				List.of(new ProbeReport.Reading("prefill t/s", "t/s", List.of(12.0, 10.0, 11.0), true, true)),
				Map.of("nPrompt", 128)));
		write(probes, new ProbeReport(ids.get(1), ProbeKind.CHECK, jsonCheckPasses,
				List.of(new ProbeReport.Check("json object", jsonCheckPasses, ""),
						new ProbeReport.Check("tool refusal", true, "")),
				List.of(), Map.of()));
		write(probes, new ProbeReport(ids.get(2), ProbeKind.READING, null, List.of(),
				List.of(new ProbeReport.Reading("mean NLL, integer kernels", "nats", List.of(nll), false, false)),
				Map.of("tokens", 1024)));
		write(probes, new ProbeReport(ids.get(3), ProbeKind.CHECK, true, List.of(), List.of(),
				Map.of("threads12.short", reply, "threads", List.of(4, 12))));
		Files.writeString(dir.resolve("SUMMARY.txt"), "summary\n");
		return dir;
	}

	private static void write(Path probes, ProbeReport r) throws IOException {
		Json.write(probes.resolve(r.id() + ".json"), r);
	}

	private Path zip(Path dir) throws IOException {
		Path zip = tmp.resolve(dir.getFileName() + ".zip");
		List<Path> files;
		try (Stream<Path> s = Files.walk(dir)) {
			files = new ArrayList<>(s.filter(Files::isRegularFile).toList());
		}
		try (OutputStream o = Files.newOutputStream(zip); ZipOutputStream z = new ZipOutputStream(o)) {
			for (Path f : files) {
				z.putNextEntry(new ZipEntry(dir.relativize(f).toString().replace('\\', '/')));
				z.write(Files.readAllBytes(f));
				z.closeEntry();
			}
		}
		return zip;
	}

	private static void copyTree(Path from, Path to) throws IOException {
		try (Stream<Path> s = Files.walk(from)) {
			for (Path p : s.toList()) {
				Path t = to.resolve(from.relativize(p).toString());
				if (Files.isDirectory(p))
					Files.createDirectories(t);
				else
					Files.copy(p, t);
			}
		}
	}
}
