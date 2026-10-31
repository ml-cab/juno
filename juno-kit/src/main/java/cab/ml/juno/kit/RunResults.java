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
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * A run's results as {@code ingest} reads them, from a results zip or a run directory. A zip is extracted into a
 * scratch directory first, and an entry that would land outside it is refused: the zips come from other machines.
 *
 * @param checkpoint {@code null} for the zip the bootstrap writes when the build fails
 * @param reports    each finished probe's report, in the order the probes finished; a probe that crashed may have none
 */
record RunResults(Path dir, Manifest manifest, Checkpoint checkpoint, Map<String, ProbeReport> reports) {

	static RunResults open(Path zipOrDir, Path scratch) throws IOException {
		Path dir = Files.isDirectory(zipOrDir) ? zipOrDir : extract(zipOrDir, scratch);
		Path m = dir.resolve(Manifest.FILE);
		if (!Files.isRegularFile(m))
			throw new IllegalArgumentException(zipOrDir + " is not a kit results zip: it has no " + Manifest.FILE);
		Manifest manifest = Manifest.read(m);
		if (manifest.schemaVersion() > Manifest.SCHEMA_VERSION)
			throw new IllegalArgumentException(zipOrDir + " was made by a newer kit (manifest schema "
					+ manifest.schemaVersion() + ", this kit reads up to " + Manifest.SCHEMA_VERSION
					+ "): update the kit and ingest it again");
		if (!"run".equals(manifest.kind()))
			return new RunResults(dir, manifest, null, Map.of());
		Checkpoint cp = Checkpoint.load(dir);
		Map<String, ProbeReport> reports = new LinkedHashMap<>();
		for (ProbeOutcome o : cp.done()) {
			Path f = dir.resolve(RunSession.PROBES_DIR).resolve(o.id() + ".json");
			if (Files.isRegularFile(f))
				reports.put(o.id(), Json.read(f, ProbeReport.class));
		}
		return new RunResults(dir, manifest, cp, reports);
	}

	boolean buildFailed() {
		return checkpoint == null;
	}

	private static Path extract(Path zip, Path into) throws IOException {
		Path root = into.toAbsolutePath().normalize();
		Files.createDirectories(root);
		try (InputStream in = Files.newInputStream(zip); ZipInputStream z = new ZipInputStream(in)) {
			for (ZipEntry e; (e = z.getNextEntry()) != null;) {
				Path target = root.resolve(e.getName()).normalize();
				if (!target.startsWith(root) || target.equals(root))
					throw new IllegalArgumentException(zip + ": the entry " + e.getName()
							+ " would land outside the folder it is extracted into");
				if (e.isDirectory()) {
					Files.createDirectories(target);
					continue;
				}
				Files.createDirectories(target.getParent());
				Files.copy(z, target);
			}
		}
		return root;
	}
}
