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

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ManifestTest {

	@Test
	void manifestRoundTrips(@TempDir Path dir) throws Exception {
		Manifest m = new Manifest(Manifest.SCHEMA_VERSION, "run", "0.1.2", "06d6d22d", false, "abc123", "quick",
				List.of("env", "calibration"), "2026-10-28T10:00:00Z", "2026-10-28T10:30:00Z",
				Map.of("os", "Linux", "threads", 12));
		Path f = dir.resolve("manifest.json");
		m.write(f);
		assertThat(Manifest.read(f)).isEqualTo(m);
	}

	@Test
	void theBootstrapsBuildFailureManifestReadsAsAManifest(@TempDir Path dir) throws Exception {
		Path f = dir.resolve("manifest.json");
		java.nio.file.Files.writeString(f,
				KitBootstrap.buildFailureManifest("0.1.2", "06d6d22d", true, "2026-10-28T10:00:00Z", "Linux", "amd64"));
		Manifest m = Manifest.read(f);
		assertThat(m.kind()).isEqualTo("build-failure");
		assertThat(m.schemaVersion()).isEqualTo(Manifest.SCHEMA_VERSION);
		assertThat(m.junoCommit()).isEqualTo("06d6d22d");
		assertThat(m.junoTreeDirty()).isTrue();
		assertThat(m.host()).containsEntry("os", "Linux");
	}

	@Test
	void checkpointRoundTripsAndKnowsWhatRemains(@TempDir Path dir) throws Exception {
		Checkpoint c = Checkpoint.start("quick", List.of("env", "calibration", "cpu.live.tinyllama"),
				List.of("--suite", "quick"), "2026-10-28T10:00:00Z")
				.with(new ProbeOutcome("env", ProbeOutcome.DONE, 1200, 1, "", 0.01, 0, false));
		c.save(dir);
		Checkpoint back = Checkpoint.load(dir);
		assertThat(back).isEqualTo(c);
		assertThat(back.remaining()).containsExactly("calibration", "cpu.live.tinyllama");
	}

	@Test
	void theJunoCommitComesFromACommitArchiveFolderNameWhenThereIsNoGit() {
		assertThat(KitBootstrap.commitFromFolderName("juno-06d6d22d5b1c6e3f6a0b9c8d7e6f5a4b3c2d1e0f"))
				.isEqualTo("06d6d22d5b1c6e3f6a0b9c8d7e6f5a4b3c2d1e0f");
		assertThat(KitBootstrap.commitFromFolderName("juno-main")).isNull();
		assertThat(KitBootstrap.commitFromFolderName("juno")).isNull();
	}
}
