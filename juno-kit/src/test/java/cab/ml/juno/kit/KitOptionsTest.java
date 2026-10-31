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

import java.nio.file.Path;
import java.time.Duration;
import java.util.List;

import org.junit.jupiter.api.Test;

class KitOptionsTest {

	private final Path root = Path.of("/repo");

	@Test
	void budgetDurations() {
		assertThat(KitOptions.duration("45m")).isEqualTo(Duration.ofMinutes(45));
		assertThat(KitOptions.duration("2h")).isEqualTo(Duration.ofHours(2));
		assertThat(KitOptions.duration("1h30m")).isEqualTo(Duration.ofMinutes(90));
		assertThat(KitOptions.duration("90")).isEqualTo(Duration.ofMinutes(90));
		assertThatThrownBy(() -> KitOptions.duration("soon")).isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	void probesImplyTheCustomSuiteAndDefaultsArePlain() {
		KitOptions o = KitOptions.parse(List.of("--probes", "env,calibration", "--yes"), root);
		assertThat(o.suite()).isEqualTo("custom");
		assertThat(o.probes()).containsExactly("env", "calibration");
		assertThat(o.yes()).isTrue();
		assertThat(o.resultsRoot()).isEqualTo(root.resolve("juno-kit-results"));
		assertThat(KitOptions.parse(List.of(), root).suite()).isNull();
	}

	@Test
	void resumeAndPackNeedARunDirectoryAndUnknownInputIsRejected() {
		assertThat(KitOptions.parse(List.of("resume", "/r/run1"), root).runDir()).isEqualTo(Path.of("/r/run1"));
		assertThatThrownBy(() -> KitOptions.parse(List.of("pack"), root)).isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("run directory");
		assertThatThrownBy(() -> KitOptions.parse(List.of("--fast"), root))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("--fast");
		assertThatThrownBy(() -> KitOptions.parse(List.of("--suite", "fast"), root))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("quick");
	}

	@Test
	void ingestTakesTheZipAndDraftsIntoPerfCompareUnlessToldOtherwise() {
		KitOptions o = KitOptions.parse(List.of("ingest", "/z/r.zip", "--reference", "/p/ref"), root);
		assertThat(o.command()).isEqualTo("ingest");
		assertThat(o.runDir()).isEqualTo(Path.of("/z/r.zip"));
		assertThat(o.reference()).isEqualTo(Path.of("/p/ref"));
		assertThat(o.referenceRun()).isFalse();
		assertThat(o.resultsRoot()).isEqualTo(root.resolve("docs").resolve("perf-compare"));
		assertThat(KitOptions.parse(List.of("ingest", "/z/r.zip", "--reference-run", "--out", "/o"), root))
				.satisfies(r -> assertThat(r.referenceRun()).isTrue())
				.satisfies(r -> assertThat(r.resultsRoot()).isEqualTo(Path.of("/o")));
		assertThatThrownBy(() -> KitOptions.parse(List.of("ingest"), root))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("results zip");
	}

	@Test
	void referenceOptionsOutsideIngestAreRefusedRatherThanIgnored() {
		assertThatThrownBy(() -> KitOptions.parse(List.of("--reference-run"), root))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("ingest");
		assertThatThrownBy(() -> KitOptions.parse(List.of("--suite", "quick", "--reference", "/p"), root))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("ingest");
		assertThatThrownBy(() -> KitOptions.parse(List.of("ingest", "/z/r.zip", "--reference", "/p",
				"--reference-run"), root)).isInstanceOf(IllegalArgumentException.class);
	}
}
