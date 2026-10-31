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

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.junit.jupiter.api.Test;

import cab.ml.juno.node.ForwardPassHandlerLoader;

class CoverageTest {

	private static final Path SCRIPTS = Path.of("..", "scripts", "performance-tests");

	/** Every cross-surface row, so the fixtures below name none as missing unless they mean to. */
	private static final List<String> SURFACES = IntStream.rangeClosed(1, Coverage.SURFACE_ROWS)
			.mapToObj(n -> "surface:" + n + "\tRow " + n + "\ta").toList();

	@Test
	void theCoverageFileMatchesTheCatalogTheScriptsAndTheArchitectures() throws IOException {
		Set<String> probes = ProbeCatalog.all().stream().map(Probe::id).collect(Collectors.toSet());
		Set<String> scripts;
		try (Stream<Path> s = Files.list(SCRIPTS)) {
			scripts = s.map(p -> p.getFileName().toString()).filter(Coverage::isSmokeOrCompareScript)
					.collect(Collectors.toSet());
		}
		assertThat(scripts).isNotEmpty();
		assertThat(Coverage.load().problems(probes, scripts, ForwardPassHandlerLoader.supportedArchitectures()))
				.isEmpty();
	}

	@Test
	void aLineWithNoProbeIsAProblem() {
		assertThat(problems(List.of("feature:shift\tContext shift\t\t"), Set.of("a")))
				.containsExactly("feature:shift: no probe");
	}

	@Test
	void aProbeScriptOrArchitectureThatDoesNotExistIsAProblem() {
		assertThat(problems(List.of("feature:x\tX\ta, gone", "script:smoke-gone.sh\tGone\ta",
				"family:gone\tGone\ta", "surface:16\tNo such row\ta", "colour:red\tRed\ta"), Set.of("a")))
				.containsExactlyInAnyOrder("feature:x: names gone, which is not a catalog probe",
						"script:smoke-gone.sh: no such smoke or compare script", "family:gone: no such architecture",
						"surface:16: no such cross-surface row", "colour:red: unknown kind");
	}

	@Test
	void aProbeScriptArchitectureOrCrossSurfaceRowWithNoLineIsAProblem() {
		Coverage c = Coverage.parse(List.of("feature:x\tX\ta"));
		assertThat(c.problems(Set.of("a", "b"), Set.of("smoke-new.sh"), Set.of("llama"))).contains("probe b: covers no line",
				"script smoke-new.sh: no line", "architecture llama: no line", "cross-surface row 1: no line",
				"cross-surface row 15: no line");
	}

	@Test
	void aTrailingStarNamesEveryCatalogProbeWithThatPrefixAndMustMatchOne() {
		assertThat(problems(List.of("feature:t\tThroughput\tcpu.throughput.*", "feature:g\tGPU\tgpu.none.*"),
				Set.of("a", "cpu.throughput.a.128", "cpu.throughput.b.512")))
				.containsExactly("feature:g: gpu.none.* matches no catalog probe");
	}

	@Test
	void aKeyListedTwiceIsAProblem() {
		assertThat(problems(List.of("feature:x\tX\ta", "feature:x\tX again\ta"), Set.of("a")))
				.containsExactly("feature:x: listed twice");
	}

	@Test
	void onlySmokeAndCompareScriptsAreScripts() {
		assertThat(Coverage.isSmokeOrCompareScript("smoke-grammar.sh")).isTrue();
		assertThat(Coverage.isSmokeOrCompareScript("compare-lora.sh")).isTrue();
		assertThat(Coverage.isSmokeOrCompareScript("perf-lib.sh")).isFalse();
		assertThat(Coverage.isSmokeOrCompareScript("smoke-notes.md")).isFalse();
	}

	private static List<String> problems(List<String> lines, Set<String> probes) {
		List<String> all = Stream.concat(SURFACES.stream(), lines.stream()).toList();
		return Coverage.parse(all).problems(probes, Set.of(), Set.of());
	}
}
