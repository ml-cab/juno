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

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.junit.jupiter.api.Test;

class SelectionTest {

	private static Probe probe(String id, Set<String> suites, Requirements req, List<String> models, int cost,
			int priority) {
		return new Probe(id, id, ProbeKind.READING, suites, req, models, cost, priority, "EnvProbe", Map.of(), 600,
				"1g");
	}

	private static final Requirements CPU = Requirements.cpu(1L << 30);

	private final List<Probe> catalog = List.of(probe("env", Set.of("quick", "cpu", "gpu"), CPU, List.of(), 10, 0),
			probe("cpu.a", Set.of("quick", "cpu"), CPU, List.of("tiny"), 600, 1),
			probe("cpu.b", Set.of("cpu"), CPU, List.of("big"), 1200, 2),
			probe("cpu.c", Set.of("quick", "cpu"), CPU, List.of(), 300, 3),
			probe("gpu.a", Set.of("gpu"), Requirements.cuda(1L << 30), List.of("tiny"), 300, 1),
			probe("cpu.wide", Set.of("quick", "cpu"), Requirements.intVectorBits(256), List.of(), 200, 2));

	private final Set<String> models = Set.of("tiny", "big");

	@Test
	void quickSuiteSelectsItsRunnableProbesInCatalogOrder() {
		Selection.Result r = Selection.select(catalog, "quick", List.of(), HostFixtures.cpuOnly128(), models, null,
				1.0);
		assertThat(r.selected()).extracting(Probe::id).containsExactly("env", "cpu.a", "cpu.c");
		assertThat(r.skipped()).extracting(e -> e.probe().id()).containsExactly("cpu.wide");
	}

	@Test
	void allSuiteConsidersEveryProbeAndSkipsWhatTheHostCannotRunWithAReason() {
		Selection.Result r = Selection.select(catalog, "all", List.of(), HostFixtures.cpuOnly128(), models, null, 1.0);
		assertThat(r.selected()).extracting(Probe::id).containsExactly("env", "cpu.a", "cpu.b", "cpu.c");
		assertThat(r.skipped()).extracting(Selection.Skip::reason).anySatisfy(s -> assertThat(s).contains("CUDA"))
				.anySatisfy(s -> assertThat(s).contains("256-bit"));
	}

	@Test
	void aWiderHostRunsTheProbesTheNarrowOneSkips() {
		Selection.Result r = Selection.select(catalog, "all", List.of(), HostFixtures.cuda256(), models, null, 1.0);
		assertThat(r.selected()).extracting(Probe::id).contains("gpu.a", "cpu.wide");
		assertThat(r.skipped()).isEmpty();
	}

	@Test
	void customRunsTheNamedProbesAndRejectsAnUnknownIdNamingTheKnownOnes() {
		Selection.Result r = Selection.select(catalog, "custom", List.of("cpu.c", "env"), HostFixtures.cpuOnly128(),
				models, null, 1.0);
		assertThat(r.selected()).extracting(Probe::id).containsExactly("env", "cpu.c");
		assertThatThrownBy(() -> Selection.select(catalog, "custom", List.of("nope"), HostFixtures.cpuOnly128(),
				models, null, 1.0)).isInstanceOf(IllegalArgumentException.class).hasMessageContaining("nope")
				.hasMessageContaining("cpu.a");
		assertThatThrownBy(() -> Selection.select(catalog, "custom", List.of(), HostFixtures.cpuOnly128(), models,
				null, 1.0)).isInstanceOf(IllegalArgumentException.class).hasMessageContaining("--probes");
	}

	@Test
	void aProbeWhoseModelIsUnavailableIsSkippedNamingTheModel() {
		Selection.Result r = Selection.select(catalog, "cpu", List.of(), HostFixtures.cpuOnly128(), Set.of("tiny"),
				null, 1.0);
		assertThat(r.skipped()).filteredOn(s -> s.probe().id().equals("cpu.b")).singleElement()
				.satisfies(s -> assertThat(s.reason()).contains("big"));
	}

	@Test
	void budgetKeepsTheHighestPriorityProbesThatFitAndAlwaysKeepsMandatoryOnes() {
		// env (0, mandatory) 10 s; cpu.a (1) 600 s fits; cpu.b (2) 1200 s does not; cpu.c (3) 300 s still fits.
		Selection.Result r = Selection.select(catalog, "cpu", List.of(), HostFixtures.cpuOnly128(), models,
				Duration.ofSeconds(1000), 1.0);
		assertThat(r.selected()).extracting(Probe::id).containsExactly("env", "cpu.a", "cpu.c");
		assertThat(r.skipped()).filteredOn(s -> s.probe().id().equals("cpu.b")).singleElement()
				.satisfies(s -> assertThat(s.reason()).contains("budget"));
	}

	@Test
	void budgetIsReadAtTheCalibratedScale() {
		// At twice the reference cost cpu.a takes 1200 s and no longer fits; cpu.c (600 s) does.
		Selection.Result r = Selection.select(catalog, "cpu", List.of(), HostFixtures.cpuOnly128(), models,
				Duration.ofSeconds(1000), 2.0);
		assertThat(r.selected()).extracting(Probe::id).containsExactly("env", "cpu.c");
	}

	@Test
	void anyGpuIsMetByCudaOrRocmAndNamedWhenNeither() {
		assertThat(Requirements.gpu(0).unmetBy(HostFixtures.cuda256())).isNull();
		assertThat(Requirements.gpu(0).unmetBy(HostFixtures.cpuOnly128())).contains("GPU");
	}

	@Test
	void aProbeNeedingMoreDeviceMemoryThanTheLargestGpuIsSkippedNamingBothAndAnUnreadGpuDoesNotBlock() {
		Requirements big = Requirements.cuda(0).withDeviceMemory(10L << 30);
		assertThat(big.unmetBy(HostFixtures.cuda256())).contains("device memory").contains("10 GiB")
				.contains("8 GiB");
		assertThat(Requirements.cuda(0).withDeviceMemory(8L << 30).unmetBy(HostFixtures.cuda256())).isNull();
		HostInfo h = HostFixtures.cuda256();
		HostInfo unread = new HostInfo(h.os(), h.osVersion(), h.arch(), h.cpuModel(), h.cores(), h.threads(),
				h.ramBytes(), h.freeDiskBytes(), h.cpuFlags(), h.intVectorBits(), h.floatVectorBits(), h.useAvx(),
				h.cuda(), h.rocm(), List.of(), h.power(), h.metered(), h.clockSource(), h.backend());
		assertThat(big.unmetBy(unread)).isNull();
	}

	@Test
	void anUnknownSuiteIsRejected() {
		assertThatThrownBy(() -> Selection.select(catalog, "fast", List.of(), HostFixtures.cpuOnly128(), models, null,
				1.0)).isInstanceOf(IllegalArgumentException.class).hasMessageContaining("quick");
	}
}
