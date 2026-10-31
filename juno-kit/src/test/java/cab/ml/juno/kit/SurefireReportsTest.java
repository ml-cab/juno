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

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class SurefireReportsTest {

	private static final String PASSING = """
			<?xml version="1.0" encoding="UTF-8"?>
			<testsuite name="cab.ml.juno.node.CudaMatVecTest" tests="3" failures="0" errors="0" skipped="1">
			  <testcase name="a" classname="cab.ml.juno.node.CudaMatVecTest" time="0.1"/>
			  <testcase name="b" classname="cab.ml.juno.node.CudaMatVecTest" time="0.1"/>
			  <testcase name="c" classname="cab.ml.juno.node.CudaMatVecTest" time="0"><skipped/></testcase>
			</testsuite>
			""";
	private static final String FAILING = """
			<?xml version="1.0" encoding="UTF-8"?>
			<testsuite name="cab.ml.juno.node.ResidentQkvPathTest" tests="2" failures="1" errors="1" skipped="0">
			  <testcase name="noDeviceAllocationPerCall" classname="cab.ml.juno.node.ResidentQkvPathTest" time="1">
			    <failure message="expected: 4096 but was: 8192" type="org.opentest4j.AssertionFailedError">trace</failure>
			  </testcase>
			  <testcase name="boom" classname="cab.ml.juno.node.ResidentQkvPathTest" time="1">
			    <error message="CUDA_ERROR_OUT_OF_MEMORY" type="java.lang.IllegalStateException">trace</error>
			  </testcase>
			</testsuite>
			""";

	@Test
	void totalsAndEachFailedTestByName(@TempDir Path dir) throws Exception {
		Path a = Files.writeString(dir.resolve("TEST-a.xml"), PASSING);
		Path b = Files.writeString(dir.resolve("TEST-b.xml"), FAILING);
		SurefireReports.Totals t = SurefireReports.read(List.of(a, b));
		assertThat(t.tests()).isEqualTo(5);
		assertThat(t.failures()).isEqualTo(1);
		assertThat(t.errors()).isEqualTo(1);
		assertThat(t.skipped()).isEqualTo(1);
		assertThat(t.failed()).extracting(ProbeReport.Check::name).containsExactly(
				"ResidentQkvPathTest.noDeviceAllocationPerCall", "ResidentQkvPathTest.boom");
		assertThat(t.failed().get(0).detail()).contains("expected: 4096 but was: 8192");
		assertThat(t.failed().get(1).detail()).contains("CUDA_ERROR_OUT_OF_MEMORY");
	}

	@Test
	void onlyReportsWrittenByThisRunAreRead(@TempDir Path root) throws Exception {
		Path reports = Files.createDirectories(root.resolve("node/target/surefire-reports"));
		Path old = Files.writeString(reports.resolve("TEST-old.xml"), PASSING);
		Files.setLastModifiedTime(old, java.nio.file.attribute.FileTime.fromMillis(1_000));
		Path fresh = Files.writeString(reports.resolve("TEST-new.xml"), PASSING);
		Files.writeString(reports.resolve("not-a-report.txt"), "x");
		assertThat(SurefireReports.find(root, 10_000)).containsExactly(fresh);
	}

	@Test
	void theMavenCommandRunsOneGroupOnTheNamedModulesAndWhatTheyNeed() {
		assertThat(UnitTestsProbe.command(List.of("mvn"), "gpu", "", "node")).containsExactly("mvn", "-B", "test",
				"-Dgroups=gpu", "-pl", "node", "-am", "-Dsurefire.failIfNoSpecifiedTests=false",
				"-Dmaven.test.failure.ignore=true");
	}

	@Test
	void theCpuGroupRunsEveryUntaggedTestAndLeavesTheGpuGroupsToTheirOwnProbes() {
		assertThat(UnitTestsProbe.command(List.of("mvn"), "", "gpu,rocm", "tokenizer,node")).containsExactly("mvn",
				"-B", "test", "-DexcludedGroups=gpu,rocm", "-pl", "tokenizer,node", "-am",
				"-Dsurefire.failIfNoSpecifiedTests=false", "-Dmaven.test.failure.ignore=true");
	}

	@Test
	void theCpuUnitTestProbeCoversTheDocumentedUnitTestModules() {
		Probe p = ProbeCatalog.byId("cpu.unit-tests");
		assertThat(p.args()).containsEntry("excludedGroups", "gpu,rocm").doesNotContainKey("groups");
		assertThat(p.args().get("modules").split(",")).containsExactlyInAnyOrder("tokenizer", "lora", "node",
				"coordinator", "sampler", "kvcache", "health", "registry", "vision", "metrics", "juno-player");
		assertThat(p.requirements().needsGpu()).isFalse();
	}
}
