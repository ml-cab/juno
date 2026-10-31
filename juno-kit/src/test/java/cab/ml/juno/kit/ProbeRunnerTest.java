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
import java.util.Map;
import java.util.Set;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** The runner against real child JVMs: a probe that works, one that crashes, one that times out. */
class ProbeRunnerTest {

	private static Probe probe(String id, ProbeKind kind, String main, Map<String, String> args, int timeout) {
		return new Probe(id, id, kind, Set.of("quick"), Requirements.cpu(0), List.of(), 10, 1, main, args, timeout,
				"512m");
	}

	@Test
	void aProbeRunsInItsOwnJvmAndLeavesItsReportAndLog(@TempDir Path runDir) throws Exception {
		ProbeOutcome o = new ProbeRunner(Map.of()).execute(probe("env", ProbeKind.READING, "EnvProbe", Map.of(), 120),
				runDir);
		assertThat(o.status()).isEqualTo(ProbeOutcome.DONE);
		assertThat(o.attempts()).isEqualTo(1);
		ProbeReport r = Json.read(runDir.resolve("probes/env.json"), ProbeReport.class);
		assertThat(r.facts()).containsKey("intVectorBits");
		assertThat(Files.readString(runDir.resolve("probes/env.log"))).contains("== attempt 1:")
				.contains("--add-modules jdk.incubator.vector");
	}

	@Test
	void aProbeGetsItsFirstModelAsModelAndEveryModelByCatalogId() {
		Probe p = new Probe("two", "two", ProbeKind.CHECK, Set.of("quick"), Requirements.cpu(0),
				List.of("qwen2.5-3b", "tinyllama-1.1b"), 10, 1, "ToolsProbe", Map.of("k", "v"), 60, "1g");
		List<String> args = ProbeRunner.probeArgs(p, Path.of("r.json"),
				Map.of("qwen2.5-3b", Path.of("/m/q.gguf"), "tinyllama-1.1b", Path.of("/m/t.gguf")), Path.of("/run/p"));
		assertThat(args).containsExactly("ToolsProbe", "two", "r.json", "k=v", "model=/m/q.gguf",
				"model.qwen2.5-3b=/m/q.gguf", "model.tinyllama-1.1b=/m/t.gguf", "logDir=/run/p");
	}

	@Test
	void aCrashIsRecordedAndDoesNotThrow(@TempDir Path runDir) throws Exception {
		ProbeOutcome o = new ProbeRunner(Map.of())
				.execute(probe("broken", ProbeKind.CHECK, "NoSuchProbe", Map.of(), 120), runDir);
		assertThat(o.status()).isEqualTo(ProbeOutcome.ERROR);
		assertThat(o.note()).contains("exit code 1").contains("probes/broken.log");
		assertThat(Files.readString(runDir.resolve("probes/broken.log"))).contains("unknown probe class NoSuchProbe");
	}

	@Test
	void aProbeThatOverrunsItsTimeoutIsStoppedAndRecorded(@TempDir Path runDir) throws Exception {
		ProbeOutcome o = new ProbeRunner(Map.of()).execute(probe("slow", ProbeKind.READING, "CalibrationProbe",
				Map.of("rows", "256", "cols", "256", "warmupMs", "600000"), 3), runDir);
		assertThat(o.status()).isEqualTo(ProbeOutcome.TIMEOUT);
		assertThat(o.note()).contains("no result after 3 s");
	}
}
