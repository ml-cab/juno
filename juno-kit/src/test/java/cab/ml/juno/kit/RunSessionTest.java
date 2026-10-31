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

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.zip.ZipFile;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class RunSessionTest {

	private static Probe probe(String id) {
		return new Probe(id, id, ProbeKind.CHECK, Set.of("quick"), Requirements.cpu(0), List.of(), 10, 1, "EnvProbe",
				Map.of(), 60, "1g");
	}

	private final List<Probe> selected = List.of(probe("a"), probe("b"), probe("c"), probe("d"));

	/** Writes each probe's result file like a real probe would, and stops the run at {@code failAt}. */
	private static RunSession.Executor executor(List<String> calls, String failAt) {
		return (probe, runDir) -> {
			if (probe.id().equals(failAt))
				throw new InterruptedException("interrupted at " + failAt);
			calls.add(probe.id());
			Files.writeString(runDir.resolve("probes").resolve(probe.id() + ".json"), "{\"id\":\"" + probe.id() + "\"}");
			Files.write(runDir.resolve("probes").resolve(probe.id() + ".jfr"), new byte[] { 1, 2, 3 });
			return new ProbeOutcome(probe.id(), ProbeOutcome.PASS, 5, 1, "", 0.0, 0, false);
		};
	}

	@Test
	void anInterruptedRunResumesWithoutRepeatingAnyFinishedProbe(@TempDir Path runDir) throws Exception {
		List<String> calls = new ArrayList<>();
		Checkpoint start = Checkpoint.start("quick", selected.stream().map(Probe::id).toList(), List.of(),
				"2026-10-28T10:00:00Z");
		RunSession first = new RunSession(runDir, selected, start, executor(calls, "c"), RunSession.Listener.NONE);
		assertThatThrownBy(first::run).isInstanceOf(InterruptedException.class);
		assertThat(calls).containsExactly("a", "b");
		assertThat(Checkpoint.load(runDir).remaining()).containsExactly("c", "d");

		List<String> resumed = new ArrayList<>();
		Checkpoint done = new RunSession(runDir, selected, Checkpoint.load(runDir), executor(resumed, null),
				RunSession.Listener.NONE).run();
		assertThat(resumed).containsExactly("c", "d");
		assertThat(done.remaining()).isEmpty();
		assertThat(done.done()).extracting(ProbeOutcome::id).containsExactly("a", "b", "c", "d");
	}

	@Test
	void packWorksOnAPartialRunAndLeavesBinaryRecordingsOut(@TempDir Path tmp) throws Exception {
		Path runDir = tmp.resolve("run");
		Files.createDirectories(runDir);
		Checkpoint start = Checkpoint.start("quick", selected.stream().map(Probe::id).toList(), List.of(),
				"2026-10-28T10:00:00Z");
		RunSession session = new RunSession(runDir, selected, start, executor(new ArrayList<>(), "b"),
				RunSession.Listener.NONE);
		assertThatThrownBy(session::run).isInstanceOf(InterruptedException.class);
		Files.writeString(runDir.resolve("kit.log"), "log of /home/alice\n");

		Path zip = ResultsPacker.pack(runDir, tmp.resolve("out.zip"),
				new KitBootstrap.Privacy("alice", "/home/alice", List.of("box")));

		try (ZipFile z = new ZipFile(zip.toFile())) {
			List<String> names = z.stream().map(e -> e.getName()).toList();
			assertThat(names).contains("checkpoint.json", "probes/a.json", "kit.log");
			assertThat(names).noneMatch(n -> n.endsWith(".jfr"));
			String log = new String(z.getInputStream(z.getEntry("kit.log")).readAllBytes());
			assertThat(log).isEqualTo("log of <home>\n");
		}
	}
}
