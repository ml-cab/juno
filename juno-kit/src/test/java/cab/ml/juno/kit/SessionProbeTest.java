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
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The session save and restore probe: its reading of a run's line, each check failing on its own, and one real save and
 * restore on the kit's synthetic model on the CPU backend (no model file, no GPU).
 */
class SessionProbeTest {

	@Test
	void theRunsLineIsParsed() throws IOException {
		String out = "loading\n" + SessionRunMain.PREFIX + "{\"tokens\":41,\"bytes\":123456,\"prefixHit\":true,"
				+ "\"secondTurn\":[5,6,7],\"secondTurnText\":\"Blue.\"}\nbye\n";
		SessionProbe.Run run = SessionProbe.parse(out, 0);
		assertThat(run.tokens()).isEqualTo(41);
		assertThat(run.secondTurn()).containsExactly(5, 6, 7);
		assertThatThrownBy(() -> SessionProbe.parse("OutOfMemoryError\n", 1)).isInstanceOf(IOException.class)
				.hasMessageContaining("exit 1");
	}

	@Test
	void eachCheckFailsOnItsOwn() {
		SessionProbe.Run saved = new SessionProbe.Run(41, 1000, true, List.of(5, 6, 7), "Blue.");
		assertThat(SessionProbe.checks("f16", saved, saved)).allSatisfy(c -> assertThat(c.passed()).isTrue());
		assertThat(SessionProbe.checks("f16", saved, new SessionProbe.Run(40, 1000, true, List.of(5, 6, 7), "Blue.")))
				.extracting(ProbeReport.Check::passed).containsExactly(false, true, true);
		assertThat(SessionProbe.checks("f16", saved, new SessionProbe.Run(41, 1000, false, List.of(5, 6, 7), "Blue.")))
				.extracting(ProbeReport.Check::passed).containsExactly(true, false, true);
		assertThat(SessionProbe.checks("f16", saved, new SessionProbe.Run(41, 1000, true, List.of(5, 6, 8), "Blue!")))
				.extracting(ProbeReport.Check::passed).containsExactly(true, true, false);
	}

	/** One real save in one JVM and restore in another, CPU backend, synthetic model. */
	@Test
	void theSyntheticModelRestoresAcrossJvms(@TempDir Path dir) throws Exception {
		Path model = SyntheticGguf.write(dir);
		ProbeReport r = new SessionProbe().run("session.save-restore.synthetic",
				Map.of("model", model.toString(), "types", "f16", "gpu", "false", "childHeap", "1g"));
		assertThat(r.checks()).as(r.toString()).hasSize(3).allSatisfy(c -> assertThat(c.passed()).as(c.detail()).isTrue());
		assertThat(r.passed()).isTrue();
		assertThat(r.facts()).containsEntry("backend", "cpu").containsKey("f16.tokens");
	}
}
