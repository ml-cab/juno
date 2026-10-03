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
package cab.ml.juno.master;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.EnumSet;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import cab.ml.juno.master.ModelLiveChecks.LiveCheck;
import cab.ml.juno.master.ModelLiveChecks.Suite;

/** The {@code ./juno test} entry point's argument handling and exit codes; no model file needed. */
class ModelLiveRunnerTest {

	@TempDir
	Path tmp;

	private final ByteArrayOutputStream buf = new ByteArrayOutputStream();
	private final PrintStream out = new PrintStream(buf, true, StandardCharsets.UTF_8);

	@Test
	void ptype_all_runs_every_suite() {
		assertThat(ModelLiveRunner.suitesFor("all")).isEqualTo(EnumSet.allOf(Suite.class));
		assertThat(ModelLiveRunner.suitesFor(null)).isEqualTo(EnumSet.allOf(Suite.class));
		assertThat(ModelLiveRunner.suitesFor(" ALL ")).isEqualTo(EnumSet.allOf(Suite.class));
	}

	@Test
	void ptype_filters_the_cluster_suite_and_keeps_the_in_process_prefill() {
		assertThat(ModelLiveRunner.suitesFor("pipeline")).containsExactlyInAnyOrder(Suite.PIPELINE, Suite.PREFILL);
		assertThat(ModelLiveRunner.suitesFor("tensor")).containsExactlyInAnyOrder(Suite.TENSOR, Suite.PREFILL);
	}

	@Test
	void unknown_ptype_is_rejected_with_its_value() {
		assertThatThrownBy(() -> ModelLiveRunner.suitesFor("ring")).isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("ring");
	}

	@Test
	void exit_code_is_zero_only_when_every_check_passed() {
		LiveCheck ok = new LiveCheck(1, "a", true, "");
		LiveCheck bad = new LiveCheck(2, "b", false, "why");
		assertThat(ModelLiveRunner.exitCode(List.of(ok, ok))).isZero();
		assertThat(ModelLiveRunner.exitCode(List.of(ok, bad))).isEqualTo(1);
		assertThat(ModelLiveRunner.exitCode(List.of())).as("no check run is not a pass").isEqualTo(1);
	}

	@Test
	void missing_model_argument_prints_usage_and_exits_two() {
		assertThat(ModelLiveRunner.run(new String[0], "all", out)).isEqualTo(2);
		assertThat(printed()).contains("Usage");
	}

	@Test
	void bad_ptype_exits_two() throws IOException {
		Path model = Files.createFile(tmp.resolve("m.gguf"));
		assertThat(ModelLiveRunner.run(new String[] { model.toString() }, "ring", out)).isEqualTo(2);
		assertThat(printed()).contains("ring");
	}

	@Test
	void missing_model_file_exits_one_naming_it() {
		String path = tmp.resolve("absent.gguf").toString();
		assertThat(ModelLiveRunner.run(new String[] { path }, "all", out)).isEqualTo(1);
		assertThat(printed()).contains("absent.gguf");
	}

	@Test
	void unreadable_model_file_exits_one_instead_of_throwing() throws IOException {
		Path junk = Files.writeString(tmp.resolve("junk.gguf"), "not a gguf file");
		assertThat(ModelLiveRunner.run(new String[] { junk.toString() }, "all", out)).isEqualTo(1);
		assertThat(printed()).contains("junk.gguf");
	}

	private String printed() {
		return buf.toString(StandardCharsets.UTF_8);
	}
}
