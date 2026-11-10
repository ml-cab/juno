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
package cab.ml.juno.player;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * What a forked node leaves behind when it fails. A node that dies mid-request shows up at the coordinator only
 * as a closed connection; with {@code JUNO_NODE_LOG_DIR} set, its standard error (warnings and errors logged
 * there), its JVM error file and its exit code say why.
 */
@DisplayName("ClusterHarness — what a failing node leaves behind")
class ClusterHarnessNodeDiagnosticsTest {

	@Test
	@DisplayName("without the folder a node's stderr and error file go where they did before")
	void without_the_folder_nothing_changes() throws Exception {
		assertThat(ClusterHarness.nodeLogDir(null)).isNull();
		assertThat(ClusterHarness.nodeLogDir("  ")).isNull();
		assertThat(ClusterHarness.nodeLogDir("/tmp/nodes")).isEqualTo(Path.of("/tmp/nodes"));
		assertThat(ClusterHarness.nodeStderr(null, "node-1", 19092, false)).isEqualTo(ProcessBuilder.Redirect.DISCARD);
		assertThat(ClusterHarness.nodeStderr(null, "node-1", 19092, true)).isEqualTo(ProcessBuilder.Redirect.INHERIT);
		assertThat(ClusterHarness.nodeErrorFileOptions(null)).isEmpty();
	}

	@Test
	@DisplayName("each start of a node appends its stderr to the node's file, after a line naming it")
	void each_start_appends_to_the_nodes_file(@TempDir Path dir) throws Exception {
		Path nodes = dir.resolve("nodes");
		for (int port : new int[] { 19092, 19095 }) {
			Process p = new ProcessBuilder(javaExe(), "-version").redirectOutput(ProcessBuilder.Redirect.DISCARD)
					.redirectError(ClusterHarness.nodeStderr(nodes, "node-1", port, false)).start();
			assertThat(p.waitFor(60, TimeUnit.SECONDS)).isTrue();
		}
		String log = Files.readString(nodes.resolve("node-1.log"));
		int first = log.indexOf("version \"");
		int second = log.indexOf("version \"", first + 1);
		assertThat(first).as("first start's stderr").isPositive();
		assertThat(second).as("second start's stderr").isGreaterThan(first);
		assertThat(log.indexOf("node-1 on port 19092")).isBetween(0, first);
		assertThat(log.indexOf("node-1 on port 19095")).isBetween(first, second);
	}

	@Test
	@DisplayName("with the folder a node logs warnings and errors to its stderr, and nothing below")
	void with_the_folder_warnings_reach_the_nodes_file(@TempDir Path dir) throws Exception {
		Path quiet = runLogging(ClusterHarness.quietLoggingConfig(false), dir, "node-1");
		Path kept = runLogging(ClusterHarness.quietLoggingConfig(true), dir, "node-2");
		assertThat(Files.readString(quiet)).doesNotContain("a severe line", "a warning line", "an info line");
		assertThat(Files.readString(kept)).contains("a severe line", "a warning line").doesNotContain("an info line");
	}

	@Test
	@DisplayName("a node JVM that crashes leaves its error file in the folder")
	void a_crash_leaves_its_error_file_in_the_folder(@TempDir Path dir) throws Exception {
		List<String> cmd = new ArrayList<>(List.of(javaExe(), "-Xmx16m", "-XX:+CrashOnOutOfMemoryError",
				"-XX:-CreateCoredumpOnCrash"));
		cmd.addAll(ClusterHarness.nodeErrorFileOptions(dir));
		cmd.addAll(List.of("-cp", System.getProperty("java.class.path"), FillsTheHeap.class.getName()));
		// its own working directory: without the option the file lands there, not in the folder or this module
		Path cwd = Files.createDirectories(dir.resolve("cwd"));
		Process p = new ProcessBuilder(cmd).directory(cwd.toFile()).redirectOutput(ProcessBuilder.Redirect.DISCARD)
				.redirectError(ProcessBuilder.Redirect.DISCARD).start();
		assertThat(p.waitFor(120, TimeUnit.SECONDS)).isTrue();
		assertThat(p.exitValue()).isNotZero();
		assertThat(dir.resolve("hs_err_pid" + p.pid() + ".log")).exists().content().contains("OutOfMemory");
	}

	@Test
	@DisplayName("a node that exited before the cluster stopped it is named, with its exit code")
	void a_node_that_exited_early_is_named(@TempDir Path dir) throws Exception {
		Process exited = new ProcessBuilder(javaExe(), "-version").redirectError(ProcessBuilder.Redirect.DISCARD)
				.start();
		assertThat(exited.waitFor(60, TimeUnit.SECONDS)).isTrue();
		Process running = new ProcessBuilder(javaExe(), "-cp", System.getProperty("java.class.path"),
				WaitsForStdin.class.getName()).start();
		try {
			assertThat(ClusterHarness.nodesExitedEarly(List.of(exited, running))).singleElement().asString()
					.contains("PID " + exited.pid()).contains("code 0");
		} finally {
			running.destroyForcibly();
		}
	}

	/** Runs {@link LogsAtThreeLevels} with this logging configuration and stderr into the node's file. */
	private static Path runLogging(Path config, Path dir, String nodeId) throws Exception {
		Process p = new ProcessBuilder(javaExe(), "-Djava.util.logging.config.file=" + config, "-cp",
				System.getProperty("java.class.path"), LogsAtThreeLevels.class.getName())
				.redirectOutput(ProcessBuilder.Redirect.DISCARD)
				.redirectError(ClusterHarness.nodeStderr(dir, nodeId, 19092, false)).start();
		assertThat(p.waitFor(60, TimeUnit.SECONDS)).isTrue();
		return dir.resolve(nodeId + ".log");
	}

	private static String javaExe() {
		return ProcessHandle.current().info().command()
				.orElse(Path.of(System.getProperty("java.home"), "bin", "java").toString());
	}

	/** Fills a small heap, so a JVM started with {@code -XX:+CrashOnOutOfMemoryError} writes an error file. */
	static final class FillsTheHeap {
		public static void main(String[] args) {
			List<long[]> held = new ArrayList<>();
			while (true)
				held.add(new long[1 << 16]);
		}
	}

	/** Logs one line at SEVERE, WARNING and INFO. */
	static final class LogsAtThreeLevels {
		public static void main(String[] args) {
			Logger l = Logger.getLogger("cab.ml.juno.node.Example");
			l.log(Level.SEVERE, "a severe line");
			l.log(Level.WARNING, "a warning line");
			l.log(Level.INFO, "an info line");
		}
	}

	/** Runs until its standard input closes, like a node. */
	static final class WaitsForStdin {
		public static void main(String[] args) throws Exception {
			System.in.transferTo(java.io.OutputStream.nullOutputStream());
		}
	}
}
