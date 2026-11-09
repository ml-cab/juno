package cab.ml.juno.player;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.node.NodeMain;

/**
 * How the local cluster stops a forked node. Destroying a process is SIGTERM on Linux, which runs the node's
 * shutdown hooks (its JFR recording is written, its gRPC server drains), and a forced termination on Windows,
 * which runs none. So the harness closes the node's standard input first, the node exits through its hooks on
 * every OS, and only a node that has not exited within the grace period is destroyed.
 */
@DisplayName("ClusterHarness — stopping a forked node")
class ClusterHarnessNodeStopTest {

	private Process node;

	@AfterEach
	void tearDown() {
		if (node != null && node.isAlive())
			node.destroyForcibly();
	}

	@Test
	@DisplayName("a node started with the harness's properties exits with code 0 once its stdin closes")
	void node_exits_cleanly_when_its_stdin_closes() throws Exception {
		node = startNode(ClusterHarness.nodeLifecycleProperties());
		long t0 = System.nanoTime();
		int code = ClusterHarness.stopNode(node, 20_000);
		assertThat(code).as("exit code").isZero();
		assertThat(TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - t0)).as("ms to exit").isLessThan(20_000);
	}

	@Test
	@DisplayName("a node that ignores its stdin is destroyed after the grace period")
	void a_node_ignoring_stdin_is_destroyed() throws Exception {
		node = startNode(List.of());
		ClusterHarness.stopNode(node, 1_000);
		assertThat(node.isAlive()).isFalse();
	}

	@Test
	@DisplayName("the harness binds its nodes to loopback")
	void the_harness_binds_its_nodes_to_loopback() {
		assertThat(ClusterHarness.nodeLifecycleProperties())
				.contains("-D" + cab.ml.juno.node.NodeBindAddress.PROPERTY + "="
						+ cab.ml.juno.node.NodeBindAddress.loopback());
	}

	/** NodeMain in stub mode (no model), the way the harness forks it, after its READY line. */
	private static Process startNode(List<String> properties) throws Exception {
		String javaExe = ProcessHandle.current().info().command()
				.orElse(Path.of(System.getProperty("java.home"), "bin", "java").toString());
		List<String> cmd = new ArrayList<>(List.of(javaExe, "-Xmx256m"));
		cmd.addAll(properties);
		cmd.addAll(List.of("-cp", System.getProperty("java.class.path"), NodeMain.class.getName(), "n1",
				String.valueOf(freePort())));
		Process p = new ProcessBuilder(cmd).redirectError(ProcessBuilder.Redirect.DISCARD).start();
		BufferedReader out = new BufferedReader(new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8));
		String line;
		while ((line = out.readLine()) != null && !line.startsWith("READY:n1"))
			;
		assertThat(line).as("READY line").isNotNull();
		Thread.ofVirtual().start(() -> {
			try {
				out.transferTo(java.io.Writer.nullWriter());
			} catch (Exception ignored) {
				// the node exited
			}
		});
		return p;
	}

	private static int freePort() throws Exception {
		try (ServerSocket s = new ServerSocket(0)) {
			return s.getLocalPort();
		}
	}
}
