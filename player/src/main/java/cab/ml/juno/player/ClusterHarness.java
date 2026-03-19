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

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

import cab.ml.juno.node.CudaAvailability;
import org.bytedeco.javacpp.Loader;

/**
 * Manages the lifecycle of 3 forked node JVM processes for integration testing.
 *
 * Each node is launched as a separate JVM via ProcessBuilder running NodeMain.
 * The harness waits for each node to print "READY:<nodeId>:<port>" before
 * proceeding — so tests don't start sending requests to nodes that aren't up
 * yet.
 *
 * Memory allocation per node JVM (16 GB host): -Xms512m -Xmx4g -XX:+UseZGC
 *
 * Usage: ClusterHarness harness = ClusterHarness.threeNodes(); harness.start();
 * // ... run tests using harness.pipelineClient() ... harness.stop();
 *
 * Or with try-with-resources (implements AutoCloseable): try (ClusterHarness
 * harness = ClusterHarness.threeNodes()) { harness.start(); // ... tests ... }
 */
public final class ClusterHarness implements AutoCloseable {

	private static final Logger log = Logger.getLogger(ClusterHarness.class.getName());

	/** How long to wait for each node to report READY (ms). */
	private static final long NODE_STARTUP_TIMEOUT_MS = 30_000;

	/** gRPC ports for the three nodes. */
	static final int NODE_1_PORT = 19092;
	static final int NODE_2_PORT = 19093;
	static final int NODE_3_PORT = 19094;

	/** TinyLlama-1.1B shape constants — matches EmbeddedNodeServer defaults. */
	private static final int VOCAB_SIZE = EmbeddedNodeServer.VOCAB_SIZE;
	private static final int TOTAL_LAYERS = EmbeddedNodeServer.TOTAL_LAYERS;

	private final List<NodeSpec> specs;
	private final List<Process> processes = new ArrayList<>();
	private ProcessPipelineClient pipelineClient;
	private static String modelPath; // null = stub mode

	private ClusterHarness(List<NodeSpec> specs) {
		this(specs, null);
	}

	private ClusterHarness(List<NodeSpec> specs, String pathToModel) {
		this.specs = specs;
		modelPath = pathToModel;
	}

	/**
	 * Create a standard 3-node cluster.
	 *
	 * Layer split for 22-layer TinyLlama across 3 nodes: Node-1: layers 0– 7 (8
	 * layers) + embeddings Node-2: layers 8–14 (7 layers) Node-3: layers 15–21 (7
	 * layers) + output projection
	 */
	public static ClusterHarness threeNodes() {
		return threeNodes(null);
	}

	/**
	 * Create a 3-node cluster that loads real weights from a GGUF file. Each node
	 * receives the model path as a CLI arg and loads its layer shard via
	 * CpuForwardPassHandler when loadShard() is called.
	 *
	 * Uses the default 22-layer split (TinyLlama). For other models call
	 * {@link #threeNodes(String, int)} and pass the actual layer count.
	 *
	 * @param modelPath path to a GGUF file, e.g.
	 *                  "/models/TinyLlama-1.1B-Chat.Q4_K_M.gguf" Pass null for stub
	 *                  mode (no real model).
	 */
	public static ClusterHarness threeNodes(String modelPath) {
		return threeNodes(modelPath, TOTAL_LAYERS);
	}

	/**
	 * Create a 3-node cluster for a model with {@code totalLayers} transformer
	 * layers. Shard boundaries are computed by dividing layers as evenly as
	 * possible across the three nodes (ceiling-first distribution).
	 *
	 * <pre>
	 *  22 layers → [0,8)  [8,15)  [15,22)   (TinyLlama-1.1B)
	 *  16 layers → [0,6)  [6,11)  [11,16)   (Llama-3.2-1B)
	 * </pre>
	 *
	 * @param modelPath   path to a GGUF/llamafile, or null for stub mode
	 * @param totalLayers actual number of transformer layers in the model
	 */
	public static ClusterHarness threeNodes(String modelPath, int totalLayers) {
		if (totalLayers < 3)
			throw new IllegalArgumentException("totalLayers must be >= 3 to split across 3 nodes, got: " + totalLayers);

		// Ceiling-first: give the extra layers to the earlier nodes.
		int base = totalLayers / 3;
		int extra = totalLayers % 3;
		int end1 = base + (extra > 0 ? 1 : 0); // node-1 layer count
		int end2 = end1 + base + (extra > 1 ? 1 : 0); // node-2 layer count
		// node-3 gets the rest up to totalLayers

		return new ClusterHarness(List.of(
				new NodeSpec("node-1", "localhost", NODE_1_PORT,
						new ProcessPipelineClient.ShardConfig(0, end1, true, false)),
				new NodeSpec("node-2", "localhost", NODE_2_PORT,
						new ProcessPipelineClient.ShardConfig(end1, end2, false, false)),
				new NodeSpec("node-3", "localhost", NODE_3_PORT,
						new ProcessPipelineClient.ShardConfig(end2, totalLayers, false, true))),
				modelPath);
	}

	/** JavaCPP cache dir (where jnicudart etc. are extracted); null if not available. */
	private static String javacppCacheDir;

	/**
	 * Fork all three node JVMs and wait until each reports READY.
	 */
	public void start() throws IOException, InterruptedException {
		// Flat cache so java.library.path to cache root finds jnicudart.dll (Loader otherwise uses subdirs per jar).
		if (System.getProperty("org.bytedeco.javacpp.cachedir.nosubdir") == null) {
			System.setProperty("org.bytedeco.javacpp.cachedir.nosubdir", "true");
		}
		// Trigger CUDA/JavaCPP native load in this JVM so cache is populated; forked nodes need it on java.library.path.
		try {
			CudaAvailability.isAvailable();
			File cache = Loader.getCacheDir();
			if (cache != null) {
				javacppCacheDir = cache.getAbsolutePath();
				if (!cache.exists()) {
					cache.mkdirs();
				}
				log.fine("JavaCPP cache dir: " + javacppCacheDir);
			}
		} catch (Throwable t) {
			log.fine("JavaCPP cache not available: " + t.getMessage());
		}

		for (NodeSpec spec : specs) {
			Process proc = launchNode(spec.nodeId(), spec.port());
			processes.add(proc);
			waitForReady(proc, spec.nodeId());
			log.info("Node [" + spec.nodeId() + "] is up on port " + spec.port());
		}

		// Wire up the pipeline client
		List<ProcessPipelineClient.NodeAddress> addresses = specs.stream()
				.map(s -> new ProcessPipelineClient.NodeAddress(s.host(), s.port())).toList();

		pipelineClient = new ProcessPipelineClient(addresses, VOCAB_SIZE);

		// Tell each node which shard it owns
		List<ProcessPipelineClient.ShardConfig> shards = specs.stream().map(NodeSpec::shard).toList();
		pipelineClient.loadShards(shards);

		log.info("Cluster ready — 3 nodes, " + TOTAL_LAYERS + " total layers distributed");
	}

	/**
	 * Returns the InferencePipeline that routes forward passes across the 3 nodes.
	 * Only valid after start() has been called.
	 */
	public ProcessPipelineClient pipelineClient() {
		if (pipelineClient == null)
			throw new IllegalStateException("ClusterHarness not started — call start() first");
		return pipelineClient;
	}

	/**
	 * Returns the node addresses in pipeline order. Use this to create additional
	 * ProcessPipelineClient instances (e.g. with a different ActivationDtype)
	 * pointing at the same running nodes.
	 */
	public List<ProcessPipelineClient.NodeAddress> nodeAddresses() {
		return specs.stream().map(s -> new ProcessPipelineClient.NodeAddress(s.host(), s.port())).toList();
	}

	/**
	 * Shut down all node processes and close gRPC channels.
	 */
	public void stop() throws InterruptedException {
		if (pipelineClient != null) {
			pipelineClient.shutdown();
		}
		for (Process proc : processes) {
			proc.destroy();
			proc.waitFor(5, TimeUnit.SECONDS);
		}
		processes.clear();
		log.info("Cluster stopped");
	}

	@Override
	public void close() throws Exception {
		stop();
	}

	// ── Private ───────────────────────────────────────────────────────────────

	private Process launchNode(String nodeId, int port) throws IOException {
		String javaExe = ProcessHandle.current().info().command()
				.orElse(Path.of(System.getProperty("java.home"), "bin", "java").toString());

		String classpath = System.getProperty("java.class.path");
		boolean verbose = "true".equalsIgnoreCase(System.getProperty("JUNO_VERBOSE"));

		java.util.List<String> cmd = new java.util.ArrayList<>(java.util.List.of(javaExe, "--enable-preview",
				"--enable-native-access=ALL-UNNAMED", "-Xms512m", "-Xmx4g", "-XX:+UseZGC"));

		// So forked nodes can load jnicudart (JavaCPP extracts to cache); flat cache = no subdirs.
		cmd.add("-Dorg.bytedeco.javacpp.cachedir.nosubdir=true");
		if (javacppCacheDir != null && !javacppCacheDir.isEmpty()) {
			String existingPath = System.getProperty("java.library.path", "");
			String newPath = existingPath.isEmpty() ? javacppCacheDir : (existingPath + File.pathSeparator + javacppCacheDir);
			cmd.add("-Djava.library.path=" + newPath);
		}

		String useGpu = System.getProperty("JUNO_USE_GPU", "true");
		cmd.add("-DJUNO_USE_GPU=" + useGpu);
		cmd.add("-DJUNO_VERBOSE=" + (verbose ? "true" : "false"));

		if (!verbose) {
			// Write a temp JUL config that silences all logging in this node JVM.
			java.io.File q = java.io.File.createTempFile("juno-quiet-", ".properties");
			q.deleteOnExit();
			try (java.io.PrintWriter pw = new java.io.PrintWriter(q)) {
				pw.println("handlers="); // no handlers -> nothing printed
				pw.println(".level=OFF");
			}
			cmd.add("-Djava.util.logging.config.file=" + q.getAbsolutePath());
		}

		cmd.addAll(java.util.List.of("-cp", classpath, NodeMain.class.getName(), nodeId, String.valueOf(port)));
		if (modelPath != null)
			cmd.add(modelPath);

		ProcessBuilder pb = new ProcessBuilder(cmd);
		// So jnicudart.dll can load cudart64_*.dll etc.: put CUDA bin and JavaCPP cache on PATH for the child.
		if ("true".equalsIgnoreCase(useGpu)) {
			java.util.Map<String, String> env = pb.environment();
			String pathKey = getPathKey(env);
			String path = pathKey != null ? env.get(pathKey) : null;
			if (path == null) path = System.getenv("PATH");
			if (path == null) path = System.getenv("Path");
			StringBuilder prefix = new StringBuilder();
			String cudaPath = System.getenv("CUDA_PATH");
			if (cudaPath == null) cudaPath = System.getenv("CUDA_HOME");
			if (cudaPath == null) cudaPath = findCudaPathFromPath(path);
			if (cudaPath == null) cudaPath = findCudaPathFromPath(System.getenv("PATH"));
			if (cudaPath == null) cudaPath = findCudaPathFromPath(System.getenv("Path"));
			if (cudaPath == null) cudaPath = findCudaPathFromCommonLocations();
			if (cudaPath != null) {
				File bin = new File(cudaPath, "bin");
				if (bin.isDirectory()) {
					File x64 = new File(bin, "x64");
					if (x64.isDirectory()) prefix.append(x64.getAbsolutePath()).append(File.pathSeparator);
					prefix.append(bin.getAbsolutePath()).append(File.pathSeparator);
				}
			}
			if (javacppCacheDir != null && !javacppCacheDir.isEmpty()) {
				prefix.append(javacppCacheDir).append(File.pathSeparator);
			}
			if (prefix.length() > 0) {
				String newPath = prefix + (path != null ? path : "");
				if (pathKey != null) env.put(pathKey, newPath);
				else env.put("PATH", newPath);
			}
		}
		pb.redirectErrorStream(false);
		if (verbose) {
			pb.redirectError(ProcessBuilder.Redirect.INHERIT);
		} else {
			// DISCARD stderr in quiet mode — do NOT leave it as PIPE.
			// Netty/gRPC debug output can exceed the 64 KB OS pipe buffer before
			// READY is printed, causing the child to block on stderr writes and
			// deadlock with the parent waiting on waitForReady().
			pb.redirectError(ProcessBuilder.Redirect.DISCARD);
		}
		Process proc = pb.start();
		log.info("Forked JVM for node [" + nodeId + "] PID=" + proc.pid());
		return proc;
	}

	/**
	 * Block until the process prints "READY:<nodeId>:..." or the timeout expires.
	 * On failure, drains stderr from the dead process and includes it in the
	 * exception.
	 */
	private static void waitForReady(Process proc, String expectedNodeId) throws IOException, InterruptedException {

		long deadline = System.currentTimeMillis() + NODE_STARTUP_TIMEOUT_MS;
		try (BufferedReader reader = new BufferedReader(new InputStreamReader(proc.getInputStream()))) {

			String line;
			while ((line = reader.readLine()) != null) {
				log.fine("[" + expectedNodeId + "] stdout: " + line);
				if (line.startsWith("READY:" + expectedNodeId)) {
					return;
				}
				if (System.currentTimeMillis() > deadline) {
					throw new IOException("Node [" + expectedNodeId + "] did not become ready within "
							+ NODE_STARTUP_TIMEOUT_MS + " ms");
				}
				if (!proc.isAlive()) {
					throw new IOException("Node [" + expectedNodeId + "] process died before becoming ready (exit="
							+ proc.exitValue() + ")\n" + drainStderr(proc));
				}
			}
		}

		// stdout EOF — process exited without READY. Capture stderr for diagnosis.
		String stderr = drainStderr(proc);
		throw new IOException("Node [" + expectedNodeId + "] stdout closed without READY signal"
				+ (stderr.isBlank() ? "" : "\nNode stderr:\n" + stderr));
	}

	/**
	 * Fallback: check common Windows CUDA install roots for a versioned dir with bin/.
	 */
	private static String findCudaPathFromCommonLocations() {
		String root = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA";
		File cudaRoot = new File(root);
		if (!cudaRoot.isDirectory()) return null;
		File[] versions = cudaRoot.listFiles(File::isDirectory);
		if (versions == null) return null;
		// Prefer higher version (e.g. v13.2 over v12.0); simple string sort puts v13 after v12.
		java.util.Arrays.sort(versions, (a, b) -> b.getName().compareTo(a.getName()));
		for (File v : versions) {
			if (v.getName().toLowerCase().startsWith("v")) {
				File bin = new File(v, "bin");
				if (bin.isDirectory()) return v.getAbsolutePath();
			}
		}
		return null;
	}

	/** On Windows the env map may use "Path" instead of "PATH". Return the key that exists. */
	private static String getPathKey(java.util.Map<String, String> env) {
		if (env.containsKey("PATH")) return "PATH";
		if (env.containsKey("Path")) return "Path";
		for (String key : env.keySet()) {
			if ("path".equalsIgnoreCase(key)) return key;
		}
		return null;
	}

	/**
	 * If CUDA_PATH/CUDA_HOME are not set, try to find a CUDA install from PATH
	 * (e.g. "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin" -> return parent of bin).
	 */
	private static String findCudaPathFromPath(String pathEnv) {
		if (pathEnv == null) return null;
		String sep = File.pathSeparator;
		int i = 0;
		while (i < pathEnv.length()) {
			int j = pathEnv.indexOf(sep, i);
			if (j < 0) j = pathEnv.length();
			String segment = pathEnv.substring(i, j).trim();
			i = j + 1;
			if (segment.isEmpty()) continue;
			File dir = new File(segment);
			if (!dir.isDirectory()) continue;
			String name = dir.getName().toLowerCase();
			// PATH often has .../CUDA/vX.Y/bin — we want .../CUDA/vX.Y
			if (name.equals("bin")) {
				File parent = dir.getParentFile();
				if (parent != null && parent.getName().toLowerCase().startsWith("v")
						&& parent.getParent() != null
						&& parent.getParentFile().getName().equalsIgnoreCase("cuda")) {
					return parent.getAbsolutePath();
				}
			}
			// Or path is .../CUDA/vX.Y/bin/x64 — want .../CUDA/vX.Y
			if (name.equals("x64")) {
				File bin = dir.getParentFile();
				if (bin != null && bin.getName().equalsIgnoreCase("bin")) {
					File parent = bin.getParentFile();
					if (parent != null && parent.getName().toLowerCase().startsWith("v")
							&& parent.getParentFile() != null
							&& parent.getParentFile().getName().equalsIgnoreCase("cuda")) {
						return parent.getAbsolutePath();
					}
				}
			}
		}
		return null;
	}

	/** Drain stderr from a (possibly dead) process into a string. */
	private static String drainStderr(Process proc) {
		// stderr is discarded in quiet mode (Redirect.DISCARD) to prevent pipe
		// deadlock — run with --verbose to see node stderr on startup failures.
		try {
			var errStream = proc.getErrorStream();
			if (errStream == null || errStream.available() == 0)
				return "(stderr discarded — rerun with --verbose for details)";
			byte[] bytes = errStream.readAllBytes();
			return new String(bytes, java.nio.charset.StandardCharsets.UTF_8).strip();
		} catch (Exception e) {
			return "(could not read stderr: " + e.getMessage() + ")";
		}
	}

	// ── Inner type ────────────────────────────────────────────────────────────

	private record NodeSpec(String nodeId, String host, int port, ProcessPipelineClient.ShardConfig shard) {
	}
}