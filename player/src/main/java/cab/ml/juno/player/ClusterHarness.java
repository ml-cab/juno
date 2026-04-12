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
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

import cab.ml.juno.node.EmbeddedNodeServer;
import cab.ml.juno.node.GgufReader;
import cab.ml.juno.node.InferencePipeline;
import cab.ml.juno.node.LlamaConfig;
import cab.ml.juno.node.NodeMain;
import cab.ml.juno.registry.ParallelismType;

/**
 * Manages the lifecycle of N forked node JVM processes for integration testing.
 *
 * Two parallelism strategies are supported, selected at construction time:
 *
 * PIPELINE (default — threeNodes() factories): 3 nodes, each holding a
 * contiguous layer block. Activations flow node-1 → node-2 → node-3 in serial
 * order. Backed by ProcessPipelineClient.
 *
 * TENSOR (tensorNodes() factories): 3 nodes, each holding ALL layers but a
 * different weight-tensor slice. Every decode step fans out to all nodes in
 * parallel; results are summed (star AllReduce) by the coordinator. Backed by
 * TensorParallelPipelineClient.
 *
 * Usage (pipeline mode): ClusterHarness harness = ClusterHarness.threeNodes();
 * harness.start(); InferencePipeline pipeline = harness.pipeline();
 *
 * Usage (tensor parallel mode): ClusterHarness harness =
 * ClusterHarness.tensorNodes(); harness.start(); InferencePipeline pipeline =
 * harness.pipeline();
 */
public final class ClusterHarness implements AutoCloseable {

	private static final Logger log = Logger.getLogger(ClusterHarness.class.getName());

	private static final long NODE_STARTUP_TIMEOUT_MS = 30_000;

	static final int NODE_1_PORT = 19092;
	static final int NODE_2_PORT = 19093;
	static final int NODE_3_PORT = 19094;

	private static final int VOCAB_SIZE = EmbeddedNodeServer.VOCAB_SIZE;
	private static final int TOTAL_LAYERS = EmbeddedNodeServer.TOTAL_LAYERS;
	private static final int NUM_HEADS = EmbeddedNodeServer.NUM_HEADS;

	private final List<NodeSpec> specs;
	private final List<Process> processes = new ArrayList<>();
	private final ParallelismType parallelismType;
	// Tensor-parallel model shape — set by tensorNodes() factory, used in
	// startTensorParallel().
	// totalLayers and numHeads come from the factory args; vocabSize is read from
	// GGUF at start().
	private final int configuredTotalLayers;

	private ProcessPipelineClient pipelineClient;
	private TensorParallelPipelineClient tensorPipelineClient;

	private static String modelPath;

	/** Non-null when the caller wants JFR on every node JVM. */
	private String jfrDuration;
	private String jfrTimestamp;

	private ClusterHarness(List<NodeSpec> specs) {
		this(specs, null, ParallelismType.PIPELINE, TOTAL_LAYERS, NUM_HEADS);
	}

	private ClusterHarness(List<NodeSpec> specs, String pathToModel, ParallelismType pType) {
		this(specs, pathToModel, pType, TOTAL_LAYERS, NUM_HEADS);
	}

	private ClusterHarness(List<NodeSpec> specs, String pathToModel, ParallelismType pType, int totalLayers,
			int numHeads) {
		this.specs = specs;
		this.parallelismType = pType;
		this.configuredTotalLayers = totalLayers;
		modelPath = pathToModel;
	}

	// ── Pipeline-parallel factories ───────────────────────────────────────────

	public static ClusterHarness threeNodes() {
		return threeNodes(null);
	}

	public static ClusterHarness threeNodes(String modelPath) {
		return threeNodes(modelPath, TOTAL_LAYERS);
	}

	/**
	 * Create a 3-node cluster for a model with {@code totalLayers} transformer
	 * layers. Shard boundaries are computed by dividing layers as evenly as
	 * possible across the three nodes (ceiling-first distribution).
	 *
	 * <pre>
	 *  22 layers -&gt; [0,8)  [8,15)  [15,22)   (TinyLlama-1.1B)
	 *  16 layers -&gt; [0,6)  [6,11)  [11,16)   (Llama-3.2-1B)
	 * </pre>
	 *
	 * @param modelPath   path to a GGUF/llamafile, or null for stub mode
	 * @param totalLayers actual number of transformer layers in the model
	 */
	public static ClusterHarness threeNodes(String modelPath, int totalLayers) {
		if (totalLayers < 3)
			throw new IllegalArgumentException("totalLayers must be >= 3 to split across 3 nodes, got: " + totalLayers);

		int base = totalLayers / 3;
		int extra = totalLayers % 3;
		int end1 = base + (extra > 0 ? 1 : 0);
		int end2 = end1 + base + (extra > 1 ? 1 : 0);

		return new ClusterHarness(
				List.of(new NodeSpec("node-1", "localhost", NODE_1_PORT,
						new ProcessPipelineClient.ShardConfig(0, end1, true, false)),
						new NodeSpec("node-2", "localhost", NODE_2_PORT,
								new ProcessPipelineClient.ShardConfig(end1, end2, false, false)),
						new NodeSpec("node-3", "localhost", NODE_3_PORT,
								new ProcessPipelineClient.ShardConfig(end2, totalLayers, false, true))),
				modelPath, ParallelismType.PIPELINE);
	}

	// ── Tensor-parallel factories (new) ───────────────────────────────────────

	/**
	 * Create a 3-node tensor-parallel cluster (stub mode, no model).
	 *
	 * All 3 nodes receive the full layer range [0, totalLayers) and tensor ranks 0,
	 * 1, 2. NUM_HEADS (32) is even — uneven head distribution across 3 nodes is
	 * fine (10 / 11 / 11); only an odd total head count is rejected.
	 */
	public static ClusterHarness tensorNodes() {
		return tensorNodes(null);
	}

	/**
	 * Create a 3-node tensor-parallel cluster that loads real weights.
	 *
	 * @param modelPath path to a GGUF file, or null for stub mode
	 */
	public static ClusterHarness tensorNodes(String modelPath) {
		return tensorNodes(modelPath, TOTAL_LAYERS, NUM_HEADS);
	}

	/**
	 * Create a 3-node tensor-parallel cluster for a specific model shape.
	 *
	 * Constraint: numHeads must be even (divisible by 2). Uneven distribution
	 * across the 3 nodes is handled by ceiling-division in TensorShardContext. All
	 * 3 nodes receive startLayer=0, endLayer=totalLayers, hasEmbeddings=true,
	 * hasOutputProjection=true. The node uses its tensorRank to slice weights.
	 *
	 * @param modelPath   path to a GGUF/llamafile, or null for stub mode
	 * @param totalLayers transformer layer count
	 * @param numHeads    attention head count — must be even (divisible by 2)
	 */
	public static ClusterHarness tensorNodes(String modelPath, int totalLayers, int numHeads) {
		if (numHeads % 2 != 0)
			throw new IllegalArgumentException(
					"numHeads (" + numHeads + ") must be even (divisible by 2) " + "for a tensor-parallel cluster");

		return new ClusterHarness(
				List.of(new NodeSpec("node-1", "localhost", NODE_1_PORT,
						new ProcessPipelineClient.ShardConfig(0, totalLayers, true, true)),
						new NodeSpec("node-2", "localhost", NODE_2_PORT,
								new ProcessPipelineClient.ShardConfig(0, totalLayers, true, true)),
						new NodeSpec("node-3", "localhost", NODE_3_PORT,
								new ProcessPipelineClient.ShardConfig(0, totalLayers, true, true))),
				modelPath, ParallelismType.TENSOR, totalLayers, numHeads);
	}

	// ── Lifecycle ─────────────────────────────────────────────────────────────

	/**
	 * Fork all three node JVMs and wait until each reports READY.
	 */
	public void start() throws IOException, InterruptedException {
		for (NodeSpec spec : specs) {
			Process proc = launchNode(spec.nodeId(), spec.port());
			processes.add(proc);
			waitForReady(proc, spec.nodeId());
			log.info("Node [" + spec.nodeId() + "] is up on port " + spec.port());
		}

		if (parallelismType == ParallelismType.TENSOR) {
			startTensorParallel();
		} else {
			startPipelineParallel();
		}
	}

	private void startPipelineParallel() {
		List<ProcessPipelineClient.NodeAddress> addresses = specs.stream()
				.map(s -> new ProcessPipelineClient.NodeAddress(s.host(), s.port())).toList();
		pipelineClient = new ProcessPipelineClient(addresses, VOCAB_SIZE);

		List<ProcessPipelineClient.ShardConfig> shards = specs.stream().map(NodeSpec::pipelineShard).toList();
		pipelineClient.loadShards(shards);

		log.info("Pipeline-parallel cluster ready — 3 nodes, " + TOTAL_LAYERS + " total layers distributed");
	}

	private void startTensorParallel() {
		int worldSize = specs.size();

		// Resolve actual vocabSize from the GGUF file when a real model is being used.
		// The class-level VOCAB_SIZE constant is TinyLlama-specific (32 000) and wrong
		// for other models (e.g. phi-3.5-mini has vocabSize=32 064).
		int vocabSize = VOCAB_SIZE; // fallback for stub mode
		if (modelPath != null) {
			try (GgufReader r = GgufReader.open(java.nio.file.Path.of(modelPath))) {
				vocabSize = LlamaConfig.from(r).vocabSize();
			} catch (java.io.IOException e) {
				log.warning("Could not read vocabSize from GGUF — falling back to " + VOCAB_SIZE + ". Cause: "
						+ e.getMessage());
			}
		}

		List<TensorParallelPipelineClient.NodeAddress> addresses = specs.stream()
				.map(s -> new TensorParallelPipelineClient.NodeAddress(s.host(), s.port())).toList();
		tensorPipelineClient = new TensorParallelPipelineClient(addresses, vocabSize);

		List<TensorParallelPipelineClient.TensorShardConfig> shards = new ArrayList<>();
		for (int rank = 0; rank < worldSize; rank++) {
			shards.add(new TensorParallelPipelineClient.TensorShardConfig(0, configuredTotalLayers, true, // hasEmbeddings
																											// — all
																											// tensor-parallel
																											// nodes
																											// embed
																											// independently
					true, // hasOutputProjection — all nodes produce partial logits
					rank, worldSize));
		}
		tensorPipelineClient.loadShards(shards);

		log.info("Tensor-parallel cluster ready — " + worldSize + " nodes, worldSize=" + worldSize + ", layers="
				+ configuredTotalLayers + ", vocabSize=" + vocabSize);
	}

	// ── Accessors ─────────────────────────────────────────────────────────────

	/**
	 * Returns the InferencePipeline for this cluster, regardless of parallelism
	 * type. Works for both PIPELINE and TENSOR modes. Only valid after start() has
	 * been called.
	 */
	public InferencePipeline pipeline() {
		if (parallelismType == ParallelismType.TENSOR) {
			if (tensorPipelineClient == null)
				throw new IllegalStateException("ClusterHarness not started — call start() first");
			return tensorPipelineClient;
		}
		return pipelineClient();
	}

	/**
	 * Returns the ProcessPipelineClient for PIPELINE mode. Only valid after start()
	 * on a threeNodes() harness. For mode-agnostic code, prefer
	 * {@link #pipeline()}.
	 */
	public ProcessPipelineClient pipelineClient() {
		if (parallelismType == ParallelismType.TENSOR)
			throw new IllegalStateException("pipelineClient() is only valid for PIPELINE mode. Use pipeline().");
		if (pipelineClient == null)
			throw new IllegalStateException("ClusterHarness not started — call start() first");
		return pipelineClient;
	}

	/**
	 * Returns the TensorParallelPipelineClient for TENSOR mode. Only valid after
	 * start() on a tensorNodes() harness.
	 */
	public TensorParallelPipelineClient tensorPipelineClient() {
		if (parallelismType != ParallelismType.TENSOR)
			throw new IllegalStateException("tensorPipelineClient() is only valid for TENSOR mode. Use pipeline().");
		if (tensorPipelineClient == null)
			throw new IllegalStateException("ClusterHarness not started — call start() first");
		return tensorPipelineClient;
	}

	/**
	 * Returns the node addresses in pipeline order. Use this to create additional
	 * ProcessPipelineClient instances.
	 */
	public List<ProcessPipelineClient.NodeAddress> nodeAddresses() {
		return specs.stream().map(s -> new ProcessPipelineClient.NodeAddress(s.host(), s.port())).toList();
	}

	/** Returns the parallelism type this cluster was created with. */
	public ParallelismType parallelismType() {
		return parallelismType;
	}

	/**
	 * Enables JFR on every forked node JVM.
	 *
	 * <p>Must be called before {@link #start()}. Each node is launched with
	 * {@code -XX:StartFlightRecording=duration=<duration>,filename=juno-<nodeId>-<stem>-<timestamp>.jfr,
	 * settings=profile,dumponexit=true} so its events are written when the process exits.
	 *
	 * @param duration  human-friendly duration string, e.g. {@code "2m"} or {@code "30s"}
	 * @param timestamp shared timestamp string (yyyyMMdd-HHmmss) — keeps coordinator and node
	 *                  filenames aligned for easy correlation
	 */
	public ClusterHarness withJfr(String duration, String timestamp) {
		this.jfrDuration = duration;
		this.jfrTimestamp = timestamp;
		return this;
	}

	/**
	 * Returns the expected JFR output paths for every node, in node order.
	 * Only meaningful after {@link #withJfr(String, String)} has been called.
	 */
	public List<Path> nodeJfrFiles() {
		if (jfrDuration == null)
			return List.of();
		String stem = modelPath != null ? stemOf(modelPath) : "model";
		return specs.stream()
				.map(s -> Path.of("juno-" + s.nodeId() + "-" + stem + "-" + jfrTimestamp + ".jfr"))
				.toList();
	}

	private static String stemOf(String path) {
		String name = Path.of(path).getFileName().toString();
		int dot = name.lastIndexOf('.');
		return dot > 0 ? name.substring(0, dot) : name;
	}

	// ── Teardown ──────────────────────────────────────────────────────────────

	public void stop() throws InterruptedException {
		if (pipelineClient != null)
			pipelineClient.shutdown();
		if (tensorPipelineClient != null)
			tensorPipelineClient.shutdown();
		for (Process proc : processes) {
			proc.destroy();
			proc.waitFor(5, TimeUnit.SECONDS);
		}
		processes.clear();
		log.info("Cluster stopped (" + parallelismType + " mode)");
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
		// Inherit heap from coordinator via -Djuno.node.heap=<size>; default 4g.
		// Large models (phi-3.5-mini, Llama-7B) need >=6g on node-0 which eagerly
		// dequantises token_embd.weight to float[vocabSize * hiddenDim].
		String nodeHeap = System.getProperty("juno.node.heap", "4g");

		java.util.List<String> cmd = new java.util.ArrayList<>(java.util.List.of(javaExe, "--enable-preview",
				"--enable-native-access=ALL-UNNAMED", "-Xms512m", "-Xmx" + nodeHeap, "-XX:+UseZGC"));

		// Partition cores across node JVMs so they do not fight over ForkJoinPool.
		// Without this, each JVM grabs availableProcessors()-1 threads, causing
		// N_nodes × (cores-1) threads to contend for the same physical cores.
		int coresPerNode = Math.max(1, Runtime.getRuntime().availableProcessors() / specs.size());
		cmd.add("-Djava.util.concurrent.ForkJoinPool.common.parallelism=" + coresPerNode);

		// JFR on the node JVM — records MatVec / ForwardPass events that fire here.
		if (jfrDuration != null && jfrTimestamp != null) {
			String stem = modelPath != null ? stemOf(modelPath) : "model";
			String nodeJfrFile = "juno-" + nodeId + "-" + stem + "-" + jfrTimestamp + ".jfr";
			cmd.add("-XX:StartFlightRecording=duration=" + jfrDuration
					+ ",filename=" + nodeJfrFile
					+ ",settings=profile,dumponexit=true");
		}

		if (!verbose) {
			java.io.File q = java.io.File.createTempFile("juno-quiet-", ".properties");
			q.deleteOnExit();
			try (java.io.PrintWriter pw = new java.io.PrintWriter(q)) {
				pw.println("handlers=");
				pw.println(".level=OFF");
			}
			cmd.add("-Djava.util.logging.config.file=" + q.getAbsolutePath());
		}

		cmd.addAll(java.util.List.of("-cp", classpath, NodeMain.class.getName(), nodeId, String.valueOf(port)));
		if (modelPath != null)
			cmd.add(modelPath);

		ProcessBuilder pb = new ProcessBuilder(cmd);
		pb.redirectErrorStream(false);
		if (verbose) {
			pb.redirectError(ProcessBuilder.Redirect.INHERIT);
		} else {
			pb.redirectError(ProcessBuilder.Redirect.DISCARD);
		}
		Process proc = pb.start();
		log.info("Forked JVM for node [" + nodeId + "] PID=" + proc.pid());
		return proc;
	}

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

		String stderr = drainStderr(proc);
		throw new IOException("Node [" + expectedNodeId + "] stdout closed without READY signal"
				+ (stderr.isBlank() ? "" : "\nNode stderr:\n" + stderr));
	}

	private static String drainStderr(Process proc) {
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

	private record NodeSpec(String nodeId, String host, int port, ProcessPipelineClient.ShardConfig pipelineShard) {
	}
}