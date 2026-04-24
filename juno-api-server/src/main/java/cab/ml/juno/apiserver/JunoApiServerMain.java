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
package cab.ml.juno.apiserver;

import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.logging.Logger;

import cab.ml.juno.coordinator.BatchConfig;
import cab.ml.juno.coordinator.GenerationLoop;
import cab.ml.juno.coordinator.ProcessPipelineClient;
import cab.ml.juno.coordinator.RequestScheduler;
import cab.ml.juno.coordinator.TensorParallelPipelineClient;
import cab.ml.juno.health.HealthMain;
import cab.ml.juno.health.HealthThresholds;
import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.node.ActivationDtype;
import cab.ml.juno.node.GgufReader;
import cab.ml.juno.node.LlamaConfig;
import cab.ml.juno.registry.ModelDescriptor;
import cab.ml.juno.registry.ModelRegistry;
import cab.ml.juno.registry.ModelStatus;
import cab.ml.juno.registry.QuantizationType;
import cab.ml.juno.registry.ShardPlanner;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.tokenizer.GgufTokenizer;
import cab.ml.juno.tokenizer.Tokenizer;

/**
 * Standalone entry point: distributed coordinator wiring plus HTTP API on {@link InferenceApiServer}.
 *
 * <p>For production systemd deployments that keep the legacy main class name
 * {@code cab.ml.juno.master.CoordinatorMain}, use the juno-master jar — that class delegates here.
 *
 * <p>Environment variables match the former in-module documentation for the coordinator entry point.
 */
public final class JunoApiServerMain {

	private static final Logger log = Logger.getLogger(JunoApiServerMain.class.getName());

	private static final long GPU_KV_BYTES = 512L * 1024 * 1024;
	private static final int CPU_KV_BLOCKS = 8192;
	private static final int MAX_QUEUE_DEFAULT = 1000;
	private static final int GRPC_PORT_DEFAULT = 19092;

	public static void main(String[] args) throws Exception {
		String rawAddresses = env("JUNO_NODE_ADDRESSES", "");
		String modelPath = env("JUNO_MODEL_PATH", "");
		String ptypeStr = env("JUNO_PTYPE", "pipeline");
		int httpPort = parseInt(env("JUNO_HTTP_PORT", "8080"), 8080);
		String dtypeStr = env("JUNO_DTYPE", "FLOAT16");
		String byteOrderStr = env("JUNO_BYTE_ORDER", "BE");
		System.setProperty("juno.byteOrder", "LE".equalsIgnoreCase(byteOrderStr.strip()) ? "LE" : "BE");
		int maxQueue = parseInt(env("JUNO_MAX_QUEUE", "1000"), MAX_QUEUE_DEFAULT);

		if (rawAddresses.isBlank()) {
			die("JUNO_NODE_ADDRESSES is not set. Expected: host:port,host:port,...");
		}
		if (modelPath.isBlank()) {
			die("JUNO_MODEL_PATH is not set.");
		}
		if (!Path.of(modelPath).toFile().exists()) {
			die("Model file not found: " + modelPath);
		}

		boolean tensorMode = "tensor".equalsIgnoreCase(ptypeStr.strip());
		ActivationDtype dtype = parseDtype(dtypeStr);

		if ("true".equalsIgnoreCase(env("JUNO_HEALTH", "false"))) {
			int healthPort = parseInt(env("JUNO_HEALTH_PORT", "8081"), 8081);
			long healthStaleMs = parseLong(env("JUNO_HEALTH_STALE_MS", "15000"), 15_000L);
			double healthWarn = parseDouble(env("JUNO_HEALTH_WARN", "0.90"), 0.90);
			double healthCrit = parseDouble(env("JUNO_HEALTH_CRITICAL", "0.98"), 0.98);
			HealthMain.startBackground(healthPort,
					new HealthThresholds(healthWarn, healthCrit, healthStaleMs));
			log.info("Health sidecar started on :" + healthPort);
		}

		List<ProcessPipelineClient.NodeAddress> nodeAddrs = parseAddresses(rawAddresses);
		int nodeCount = nodeAddrs.size();
		log.info("Coordinator starting — " + nodeCount + " nodes, pType=" + ptypeStr
				+ ", dtype=" + dtype + ", port=" + httpPort);

		LlamaConfig config;
		Tokenizer tokenizer;
		try (GgufReader reader = GgufReader.open(Path.of(modelPath))) {
			config = LlamaConfig.from(reader);
			tokenizer = GgufTokenizer.load(reader);
		}
		log.info("Model config: " + config);

		cab.ml.juno.node.InferencePipeline pipeline;

		if (tensorMode) {
			pipeline = buildTensorPipeline(nodeAddrs, config, dtype);
		} else {
			pipeline = buildPipelinePipeline(nodeAddrs, config, dtype);
		}

		var kvCache = new KVCacheManager(new GpuKVCache(GPU_KV_BYTES), new CpuKVCache(CPU_KV_BLOCKS));
		var loop = new GenerationLoop(tokenizer, Sampler.create(), pipeline, kvCache);
		var scheduler = new RequestScheduler(maxQueue, loop, BatchConfig.disabled());

		ModelRegistry registry = buildRegistry(config, modelPath);

		InferenceApiServer apiServer = new InferenceApiServer(scheduler, registry, byteOrderStr);

		String healthUrl = env("JUNO_HEALTH", "false").equals("true")
				? "http://localhost:" + parseInt(env("JUNO_HEALTH_PORT", "8081"), 8081)
				: null;
		if (healthUrl != null) {
			apiServer.setHealthSidecarUrl(healthUrl);
			log.info("Health dashboard wired: GET /health-ui → sidecar at " + healthUrl);

			cab.ml.juno.health.HealthReporter coordReporter =
					new cab.ml.juno.health.HealthReporter("coordinator", "coordinator", healthUrl,
							cab.ml.juno.health.HealthReporter.DEFAULT_INTERVAL_MS);
			coordReporter.startBackground();
			apiServer.setLatencyReporter(coordReporter);
			Runtime.getRuntime().addShutdownHook(
					Thread.ofVirtual().unstarted(coordReporter::stop));
		}

		apiServer.start(httpPort);
		log.info("Coordinator REST server started on port " + httpPort);
		System.out.println("COORDINATOR_READY:" + httpPort);
		System.out.flush();

		CountDownLatch latch = new CountDownLatch(1);
		Runtime.getRuntime().addShutdownHook(Thread.ofVirtual().unstarted(() -> {
			log.info("Shutdown signal — draining scheduler...");
			scheduler.shutdown();
			try {
				shutdownPipeline(pipeline);
			} catch (Exception ignored) {
			}
			latch.countDown();
		}));

		log.info("Coordinator is up. Waiting for requests on http://0.0.0.0:" + httpPort);
		latch.await();
	}

	private static cab.ml.juno.node.InferencePipeline buildPipelinePipeline(
			List<ProcessPipelineClient.NodeAddress> nodes,
			LlamaConfig config,
			ActivationDtype dtype) {

		List<ProcessPipelineClient.ShardConfig> shards = splitLayersEvenly(
				config.numLayers(), nodes.size());

		ProcessPipelineClient client = new ProcessPipelineClient(nodes, config.vocabSize(), dtype);
		log.info("Loading shards across " + nodes.size() + " pipeline nodes...");
		client.loadShards(shards);
		log.info("All pipeline shards loaded.");
		return client;
	}

	private static cab.ml.juno.node.InferencePipeline buildTensorPipeline(
			List<ProcessPipelineClient.NodeAddress> nodes,
			LlamaConfig config,
			ActivationDtype dtype) {

		int worldSize = nodes.size();
		List<TensorParallelPipelineClient.NodeAddress> tensorAddrs = nodes.stream()
				.map(a -> new TensorParallelPipelineClient.NodeAddress(a.host(), a.port()))
				.toList();

		TensorParallelPipelineClient client =
				new TensorParallelPipelineClient(tensorAddrs, config.vocabSize());

		List<TensorParallelPipelineClient.TensorShardConfig> shards = new ArrayList<>();
		for (int rank = 0; rank < worldSize; rank++) {
			shards.add(new TensorParallelPipelineClient.TensorShardConfig(
					0, config.numLayers(), true, true, rank, worldSize));
		}

		log.info("Loading shards across " + worldSize + " tensor-parallel nodes...");
		client.loadShards(shards);
		log.info("All tensor shards loaded.");
		return client;
	}

	static List<ProcessPipelineClient.ShardConfig> splitLayersEvenly(int totalLayers, int nodeCount) {
		if (nodeCount < 1)
			throw new IllegalArgumentException("nodeCount must be >= 1");
		if (totalLayers < nodeCount)
			throw new IllegalArgumentException(
					"totalLayers (" + totalLayers + ") must be >= nodeCount (" + nodeCount + ")");

		int base = totalLayers / nodeCount;
		int extra = totalLayers % nodeCount;

		List<ProcessPipelineClient.ShardConfig> shards = new ArrayList<>(nodeCount);
		int start = 0;
		for (int i = 0; i < nodeCount; i++) {
			int layersForNode = base + (i < extra ? 1 : 0);
			int end = start + layersForNode;

			boolean hasEmbed = (i == 0);
			boolean hasOutput = (i == nodeCount - 1);

			shards.add(new ProcessPipelineClient.ShardConfig(start, end, hasEmbed, hasOutput));
			log.info("  node-" + (i + 1) + ": layers [" + start + ", " + end + ")"
					+ (hasEmbed ? " +embed" : "")
					+ (hasOutput ? " +output" : ""));
			start = end;
		}
		return shards;
	}

	private static ModelRegistry buildRegistry(LlamaConfig config, String modelPath) {
		ModelRegistry registry = new ModelRegistry(ShardPlanner.create());

		long vramPerLayer = 4L * config.hiddenDim() * config.hiddenDim() * 2;

		String filename = Path.of(modelPath).getFileName().toString();
		String modelId = filename;

		QuantizationType quant;
		try (GgufReader r = GgufReader.open(Path.of(modelPath))) {
			quant = LlamaConfig.detectQuantization(r, filename);
		} catch (java.io.IOException e) {
			log.warning("Could not re-open GGUF for quant detection: " + e.getMessage());
			quant = LlamaConfig.fromFilename(filename);
		}

		ModelDescriptor descriptor = new ModelDescriptor(
				modelId,
				config.architecture(),
				config.numLayers(),
				config.hiddenDim(),
				config.vocabSize(),
				config.numHeads(),
				vramPerLayer,
				quant,
				modelPath,
				ModelStatus.LOADED,
				Instant.now());

		registry.putLoaded(descriptor);
		return registry;
	}

	private static List<ProcessPipelineClient.NodeAddress> parseAddresses(String raw) {
		List<ProcessPipelineClient.NodeAddress> result = new ArrayList<>();
		for (String part : raw.split(",")) {
			String trimmed = part.strip();
			if (trimmed.isEmpty())
				continue;
			int colon = trimmed.lastIndexOf(':');
			if (colon < 0) {
				result.add(new ProcessPipelineClient.NodeAddress(trimmed, GRPC_PORT_DEFAULT));
			} else {
				String host = trimmed.substring(0, colon);
				int port = parseInt(trimmed.substring(colon + 1), GRPC_PORT_DEFAULT);
				result.add(new ProcessPipelineClient.NodeAddress(host, port));
			}
		}
		if (result.isEmpty())
			die("No valid node addresses parsed from JUNO_NODE_ADDRESSES: \"" + raw + "\"");
		return result;
	}

	private static ActivationDtype parseDtype(String s) {
		return switch (s.toUpperCase().strip()) {
			case "FLOAT32", "F32" -> ActivationDtype.FLOAT32;
			case "INT8", "I8" -> ActivationDtype.INT8;
			default -> ActivationDtype.FLOAT16;
		};
	}

	private static String env(String key, String defaultValue) {
		String v = System.getProperty(key);
		if (v == null || v.isBlank())
			v = System.getenv(key);
		return (v != null && !v.isBlank()) ? v : defaultValue;
	}

	private static int parseInt(String s, int def) {
		try {
			return Integer.parseInt(s.strip());
		} catch (NumberFormatException e) {
			return def;
		}
	}

	private static long parseLong(String s, long def) {
		try {
			return Long.parseLong(s.strip());
		} catch (NumberFormatException e) {
			return def;
		}
	}

	private static double parseDouble(String s, double def) {
		try {
			return Double.parseDouble(s.strip());
		} catch (NumberFormatException e) {
			return def;
		}
	}

	private static void shutdownPipeline(cab.ml.juno.node.InferencePipeline pipeline) throws Exception {
		if (pipeline instanceof ProcessPipelineClient c)
			c.shutdown();
		else if (pipeline instanceof TensorParallelPipelineClient c)
			c.shutdown();
	}

	private static void die(String msg) {
		System.err.println("[JunoApiServerMain] FATAL: " + msg);
		System.exit(1);
	}
}
