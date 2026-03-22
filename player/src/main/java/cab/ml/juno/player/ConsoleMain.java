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
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

import cab.ml.juno.coordinator.GenerationLoop;
import cab.ml.juno.coordinator.GenerationResult;
import cab.ml.juno.coordinator.InferenceRequest;
import cab.ml.juno.coordinator.RequestPriority;
import cab.ml.juno.coordinator.TokenConsumer;
import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.node.ActivationDtype;
import cab.ml.juno.node.ForwardPassHandler;
import cab.ml.juno.node.ForwardPassHandlerLoader;
import cab.ml.juno.node.GgufReader;
import cab.ml.juno.node.LlamaConfig;
import cab.ml.juno.node.LocalInferencePipeline;
import cab.ml.juno.registry.NodeDescriptor;
import cab.ml.juno.registry.NodeStatus;
import cab.ml.juno.registry.ParallelismType;
import cab.ml.juno.registry.ShardMap;
import cab.ml.juno.registry.ShardPlanner;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.GgufTokenizer;
import cab.ml.juno.tokenizer.Tokenizer;

/**
 * Interactive REPL that runs a model using the Juno engine.
 *
 * Can operate in two modes: - cluster mode (default): forks 3 node JVMs (as
 * before) - local mode (--local): runs all nodes in‑process, no child JVMs
 *
 * Command‑line arguments (overrides environment variables / system properties):
 * --model-path PATH Path to GGUF file (required) --dtype FLOAT32|FLOAT16|INT8
 * Activation wire format (default: FLOAT16) --max-tokens N Max generated tokens
 * (default: 200) --temperature F Sampling temperature (default: 0.7) --heap
 * SIZE JVM heap size hint (ignored when run as jar) --local Use in‑process
 * nodes instead of forking --nodes N Number of in‑process nodes (default: 3,
 * only with --local) --verbose Show full gRPC + Maven logs (ignored in local
 * mode) --help Show this help
 *
 * Example: java --enable-preview --enable-native-access=ALL-UNNAMED \
 * --add-opens java.base/java.lang=ALL-UNNAMED \ --add-opens
 * java.base/java.nio=ALL-UNNAMED \ -jar player.jar --model-path
 * /models/tinyllama.gguf --local
 */
public final class ConsoleMain {

	@SuppressWarnings("unused")
	private static final Logger log = Logger.getLogger(ConsoleMain.class.getName());

	// Silence logging unless verbose (same as original)
	static {
		boolean verbose = Boolean.getBoolean("JUNO_VERBOSE") || "true".equalsIgnoreCase(System.getenv("JUNO_VERBOSE"));
		if (!verbose) {
			java.util.logging.LogManager.getLogManager().reset();
			java.util.logging.Logger.getLogger("").setLevel(java.util.logging.Level.OFF);
			for (String ns : new String[] { "io.grpc", "io.netty", "cab.ml.juno", "com.google", "org.slf4j", "" }) {
				java.util.logging.Logger.getLogger(ns).setLevel(java.util.logging.Level.OFF);
			}
		}
	}

	// Argument holders
	private static String modelPath = null;
	private static ActivationDtype dtype = ActivationDtype.FLOAT16;
	private static int maxTokens = 200;
	// Default sampling params for console/cluster runs
	// temperature=0.6, topK=20, topP=0.95 as requested
	private static float temperature = 0.6f;
	private static int topK = 20;
	private static float topP = 0.95f;
	private static boolean localMode = false;
	private static int nodeCount = 3;
	private static boolean verbose = false;
	private static boolean help = false;
	private static ParallelismType pType = ParallelismType.PIPELINE;
	private static String jfrDuration = null;

	public static void main(String[] args) throws Exception {
		AnsiSupport.enable();
		parseArgs(args);
		if (help) {
			printHelp();
			System.exit(0);
		}

		if (modelPath == null) {
			System.err.println("ERROR: --model-path is required");
			printHelp();
			System.exit(1);
		}

		if (!Path.of(modelPath).toFile().exists()) {
			System.err.println("ERROR: Model file not found: " + modelPath);
			System.exit(1);
		}

		// Set system properties for legacy code (ClusterHarness reads these)
		System.setProperty("MODEL_PATH", modelPath);
		System.setProperty("DTYPE", dtype.name());
		System.setProperty("MAX_TOKENS", String.valueOf(maxTokens));
		System.setProperty("TEMPERATURE", String.valueOf(temperature));
		System.setProperty("TOP_K", String.valueOf(topK));
		System.setProperty("TOP_P", String.valueOf(topP));
		if (verbose) {
			System.setProperty("JUNO_VERBOSE", "true");
		}
		if (jfrDuration != null) {
			System.setProperty("juno.jfr.duration", jfrDuration);
		}

		// Show banner
		banner();

		if (localMode) {
			runLocalRepl();
		} else {
			runClusterRepl();
		}
	}

	private static void parseArgs(String[] args) {
		for (int i = 0; i < args.length; i++) {
			switch (args[i]) {
			case "--model-path":
				if (i + 1 < args.length)
					modelPath = args[++i];
				break;
			case "--dtype":
				if (i + 1 < args.length)
					dtype = parseDtype(args[++i]);
				break;
			case "--max-tokens":
				if (i + 1 < args.length)
					maxTokens = parseInt(args[++i], 200);
				break;
			case "--top-k":
				if (i + 1 < args.length)
					topK = parseInt(args[++i], 20);
				break;
			case "--top-p":
				if (i + 1 < args.length)
					topP = parseFloat(args[++i], 0.95f);
				break;
			case "--temperature":
				if (i + 1 < args.length)
					temperature = parseFloat(args[++i], 0.6f);
				break;
			case "--pType":
			case "--ptype":
				if (i + 1 < args.length)
					pType = parseParallelismType(args[++i]);
				break;
			case "--heap":
				// consumed by run.sh as -Xmx; ignored here but must be consumed
				if (i + 1 < args.length)
					i++;
				break;
			case "--jfr":
				// consumed by run.sh/run.bat as -XX:StartFlightRecording; recognised here
				// so that direct `java -jar player.jar --jfr 5m` invocations don't fail,
				// and so juno.jfr.duration is available as a system property at runtime.
				if (i + 1 < args.length)
					jfrDuration = args[++i];
				break;
			case "--local":
				localMode = true;
				break;
			case "--nodes":
				if (i + 1 < args.length)
					nodeCount = parseInt(args[++i], 3);
				break;
			case "--verbose":
			case "-v":
				verbose = true;
				break;
			case "--help":
			case "-h":
				help = true;
				return;
			default:
				System.err.println("Unknown option: " + args[i]);
				help = true;
				return;
			}
		}
	}

	private static void printHelp() {
		System.out.println();
		System.out.println("Usage: java -jar player.jar [options]");
		System.out.println();
		System.out.println("Required:");
		System.out.println("  --model-path PATH          Path to GGUF model file");
		System.out.println();
		System.out.println("Options:");
		System.out.println("  --pType pipeline|tensor    Parallelism type (default: pipeline)");
		System.out.println("    pipeline: contiguous layer blocks, serial activation flow");
		System.out.println("    tensor:   all layers on every node, weight slices, AllReduce");
		System.out.println("  --dtype FLOAT32|FLOAT16|INT8   Activation wire format (default: FLOAT16)");
		System.out.println("  --max-tokens N             Max generated tokens (default: 200)");
		System.out.println("  --temperature F            Sampling temperature (default: 0.6)");
		System.out.println("  --top-k N                  Top-K sampling cutoff (default: 20, 0 = disabled)");
		System.out.println("  --top-p F                  Nucleus sampling top-p (default: 0.95, 0 = disabled)");
		System.out.println("  --local                    Use in-process nodes (no forking)");
		System.out.println("  --nodes N                  Number of in-process nodes (default: 3, only with --local)");
		System.out.println("  --jfr DURATION             Java Flight Recording duration, e.g. 5m 30s 1h");
		System.out.println("                             (handled by run.sh/run.bat as -XX:StartFlightRecording;");
		System.out.println("                             also accepted here for direct jar invocations)");
		System.out.println("  --verbose, -v              Show more logging");
		System.out.println("  --help, -h                 Show this help");
		System.out.println();
		System.out.println("JVM flags required:");
		System.out.println("  --enable-preview");
		System.out.println("  --enable-native-access=ALL-UNNAMED");
		System.out.println("  --add-opens java.base/java.lang=ALL-UNNAMED");
		System.out.println("  --add-opens java.base/java.nio=ALL-UNNAMED");
	}

	// -------------------------------------------------------------------------
	// Local mode (single JVM, no child processes)
	// -------------------------------------------------------------------------

	private static void runLocalRepl() throws Exception {
		print(Color.CYAN + "▶ Starting local in‑process " + nodeCount + "-node pipeline..." + Color.RESET);

		// Read model config and tokenizer from GGUF
		LlamaConfig config;
		Tokenizer tokenizer;
		try (GgufReader reader = GgufReader.open(Path.of(modelPath))) {
			config = LlamaConfig.from(reader);
			tokenizer = GgufTokenizer.load(reader);
		}

		// Create dummy node descriptors for ShardPlanner (one per in‑process node)
		// Each node needs enough VRAM to hold all layers; we allocate a generous
		// amount.
		long vramPerLayerBytes = estimateVramPerLayer(config.hiddenDim()); // rough estimate
		long nodeVramBytes = config.numLayers() * vramPerLayerBytes * 2; // plenty

		List<NodeDescriptor> nodes = new ArrayList<>();
		for (int i = 0; i < nodeCount; i++) {
			nodes.add(new NodeDescriptor("node-" + i, "localhost", 9092 + i, // dummy port, not used
					nodeVramBytes, nodeVramBytes, NodeStatus.READY, 1.0, Instant.now(), Instant.now()));
		}

		// Compute shard map
		ShardMap shardMap = ShardPlanner.create().plan("model", config.numLayers(), vramPerLayerBytes, nodes);

		// Load one handler per shard — ForwardPassHandlerLoader selects the
		// correct implementation based on general.architecture in the GGUF file.
		List<ForwardPassHandler> handlers = new ArrayList<>();
		for (var assignment : shardMap.assignments()) {
			var context = cab.ml.juno.node.ShardContext.from(assignment, config.vocabSize(), config.hiddenDim(),
					config.numHeads());
			handlers.add(ForwardPassHandlerLoader.load(Path.of(modelPath), context));
		}

		// Build in‑process pipeline
		var pipeline = LocalInferencePipeline.from(shardMap, new ArrayList<>(handlers), config.vocabSize(),
				config.hiddenDim(), config.numHeads());

		// KV cache (size generous)
		var kvCache = new KVCacheManager(new GpuKVCache(512L * 1024 * 1024), new CpuKVCache(4096));

		var loop = new GenerationLoop(tokenizer, Sampler.create(), pipeline, kvCache);

		startRepl(loop, tokenizer);
	}

	// -------------------------------------------------------------------------
	// Cluster mode (forked JVMs) – same as original, but uses parsed arguments
	// -------------------------------------------------------------------------

	private static void runClusterRepl() throws Exception {
		String modeLabel = pType == ParallelismType.TENSOR ? "tensor-parallel" : "pipeline-parallel";
		print(Color.CYAN_BOLD + "▶ Starting 3-node " + modeLabel + " cluster (forked JVMs)..." + Color.RESET);

		int totalLayers;
		int numHeads;
		int vocabSize;
		try (GgufReader cfgReader = GgufReader.open(Path.of(modelPath))) {
			LlamaConfig cfg = LlamaConfig.from(cfgReader);
			totalLayers = cfg.numLayers();
			numHeads    = cfg.numHeads();
			vocabSize   = cfg.vocabSize();
		}

		ClusterHarness harness = (pType == ParallelismType.TENSOR)
				? ClusterHarness.tensorNodes(modelPath, totalLayers, numHeads)
				: ClusterHarness.threeNodes(modelPath, totalLayers);

		Runtime.getRuntime().addShutdownHook(Thread.ofVirtual().unstarted(() -> {
			print("\n" + Color.YELLOW + "⏹ Shutting down cluster..." + Color.RESET);
			try {
				harness.stop();
			} catch (Exception e) {
				/* best effort */ }
			print(Color.YELLOW + "✔ Cluster stopped." + Color.RESET);
		}));

		harness.start();
		print(Color.GREEN + "✔ Cluster ready  (" + modeLabel + "  " + dtype + " activations)" + Color.RESET + "\n");

		// For tensor mode harness.pipeline() returns TensorParallelPipelineClient
		// (built with actual vocabSize inside startTensorParallel).
		// For pipeline mode we build our own client with the actual vocabSize from GGUF.
		var pipeline = (pType == ParallelismType.TENSOR)
				? harness.pipeline()
				: new ProcessPipelineClient(harness.nodeAddresses(), vocabSize, dtype);

		Tokenizer tokenizer;
		try (GgufReader reader = GgufReader.open(Path.of(modelPath))) {
			tokenizer = GgufTokenizer.load(reader);
		}

		var kvCache = new KVCacheManager(new GpuKVCache(512L * 1024 * 1024), new CpuKVCache(4096));
		var loop    = new GenerationLoop(tokenizer, Sampler.create(), pipeline, kvCache);

		startRepl(loop, tokenizer);
	}

	// -------------------------------------------------------------------------
	// Common REPL loop
	// -------------------------------------------------------------------------

	private static void startRepl(GenerationLoop loop, Tokenizer tokenizer) throws IOException {
		SamplingParams params = SamplingParams.defaults().withMaxTokens(maxTokens).withTemperature(temperature)
				.withTopK(topK).withTopP(topP);

		ChatHistory history = new ChatHistory();

		print(Color.DIM + "Type your prompt and press Enter. Type 'exit' or Ctrl-C to quit." + Color.RESET);
		print("");

		BufferedReader stdin = new BufferedReader(new InputStreamReader(System.in));
		String line;

		while (true) {
			System.out.print(Color.CYAN_BOLD + "you> " + Color.RESET);
			System.out.flush();

			line = stdin.readLine();
			if (line == null)
				break;
			line = line.strip();
			if (line.isEmpty())
				continue;
			if (line.equalsIgnoreCase("exit") || line.equalsIgnoreCase("quit"))
				break;

			history.addUser(line);
			String modelType = ChatModelType.fromPath(modelPath);

			// Use ofSession so the GenerationLoop can reuse KV blocks from prior turns.
			// The session key (history.sessionId()) is stable for the entire conversation.
			InferenceRequest request = InferenceRequest.ofSession(history.sessionId(), modelType, history.getMessages(),
					params, RequestPriority.NORMAL);

			System.out.print(Color.GREEN_BOLD + "bot> " + Color.RESET);
			System.out.flush();

			long start = System.currentTimeMillis();

			var consumer = new TokenConsumer() {
				@Override
				public void onToken(String piece, int tokenId, int step) {
					if (!verbose)
						System.out.print(piece);
					else
						System.out.println("[" + step + ":" + tokenId + "]" + piece);
					System.out.flush();
				}

				@Override
				public void onPrefillStart(int promptLen) {
					System.out.print(Color.DIM + "(prefilling " + promptLen + " tokens…) " + Color.RESET);
					System.out.flush();
				}

				@Override
				public void onPrefillComplete() {
					System.out.print("\r" + Color.GREEN + "bot> " + Color.RESET);
					System.out.flush();
				}
			};

			GenerationResult result = loop.generate(request, consumer);

			history.addAssistant(result.text());

			long elapsed = System.currentTimeMillis() - start;
			System.out.println();
			System.out.printf(Color.GREEN + "     [%d tokens · %d ms · %s]" + Color.RESET + "%n",
					result.generatedTokens(), elapsed, dtype);
			System.out.println();
		}

		// Release KV memory for the session before exiting.
		loop.evictSession(history.sessionId());

		print(Color.YELLOW + "\nbye." + Color.RESET);
		System.exit(0);
	}

	// -------------------------------------------------------------------------
	// Helpers
	// -------------------------------------------------------------------------

	private static void banner() {
		System.out.println(String.format("  %sJuno interactive console  ·  model: %s%s%n", Color.YELLOW_BOLD_BRIGHT,
				Path.of(modelPath).getFileName(), Color.RESET));
		System.out.println(Color.RED_BOLD + "░▀▀█" + Color.GREEN_BOLD + "░█░█" + Color.RESET);
		System.out.println(Color.RED + "░░░█" + Color.GREEN + "░█░█" + Color.RESET);
		System.out.println(Color.RED + "░▀▀░" + Color.GREEN + "░▀▀▀" + Color.RESET);
		System.out.println(Color.BLUE_BOLD + "░█▀█" + Color.YELLOW_BOLD + "░█▀█" + Color.RESET);
		System.out.println(Color.BLUE + "░█░█" + Color.YELLOW + "░█░█" + Color.RESET);
		System.out.println(Color.BLUE + "░▀░▀" + Color.YELLOW + "░▀▀▀" + Color.RESET + "\n");
		System.out.println(
				String.format("  %sdtype=%s · max_tokens=%d · temperature=%.2f · top_k=%d · top_p=%.2f · %s nodes=%d%s%n",
						Color.GREEN_BOLD_BRIGHT, dtype, maxTokens, temperature, topK, topP,
						localMode ? "local" : "cluster", nodeCount, Color.RESET));
		if (jfrDuration != null) {
			System.out.println(String.format("  %s⏱ JFR active · duration=%s%s%n",
					Color.YELLOW, jfrDuration, Color.RESET));
		}
	}

	private static void print(String msg) {
		System.out.println(msg);
		System.out.flush();
	}

	private static ActivationDtype parseDtype(String s) {
		if (s == null)
			return ActivationDtype.FLOAT16;
		return switch (s.toUpperCase()) {
		case "FLOAT16", "F16", "FP16" -> ActivationDtype.FLOAT16;
		case "INT8", "I8" -> ActivationDtype.INT8;
		default -> ActivationDtype.FLOAT32;
		};
	}

	private static ParallelismType parseParallelismType(String s) {
		if (s == null)
			return ParallelismType.PIPELINE;
		return switch (s.toLowerCase()) {
		case "tensor" -> ParallelismType.TENSOR;
		default       -> ParallelismType.PIPELINE;
		};
	}

	private static int parseInt(String s, int def) {
		try {
			return Integer.parseInt(s);
		} catch (NumberFormatException e) {
			return def;
		}
	}

	private static float parseFloat(String s, float def) {
		try {
			return Float.parseFloat(s);
		} catch (NumberFormatException e) {
			return def;
		}
	}

	// Rough estimate of VRAM per layer (same as
	// ModelDescriptor.estimateVramPerLayer)
	private static long estimateVramPerLayer(int hiddenDim) {
		long params = 4L * hiddenDim * hiddenDim;
		return (long) (params * 2.0); // assume FP16 for estimation (bytes per param = 2)
	}

}