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
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.logging.Logger;

import jdk.jfr.Configuration;
import jdk.jfr.Recording;
import jdk.jfr.RecordingState;

import cab.ml.juno.metrics.MetricsMain;

import cab.ml.juno.coordinator.GenerationLoop;
import cab.ml.juno.coordinator.GenerationResult;
import cab.ml.juno.coordinator.InferenceRequest;
import cab.ml.juno.coordinator.RequestPriority;
import cab.ml.juno.coordinator.TokenConsumer;
import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.node.ActivationDtype;
import cab.ml.juno.node.CudaAvailability;
import cab.ml.juno.node.ForwardPassHandler;
import cab.ml.juno.node.CudaMatVec;
import cab.ml.juno.node.ForwardPassHandlerLoader;
import cab.ml.juno.node.GgufReader;
import cab.ml.juno.node.GpuContext;

import cab.ml.juno.node.LlamaConfig;
import cab.ml.juno.node.LocalInferencePipeline;
import cab.ml.juno.node.LoraAdamOptimizer;
import cab.ml.juno.node.LoraAdapterSet;
import cab.ml.juno.node.LoraTrainableHandler;
import cab.ml.juno.node.ShardContext;
import cab.ml.juno.registry.NodeDescriptor;
import cab.ml.juno.registry.NodeStatus;
import cab.ml.juno.registry.ParallelismType;
import cab.ml.juno.registry.ShardAssignment;
import cab.ml.juno.registry.ShardMap;
import cab.ml.juno.registry.ShardPlanner;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.GgufTokenizer;
import cab.ml.juno.tokenizer.Tokenizer;

/**
 * Interactive REPL that runs a model using the Juno engine.
 *
 * Can operate in three modes: - cluster mode (default): forks 3 node JVMs (as
 * before) - local mode (--local): runs all nodes in-process, no child JVMs -
 * lora mode (--lora): runs a single in-process node with LoRA fine-tuning
 *
 * LoRA persistence: adapters are saved to a separate .lora file, NOT packed
 * into the GGUF. This keeps the base model untouched and lets you swap adapters
 * freely. Use /merge-hint in the REPL to see how to bake weights in.
 *
 * Command-line arguments: --model-path PATH Path to GGUF file (required) --cpu
 * Force computation on CPU --dtype FLOAT32|FLOAT16 Activation wire format
 * (default: FLOAT16) --max-tokens N Max generated tokens (default: 200)
 * --temperature F Sampling temperature (default: 0.7) --local Use in-process
 * nodes instead of forking --nodes N Number of in-process nodes (default: 3)
 * --lora Enable LoRA fine-tuning mode (forces --local --nodes 1) --lora-path
 * PATH .lora checkpoint file (default: <model>.lora) --lora-rank N LoRA rank
 * (default: 8) --lora-alpha F LoRA alpha scaling (default: same as rank)
 * --lora-lr F Adam learning rate for LoRA (default: 1e-4) --lora-steps N
 * Gradient steps per /train command (default: 50) --verbose Show more logging
 * --help Show this help
 */
public final class ConsoleMain {

	@SuppressWarnings("unused")
	private static final Logger log = Logger.getLogger(ConsoleMain.class.getName());

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

	// ── Standard arguments ────────────────────────────────────────────────────
	private static String modelPath = null;
	private static ActivationDtype dtype = ActivationDtype.FLOAT16;
	private static int maxTokens = 200;
	private static float temperature = 0.6f;
	private static int topK = 20;
	private static float topP = 0.95f;
	private static boolean localMode = false;
	private static int nodeCount = 3;
	private static boolean verbose = false;
	private static boolean help = false;
	private static ParallelismType pType = ParallelismType.PIPELINE;
	private static String jfrDuration = null;
	// ── Byte-order argument ───────────────────────────────────────────────────
	/** Activation codec byte order: {@code "BE"} (default) or {@code "LE"}. */
	private static String byteOrder = "BE";
	// ── GPU arguments ─────────────────────────────────────────────────────────
	private static boolean useGpu = true; // use CPU
	// ── LoRA arguments ────────────────────────────────────────────────────────
	private static boolean loraMode = false;
	private static String loraPath = null; // auto-derived if null
	private static int loraRank = 8;
	private static float loraAlpha = -1f; // sentinel: default to loraRank
	private static double loraLr = 1e-4;
	private static int loraSteps = 50; // steps per chunk for /train
	private static int loraStepsQa = 10; // steps per chunk for /train-qa
	private static float loraEarlyStop = 0.25f; // stop training when loss drops below this

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

		// LoRA forces single in-process node
		if (loraMode) {
			localMode = true;
			nodeCount = 1;
			if (loraAlpha < 0)
				loraAlpha = loraRank;
			if (loraPath == null)
				loraPath = deriveLoraPath(modelPath);
		}

		System.setProperty("JUNO_USE_GPU", String.valueOf(useGpu));
		System.setProperty("juno.byteOrder", byteOrder);
		System.setProperty("MODEL_PATH", modelPath);
		System.setProperty("DTYPE", dtype.name());
		System.setProperty("MAX_TOKENS", String.valueOf(maxTokens));
		System.setProperty("TEMPERATURE", String.valueOf(temperature));
		System.setProperty("TOP_K", String.valueOf(topK));
		System.setProperty("TOP_P", String.valueOf(topP));
		if (verbose)
			System.setProperty("JUNO_VERBOSE", "true");
		// For lora mode the JFR lifecycle is delegated to the JVM flag set by run.sh.
		// For local and cluster modes, ConsoleMain manages JFR programmatically via
		// startLocalJfr() / startClusterJfr() — no JVM flag is involved.
		if (jfrDuration != null && !localMode && loraMode)
			System.setProperty("juno.jfr.duration", jfrDuration);

		banner();

		if (loraMode) {
			runLoraRepl();
		} else if (localMode && jfrDuration != null) {
			startLocalJfr();
		} else if (localMode) {
			runLocalRepl();
		} else if (jfrDuration != null) {
			startClusterJfr();
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
				if (i + 1 < args.length)
					i++;
				break;
			case "--jfr":
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
			// ── Byte order ───────────────────────────────────────────────────────
			case "--byteOrder":
			case "--byte-order":
			case "--byteorder":
				if (i + 1 < args.length) {
					String bo = args[++i].toUpperCase();
					byteOrder = "LE".equals(bo) ? "LE" : "BE";
				}
				break;
			// ── GPU ──────────────────────────────────────────────────────────
			case "--gpu":
				useGpu = true;
				break;
			case "--cpu":
				useGpu = false;
				break;
				// ── LoRA ──────────────────────────────────────────────────────────
			case "--lora":
				loraMode = true;
				break;
			case "--lora-path":
				if (i + 1 < args.length)
					loraPath = args[++i];
				break;
			case "--lora-rank":
				if (i + 1 < args.length)
					loraRank = parseInt(args[++i], 8);
				break;
			case "--lora-alpha":
				if (i + 1 < args.length)
					loraAlpha = parseFloat(args[++i], -1f);
				break;
			case "--lora-lr":
				if (i + 1 < args.length)
					loraLr = parseDouble(args[++i], 1e-4);
				break;
			case "--lora-steps":
				if (i + 1 < args.length)
					loraSteps = parseInt(args[++i], 50);
				break;
			case "--lora-steps-qa":
				if (i + 1 < args.length)
					loraStepsQa = parseInt(args[++i], 10);
				break;
			case "--lora-early-stop":
				if (i + 1 < args.length)
					loraEarlyStop = parseFloat(args[++i], 0.25f);
				break;
			// ─────────────────────────────────────────────────────────────────
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
		System.out.println("Inference options:");
		System.out.println("  --gpu                      Use GPU (default, no need to set)");
		System.out.println("  --cpu                      Force to use CPU");
		System.out.println("  --pType pipeline|tensor    Parallelism type (default: pipeline)");
		System.out.println("  --dtype FLOAT32|FLOAT16    Activation wire format (default: FLOAT16)");
		System.out.println("  --max-tokens N             Max generated tokens (default: 200)");
		System.out.println("  --temperature F            Sampling temperature (default: 0.6)");
		System.out.println("  --top-k N                  Top-K sampling cutoff (default: 20)");
		System.out.println("  --top-p F                  Nucleus sampling top-p (default: 0.95)");
		System.out.println("  --byteOrder BE|LE          Activation codec byte order (default: BE)");
		System.out.println("                             BE = big-endian (default, hardware-validated)");
		System.out.println("                             LE = little-endian (native x86 order)");
		System.out.println("  --local                    Use in-process nodes (no forking)");
		System.out.println("  --nodes N                  Number of in-process nodes (default: 3)");
		System.out.println();
		System.out.println("LoRA fine-tuning (forces --local --nodes 1):");
		System.out.println("  --lora                     Enable LoRA fine-tuning mode");
		System.out.println("  --lora-path PATH           Adapter checkpoint file (default: <model>.lora)");
		System.out.println("  --lora-rank N              Low-rank bottleneck dimension (default: 8)");
		System.out.println("  --lora-alpha F             Scale factor alpha (default: same as rank)");
		System.out.println("  --lora-lr F                Adam learning rate (default: 1e-4)");
		System.out.println("  --lora-steps N             Gradient steps per /train command (default: 50)");
		System.out.println();
		System.out.println("  LoRA REPL commands:");
		System.out.println("    /train <text>            Fine-tune on inline text");
		System.out.println("    /train-file <path>       Fine-tune on a text file (splits into chunks)");
		System.out.println("    /save                    Save adapter checkpoint to --lora-path");
		System.out.println("    /reset                   Reinitialize adapters (loses training)");
		System.out.println("    /status                  Show adapter info and training stats");
		System.out.println("    /merge-hint              Explain how to bake LoRA into GGUF weights");
		System.out.println("    Regular chat input       Inference with current adapter applied");
		System.out.println();
		System.out.println("  LoRA training controls:");
		System.out.println("  --lora-steps N           Gradient steps/chunk for /train (default: 50)");
		System.out.println("  --lora-steps-qa N        Gradient steps/chunk for /train-qa (default: 10)");
		System.out.println("  --lora-early-stop F      Stop when loss < F (default: 0.25). Prevents");
		System.out.println("                           catastrophic overfitting. Set 0 to disable.");
		System.out.println();
		System.out.println("Other:");
		System.out.println("  --jfr DURATION             Java Flight Recording duration (e.g. 5m)");
		System.out.println("  --verbose, -v              Show more logging");
		System.out.println("  --help, -h                 Show this help");
	}

	// ── LoRA mode ─────────────────────────────────────────────────────────────

	/**
	 * LoRA fine-tuning REPL. Runs a single in-process LoraTrainableHandler that
	 * serves both inference (with LoRA delta) and training (/train commands).
	 *
	 * Adapters are persisted in a separate .lora file alongside the model. The base
	 * GGUF is never modified.
	 */
	private static void runLoraRepl() throws Exception {
		print(Color.DIM + "  Adapter file: " + loraPath + Color.RESET);

		LlamaConfig config;
		Tokenizer tokenizer;
		try (GgufReader reader = GgufReader.open(Path.of(modelPath))) {
			config = LlamaConfig.from(reader);
			tokenizer = GgufTokenizer.load(reader);
		}

		// Load or create adapter set
		LoraAdapterSet adapters;
		Path adapterFile = Path.of(loraPath);
		if (Files.exists(adapterFile)) {
			adapters = LoraAdapterSet.load(adapterFile);
			print(Color.GREEN + "  ✔ Loaded checkpoint: " + adapters.size() + " adapters from " + loraPath
					+ Color.RESET);
		} else {
			adapters = LoraAdapterSet.qv(config, loraRank, loraAlpha, new Random(42));
			print(Color.YELLOW + "  ✦ New adapters initialised (" + adapters.size() + " total · /save to persist)"
					+ Color.RESET);
		}

		// Single-node ShardContext covering the full model
		ShardAssignment assignment = new ShardAssignment("lora-node", "localhost", 0, 0, config.numLayers(), true,
				true);
		ShardMap shardMap = new ShardMap("model", config.numLayers(), List.of(assignment), Instant.now());
		ShardContext ctx = ShardContext.from(assignment, config.vocabSize(), config.hiddenDim(), config.numHeads());

		print(Color.DIM + "  Loading model weights…" + Color.RESET);
		LoraTrainableHandler handler = LoraTrainableHandler.load(Path.of(modelPath), ctx, adapters);
		print(Color.GREEN + "  ✔ Model loaded  (" + config + ")" + Color.RESET + "\n");

		// Wrap in LocalInferencePipeline for standard inference path
		var pipeline = LocalInferencePipeline.from(shardMap, List.of(handler), config.vocabSize(), config.hiddenDim(),
				config.numHeads());
		var kvCache = new KVCacheManager(new GpuKVCache(512L * 1024 * 1024), new CpuKVCache(4096));
		var loop = new GenerationLoop(tokenizer, Sampler.create(), pipeline, kvCache);

		LoraAdamOptimizer optimizer = LoraAdamOptimizer.defaults(loraLr);
		int[] totalStepsTrained = { 0 };
		boolean[] dirty = { false }; // unsaved changes?

		SamplingParams params = SamplingParams.defaults().withMaxTokens(maxTokens).withTemperature(temperature)
				.withTopK(topK).withTopP(topP);

		ChatHistory history = new ChatHistory();

		print(Color.DIM + "Type to chat, or use /train <text>  /save  /status  /help" + Color.RESET);
		print("");

		BufferedReader stdin = new BufferedReader(new InputStreamReader(System.in));
		String line;

		while (true) {
			System.out.print(Color.CYAN_BOLD + "you" + Color.RESET + Color.YELLOW + (dirty[0] ? "*" : " ") + Color.RESET
					+ Color.CYAN_BOLD + "> " + Color.RESET);
			System.out.flush();

			line = stdin.readLine();
			if (line == null)
				break;
			line = line.strip();
			if (line.isEmpty())
				continue;

			// ── LoRA commands ──────────────────────────────────────────────
			if (line.startsWith("/")) {
				handleLoraCommand(line, adapters, optimizer, handler, tokenizer, adapterFile, totalStepsTrained, dirty);
				continue;
			}

			if (line.equalsIgnoreCase("exit") || line.equalsIgnoreCase("quit")) {
				if (dirty[0]) {
					System.out
							.print(Color.YELLOW + "  Unsaved adapter changes. Save before exit? [y/N] " + Color.RESET);
					System.out.flush();
					String yn = stdin.readLine();
					if (yn != null && yn.strip().equalsIgnoreCase("y")) {
						saveAdapters(adapters, adapterFile, totalStepsTrained[0]);
					}
				}
				break;
			}

			// ── Regular inference ──────────────────────────────────────────
			history.addUser(line);
			String modelType = ChatModelType.fromPath(modelPath);
			InferenceRequest request = InferenceRequest.ofSession(history.sessionId(), modelType, history.getMessages(),
					params, RequestPriority.NORMAL);

			System.out.print(Color.GREEN_BOLD + "bot> " + Color.RESET);
			System.out.flush();

			long start = System.currentTimeMillis();
			var consumer = streamingConsumer(verbose);
			GenerationResult result = loop.generate(request, consumer);
			history.addAssistant(result.text());

			long elapsed = System.currentTimeMillis() - start;
			System.out.println();
			System.out.printf(Color.GREEN + "     [%d tokens · %d ms · LoRA rank=%d]" + Color.RESET + "%n",
					result.generatedTokens(), elapsed, loraRank);
			System.out.println();
		}

		loop.evictSession(history.sessionId());
		print(Color.YELLOW + "\nbye." + Color.RESET);
		System.exit(0);
	}

	private static void handleLoraCommand(String line, LoraAdapterSet adapters, LoraAdamOptimizer optimizer,
			LoraTrainableHandler handler, Tokenizer tokenizer, Path adapterFile, int[] totalSteps, boolean[] dirty)
			throws Exception {

		String[] parts = line.split("\\s+", 2);
		String cmd = parts[0].toLowerCase();

		switch (cmd) {

		case "/train" -> {
			if (parts.length < 2 || parts[1].isBlank()) {
				print(Color.RED + "  Usage: /train <text to learn>" + Color.RESET);
				return;
			}
			trainOnText(parts[1], adapters, optimizer, handler, tokenizer, totalSteps, dirty);
		}

		case "/train-qa" -> {
			// Format: /train-qa Q: <question> A: <answer>
			// OR two-arg form (separator is " A: "): /train-qa What is my name? A: Dima
			if (parts.length < 2 || parts[1].isBlank()) {
				print(Color.RED + "  Usage: /train-qa <question> A: <answer>" + Color.RESET);
				print(Color.RED + "  Example: /train-qa What is my name? A: Your name is Dima." + Color.RESET);
				return;
			}
			String qaSrc = parts[1].trim();
			// Strip optional leading "Q:" prefix
			if (qaSrc.toLowerCase().startsWith("q:"))
				qaSrc = qaSrc.substring(2).trim();
			int sepIdx = qaSrc.indexOf(" A: ");
			if (sepIdx < 0) {
				print(Color.RED + "  Could not find \" A: \" separator." + Color.RESET);
				print(Color.RED + "  Usage: /train-qa What is my name? A: Your name is Dima." + Color.RESET);
				return;
			}
			String question = qaSrc.substring(0, sepIdx).trim();
			String answer = qaSrc.substring(sepIdx + 4).trim();
			trainOnQA(question, answer, adapters, optimizer, handler, tokenizer, totalSteps, dirty,
					ChatModelType.fromPath(modelPath));
		}

		case "/train-file" -> {
			if (parts.length < 2 || parts[1].isBlank()) {
				print(Color.RED + "  Usage: /train-file <path>" + Color.RESET);
				return;
			}
			Path p = Path.of(parts[1].strip());
			if (!Files.exists(p)) {
				print(Color.RED + "  File not found: " + p + Color.RESET);
				return;
			}
			String text = Files.readString(p);
			print(Color.DIM + "  Loaded: " + p.getFileName() + "  (" + text.length() + " chars)" + Color.RESET);
			trainOnText(text, adapters, optimizer, handler, tokenizer, totalSteps, dirty);
		}

		case "/save" -> saveAdapters(adapters, adapterFile, totalSteps[0]);

		case "/reset" -> {
			System.out.print(Color.YELLOW + "  Reset adapter weights? All training will be lost. [y/N] " + Color.RESET);
			System.out.flush();
			String yn = new BufferedReader(new InputStreamReader(System.in)).readLine();
			if (yn != null && yn.strip().equalsIgnoreCase("y")) {
				// Reinitialise B to zero and reset optimizer
				for (var a : adapters.all())
					java.util.Arrays.fill(a.b(), 0f);
				optimizer.reset();
				totalSteps[0] = 0;
				dirty[0] = false;
				print(Color.GREEN + "  ✔ Adapters reset to initial state." + Color.RESET);
			}
		}

		case "/status" -> {
			long adapterBytes = adapters.all().stream().mapToLong(a -> (a.a().length + a.b().length) * 4L).sum();
			print("");
			print(Color.CYAN_BOLD + "  LoRA status" + Color.RESET);
			print("  ─────────────────────────────────");
			print("  adapters  : " + adapters.size() + "  (wq + wv on all layers)");
			print("  rank      : " + loraRank);
			print("  alpha     : " + loraAlpha + "  (scale = " + (loraAlpha / loraRank) + ")");
			print("  parameters: " + (adapterBytes / 4) + "  (" + (adapterBytes / 1024) + " KB)");
			print("  trained   : " + totalSteps[0] + " gradient steps");
			print("  checkpoint: " + adapterFile + (dirty[0] ? "  " + Color.YELLOW + "[unsaved]" + Color.RESET : ""));
			print("  lr        : " + loraLr + "  ·  optimizer step = " + optimizer.step());
			print("");
		}

		case "/merge-hint" -> {
			print("");
			print(Color.CYAN_BOLD + "  Merging LoRA into base weights (offline)" + Color.RESET);
			print("  ─────────────────────────────────────────────────────────────");
			print("  Juno keeps adapters separate by design. The base GGUF is never");
			print("  modified. To bake the adapter in (merged = W + scale·B·A):");
			print("");
			print("  1. Dequantize each projection weight W to float32.");
			print("  2. For each LoRA adapter: W_eff = W + (alpha/rank) × B × A");
			print("  3. Re-quantize W_eff back to the original format (Q4_K etc.).");
			print("  4. Write a new GGUF file with the merged weights.");
			print("");
			print("  This creates a standalone model that doesn't need the .lora file.");
			print("  Juno doesn't include a merge tool yet — contributions welcome.");
			print("  See LORA.md for the weight formula reference.");
			print("");
		}

		case "/help" -> {
			print("");
			print(Color.CYAN_BOLD + "  LoRA REPL commands" + Color.RESET);
			print("  /train-qa <q> A: <a>  Fine-tune on a Q&A pair in the correct chat format  ← USE THIS");
			print("  /train <text>          Fine-tune on raw text (no chat template applied)");
			print("  /train-file <path>     Fine-tune on a text file (chunks of ~128 tokens)");
			print("  /save                  Save adapter to " + adapterFile);
			print("  /reset                 Reinitialise adapters (clear all training)");
			print("  /status                Show adapter info and training statistics");
			print("  /merge-hint            Explain how to bake adapters into model weights");
			print("  Regular input          Chat using the current adapter for inference");
			print("");
			print("  " + Color.YELLOW + "TIP:" + Color.RESET
					+ " Use /train-qa for factual recall (names, dates, preferences).");
			print("       /train is for style/vocabulary adaptation.");
			print("");
		}

		default -> print(Color.RED + "  Unknown command: " + cmd + "  (type /help for commands)" + Color.RESET);
		}
	}

	/**
	 * Fine-tune on a question/answer pair using the model's own chat template.
	 *
	 * <p>
	 * This is the correct way to teach the model factual recall. The training text
	 * is formatted with the same Zephyr/phi3/llama3/... template that the model
	 * sees during inference, so the LoRA adapters learn the right token
	 * distribution. Using {@code /train} with raw text produces no effect on
	 * question-answering because the training context (plain sentence) doesn't
	 * match the inference context (chat-templated question+answer).
	 *
	 * <p>
	 * Generates multiple phrasings of the question automatically to avoid the model
	 * overfitting to a single exact wording.
	 */
	private static void trainOnQA(String question, String answer, LoraAdapterSet adapters, LoraAdamOptimizer optimizer,
			LoraTrainableHandler handler, Tokenizer tokenizer, int[] totalSteps, boolean[] dirty, String modelType)
			throws Exception {

		// Echo the parsed question and answer BEFORE training starts — catches typos.
		// "mt" vs "my" won't be caught by the model; it must be caught by the human.
		print("");
		print(Color.CYAN_BOLD + "  Question: " + Color.RESET + question);
		print(Color.CYAN_BOLD + "  Answer  : " + Color.RESET + answer);
		print(Color.DIM + "  (check spelling above — typos in Q won't match inference phrasing)" + Color.RESET);
		print("");

		// Build several phrasings of the same Q&A so the model generalises.
		// All variations must use the EXACT spelling of the key terms (name, etc.)
		// because the BPE tokenization of "mt" ≠ "my" → different token IDs → no
		// transfer.
		String q = question.endsWith("?") ? question : question + "?";
		String qLow = q.substring(0, 1).toLowerCase() + q.substring(1); // lower-case version
		String[] questions = { q, qLow, "Can you tell me: " + qLow, "Please answer: " + qLow, };

		// Format each as a full chat exchange using the model's own template
		// so training and inference see identical token sequences.
		StringBuilder sb = new StringBuilder();
		for (String variant : questions) {
			sb.append(formatQA(variant, answer, modelType));
		}
		String trainingText = sb.toString();

		print(Color.DIM + "  Formatted as " + questions.length + " Q&A pairs  ·  model type: " + modelType
				+ Color.RESET);
		print(Color.DIM + "  steps/chunk=" + loraStepsQa + "  early-stop=" + loraEarlyStop
				+ "  (tune with --lora-steps-qa N  --lora-early-stop F)" + Color.RESET);

		trainOnText(trainingText, adapters, optimizer, handler, tokenizer, totalSteps, dirty, loraStepsQa);
	}

	/**
	 * Format a single question/answer pair using the chat template for
	 * {@code modelType}. Mirrors the format applied by
	 * {@link cab.ml.juno.tokenizer.ChatTemplateFormatter} so training and inference
	 * see identical token sequences.
	 */
	private static String formatQA(String question, String answer, String modelType) {
		return switch (modelType) {
		case "tinyllama", "zephyr" -> "<|user|>\n" + question + "</s>\n<|assistant|>\n" + answer + "</s>\n";
		case "phi3", "phi-3" -> "<|user|>\n" + question + "<|end|>\n<|assistant|>\n" + answer + "<|end|>\n";
		case "llama3" -> "<|start_header_id|>user<|end_header_id|>\n\n" + question
				+ "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" + answer + "<|eot_id|>";
		case "mistral" -> "[INST] " + question + " [/INST] " + answer + "</s>";
		case "gemma" -> "<start_of_turn>user\n" + question + "<end_of_turn>\n" + "<start_of_turn>model\n" + answer
				+ "<end_of_turn>\n";
		default -> // chatml
			"<|im_start|>user\n" + question + "<|im_end|>\n" + "<|im_start|>assistant\n" + answer + "<|im_end|>\n";
		};
	}

	/**
	 * Tokenize {@code text}, split into chunks of ≤128 tokens, and run
	 * {@link LoraTrainableHandler#trainStep} for {@code loraSteps} gradient steps
	 * per chunk, printing a compact progress bar.
	 */
	private static void trainOnText(String text, LoraAdapterSet adapters, LoraAdamOptimizer optimizer,
			LoraTrainableHandler handler, Tokenizer tokenizer, int[] totalSteps, boolean[] dirty) throws Exception {
		trainOnText(text, adapters, optimizer, handler, tokenizer, totalSteps, dirty, loraSteps);
	}

	private static void trainOnText(String text, LoraAdapterSet adapters, LoraAdamOptimizer optimizer,
			LoraTrainableHandler handler, Tokenizer tokenizer, int[] totalSteps, boolean[] dirty, int stepsPerChunk)
			throws Exception {

		int[] allTokens = tokenizer.encode(text);
		if (allTokens.length < 2) {
			print(Color.YELLOW + "  Input too short to train on (need ≥ 2 tokens)." + Color.RESET);
			return;
		}

		int[] withBos = new int[allTokens.length + 1];
		withBos[0] = 1; // BOS
		System.arraycopy(allTokens, 0, withBos, 1, allTokens.length);

		// CHUNK=32 keeps T (prediction positions per step) small so steps complete
		// in ~10–15 s on CPU. CHUNK=128 gives T=128 → ~50 s/step → appears frozen.
		final int CHUNK = 32;
		List<int[]> chunks = new ArrayList<>();
		for (int start = 0; start < withBos.length - 1; start += CHUNK) {
			int end = Math.min(start + CHUNK + 1, withBos.length);
			if (end - start < 2)
				break;
			int[] chunk = new int[end - start];
			System.arraycopy(withBos, start, chunk, 0, chunk.length);
			chunks.add(chunk);
		}

		int totalGradSteps = chunks.size() * stepsPerChunk;
		print("");
		System.out.printf("  %sTraining%s  rank=%d · lr=%s · %d steps · %d chunk(s) · %d tokens%n", Color.CYAN_BOLD,
				Color.RESET, loraRank, loraLr, totalGradSteps, chunks.size(), withBos.length);
		print("  " + "─".repeat(62));

		float firstLoss = Float.NaN, lastLoss = Float.NaN;
		int stepsDone = 0;
		long trainStart = System.currentTimeMillis();
		boolean stoppedEarly = false;

		outer: for (int[] chunk : chunks) {
			for (int s = 0; s < stepsPerChunk; s++) {
				long stepStart = System.currentTimeMillis();
				float loss = handler.trainStep(chunk, optimizer);
				long stepMs = System.currentTimeMillis() - stepStart;

				if (Float.isNaN(firstLoss))
					firstLoss = loss;
				lastLoss = loss;
				stepsDone++;

				// Early stop: loss below threshold → model has memorised the chunk.
				// Continuing would drive loss toward 0 and destroy generalisation.
				if (loraEarlyStop > 0 && loss < loraEarlyStop) {
					int pct = (int) (100.0 * stepsDone / totalGradSteps);
					int bars = Math.min(20, pct / 5);
					String bar = Color.GREEN + "▓".repeat(bars) + Color.DIM + "░".repeat(20 - bars) + Color.RESET;
					System.out.printf("\r  step %3d/%-3d  loss=%.4f  %s %3d%%  %4dms/step  ETA %-8s", stepsDone,
							totalGradSteps, loss, bar, pct, stepMs, "0s");
					System.out.flush();
					System.out.println();
					System.out.printf("  %s⚠ Early stop%s  loss=%.4f < threshold=%.2f%n", Color.YELLOW, Color.RESET,
							loss, loraEarlyStop);
					stoppedEarly = true;
					break outer;
				}

				// Overfitting warning: loss < 0.5 but early stop not triggered
				// (user set loraEarlyStop=0 or threshold is very low)
				if (loss < 0.5f && loraEarlyStop == 0) {
					System.out.printf(
							"\r  %s⚠ loss=%.4f — risk of overfitting. "
									+ "Consider stopping soon or lowering --lora-steps-qa.%s%n",
							Color.YELLOW, loss, Color.RESET);
				}

				// ETA based on rolling average of all steps so far
				long elapsed = System.currentTimeMillis() - trainStart;
				long msPerStep = elapsed / stepsDone;
				long etaMs = msPerStep * (totalGradSteps - stepsDone);
				String eta = etaMs > 60_000 ? String.format("%dm%02ds", etaMs / 60_000, (etaMs % 60_000) / 1000)
						: String.format("%ds", etaMs / 1000);

				int pct = (int) (100.0 * stepsDone / totalGradSteps);
				int bars = pct / 5;
				String bar = Color.GREEN + "▓".repeat(bars) + Color.DIM + "░".repeat(20 - bars) + Color.RESET;
				System.out.printf("\r  step %3d/%-3d  loss=%.4f  %s %3d%%  %4dms/step  ETA %-8s", stepsDone,
						totalGradSteps, loss, bar, pct, stepMs, eta);
				System.out.flush();
			}
		}

		System.out.println();
		long totalMs = System.currentTimeMillis() - trainStart;
		float delta = Float.isNaN(firstLoss) ? 0f : lastLoss - firstLoss;
		String trend = delta < 0 ? Color.GREEN + String.format("▼ %.4f (−%.4f)", lastLoss, -delta) + Color.RESET
				: Color.YELLOW + String.format("▲ %.4f (+%.4f)", lastLoss, delta) + Color.RESET;
		String doneLabel = stoppedEarly ? Color.YELLOW + "⚠ stopped early" + Color.RESET
				: Color.GREEN + "✔ done" + Color.RESET;
		System.out.printf("  %s  loss=%s  %ds total  · /save to persist%n", doneLabel, trend, totalMs / 1000);
		if (lastLoss < 0.1f) {
			System.out.printf("  %s⚠ WARNING: loss=%.4f — adapter is severely overfit.%s%n", Color.RED, lastLoss,
					Color.RESET);
			System.out.printf("  %s  The model will generate garbage until you /reset adapters.%s%n%n", Color.RED,
					Color.RESET);
		} else if (lastLoss < 0.5f && !stoppedEarly) {
			System.out.printf("  %s⚠ loss < 0.5 — consider stopping here. " + "More steps risk overfitting.%s%n%n",
					Color.YELLOW, Color.RESET);
		} else {
			System.out.println();
		}

		totalSteps[0] += stepsDone;
		dirty[0] = true;
	}

	private static void saveAdapters(LoraAdapterSet adapters, Path path, int stepsTrained) {
		try {
			Files.createDirectories(path.getParent() != null ? path.getParent() : Path.of("."));
			adapters.save(path);
			long kb = Files.size(path) / 1024;
			print(Color.GREEN + "  ✔ Saved → " + path + "  (" + adapters.size() + " adapters · " + kb + " KB" + "  · "
					+ stepsTrained + " steps trained)" + Color.RESET);
		} catch (IOException e) {
			print(Color.RED + "  ✘ Save failed: " + e.getMessage() + Color.RESET);
		}
	}

	/** Derive a .lora path from the model path (same dir, same stem). */
	private static String deriveLoraPath(String mp) {
		Path p = Path.of(mp);
		String name = p.getFileName().toString();
		int dot = name.lastIndexOf('.');
		String stem = dot > 0 ? name.substring(0, dot) : name;
		Path parent = p.getParent();
		return (parent != null ? parent.resolve(stem) : Path.of(stem)) + ".lora";
	}

	// ── JFR local mode ────────────────────────────────────────────────────────

	/**
	 * Starts a programmatic JFR recording, runs the local REPL, and once the
	 * recording period expires automatically extracts and prints metrics JSON.
	 *
	 * <p>Uses {@code jdk.jfr.Recording} so the JFR lifecycle is fully managed
	 * in-process — no JVM flags required. A daemon virtual thread polls
	 * {@code RecordingState}; when the state becomes {@code STOPPED} (duration
	 * elapsed), {@link #extractAndPrintJfrMetrics(Path)} is called.
	 */
	private static void startLocalJfr() throws Exception {
		String modelName = Path.of(modelPath).getFileName().toString();
		String modelStem = modelName.contains(".") ? modelName.substring(0, modelName.lastIndexOf('.')) : modelName;
		String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"));
		String jfrFileName = "juno-" + modelStem + "-" + timestamp + ".jfr";
		Path jfrFile = Path.of(jfrFileName);

		Duration duration = parseJfrDuration(jfrDuration);

		Configuration cfg = Configuration.getConfiguration("profile");
		Recording rec = new Recording(cfg);
		rec.setDuration(duration);
		rec.setDestination(jfrFile);
		rec.start();

		print(Color.YELLOW + "  ⏱ JFR recording started — duration=" + jfrDuration
				+ "  output=" + jfrFileName + Color.RESET + "\n");

		// Shutdown hook guarantees extraction runs even when startRepl() calls System.exit(0).
		// We capture rec/jfrFile/modelStem/modelName in effectively-final locals.
		final Recording recRef = rec;
		final String modelStemFinal = modelStem;
		final String modelNameFinal = modelName;
		Runtime.getRuntime().addShutdownHook(Thread.ofVirtual().unstarted(() -> {
			try {
				if (recRef.getState() == RecordingState.RUNNING) {
					recRef.stop();
				}
				// Brief wait for the file to be fully written
				Thread.sleep(500);
				extractAndPrintJfrMetrics(jfrFile, modelStemFinal, modelNameFinal);
			} catch (Exception e) {
				System.err.println("JFR metrics extraction failed: " + e.getMessage());
			} finally {
				recRef.close();
			}
		}));

		runLocalRepl(); // calls System.exit(0) on quit — shutdown hook fires from there
	}

	/**
	 * Starts a programmatic JFR recording on the coordinator, injects
	 * {@code -XX:StartFlightRecording} into every forked node JVM via
	 * {@link ClusterHarness#withJfr}, runs the cluster REPL, and on exit
	 * aggregates all four JFR files (coordinator + 3 nodes) before printing
	 * the merged metrics summary.
	 *
	 * <p>A <em>single</em> shutdown hook owns the full teardown sequence so that
	 * ordering is guaranteed:
	 * <ol>
	 *   <li>Stop coordinator's {@link Recording} (flushes its JFR file).</li>
	 *   <li>{@link ClusterHarness#stop()} — destroys node processes; their
	 *       {@code dumponexit=true} flag writes each node's JFR file.</li>
	 *   <li>Brief sleep to let OS flush all files to disk.</li>
	 *   <li>Merge-extract from coordinator + node files → print JSON summary.</li>
	 * </ol>
	 */
	private static void startClusterJfr() throws Exception {
		String modelName = Path.of(modelPath).getFileName().toString();
		String modelStem = modelName.contains(".") ? modelName.substring(0, modelName.lastIndexOf('.')) : modelName;
		String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"));
		String coordinatorJfrName = "juno-" + modelStem + "-" + timestamp + ".jfr";
		Path coordinatorJfrFile = Path.of(coordinatorJfrName);

		// ── Coordinator recording ─────────────────────────────────────────────
		Duration duration = parseJfrDuration(jfrDuration);
		Configuration cfg = Configuration.getConfiguration("profile");
		Recording rec = new Recording(cfg);
		rec.setDuration(duration);
		rec.setDestination(coordinatorJfrFile);
		rec.start();

		print(Color.YELLOW + "  ⏱ JFR recording started — duration=" + jfrDuration
				+ "  output=" + coordinatorJfrName + Color.RESET + "\n");

		// ── Cluster setup — nodes get their own JFR via withJfr() ─────────────
		String modeLabel = pType == ParallelismType.TENSOR ? "tensor-parallel" : "pipeline-parallel";
		print(Color.CYAN_BOLD + "▶ Starting 3-node " + modeLabel + " cluster (forked JVMs)..." + Color.RESET);

		int totalLayers, numHeads, vocabSize;
		try (GgufReader cfgReader = GgufReader.open(Path.of(modelPath))) {
			LlamaConfig cfg2 = LlamaConfig.from(cfgReader);
			totalLayers = cfg2.numLayers();
			numHeads = cfg2.numHeads();
			vocabSize = cfg2.vocabSize();
		}

		ClusterHarness harness = ((pType == ParallelismType.TENSOR)
				? ClusterHarness.tensorNodes(modelPath, totalLayers, numHeads)
				: ClusterHarness.threeNodes(modelPath, totalLayers))
				.withJfr(jfrDuration, timestamp);

		// ── Single combined shutdown hook — ordering matters ──────────────────
		final Recording recRef = rec;
		final String modelStemFinal = modelStem;
		final String modelNameFinal = modelName;
		final ClusterHarness harnessRef = harness;
		final Path coordFile = coordinatorJfrFile;
		Runtime.getRuntime().addShutdownHook(Thread.ofVirtual().unstarted(() -> {
			// 1. Stop coordinator recording so its JFR file is fully written.
			try {
				if (recRef.getState() == RecordingState.RUNNING)
					recRef.stop();
			} catch (Exception ignored) {}

			// 2. Stop cluster → destroys node processes → dumponexit fires on each node.
			print("\n" + Color.YELLOW + "⏹ Shutting down cluster..." + Color.RESET);
			try { harnessRef.stop(); } catch (Exception ignored) {}
			print(Color.YELLOW + "✔ Cluster stopped." + Color.RESET);

			// 3. Wait for coordinator + node JFR files to be fully flushed to disk.
			try { Thread.sleep(2000); } catch (InterruptedException ignored) {}

			// 4. Merge-extract from coordinator + all node files and print.
			try {
				recRef.close();
				List<Path> allFiles = new ArrayList<>();
				allFiles.add(coordFile);
				allFiles.addAll(harnessRef.nodeJfrFiles());
				extractAndPrintJfrMetricsMerged(allFiles, modelStemFinal, modelNameFinal);
			} catch (Exception e) {
				System.err.println("JFR metrics extraction failed: " + e.getMessage());
			}
		}));

		harness.start();
		print(Color.GREEN + "✔ Cluster ready  (" + modeLabel + "  " + dtype + " activations)" + Color.RESET + "\n");

		var pipeline = (pType == ParallelismType.TENSOR)
				? harness.pipeline()
				: new ProcessPipelineClient(harness.nodeAddresses(), vocabSize, dtype);

		Tokenizer tokenizer;
		try (GgufReader reader = GgufReader.open(Path.of(modelPath))) {
			tokenizer = GgufTokenizer.load(reader);
		}

		var kvCache = new KVCacheManager(new GpuKVCache(512L * 1024 * 1024), new CpuKVCache(4096));
		var loop = new GenerationLoop(tokenizer, Sampler.create(), pipeline, kvCache);

		startRepl(loop, tokenizer); // calls System.exit(0) on quit — shutdown hook fires from there
	}

	/**
	 * Merges the given JFR files (coordinator + node files), extracts metrics, and
	 * prints the JSON summary to the console — same presentation as
	 * {@link #extractAndPrintJfrMetrics} but across multiple files.
	 *
	 * <p>Files that do not exist (e.g. a node that crashed before dumping) are silently
	 * skipped so a partial result is still reported.
	 */
	private static void extractAndPrintJfrMetricsMerged(List<Path> jfrFiles, String modelStem, String modelFilename) {
		try {
			List<Path> existing = jfrFiles.stream().filter(java.nio.file.Files::exists).toList();
			print("\n" + Color.CYAN_BOLD + "  ┌─────────────────────────────────────────────────┐");
			print("  │              JFR Metrics Summary                │");
			print("  └─────────────────────────────────────────────────┘" + Color.RESET);
			print(Color.GREEN + "  ✔ Metrics written → target/metrics/metrics.json" + Color.RESET);
			for (Path f : existing) {
				print(Color.DIM + "  JFR file         → " + f.toAbsolutePath() + Color.RESET);
				String json = MetricsMain.extractToJson(f, modelStem, modelFilename);
				print(Color.DIM + json + Color.RESET);
			}
			print("");
		} catch (Exception e) {
			print(Color.RED + "  ✘ Could not extract JFR metrics: " + e.getMessage() + Color.RESET);
		}
	}

	private static Duration parseJfrDuration(String s) {
		if (s == null || s.isBlank())
			return Duration.ofMinutes(5);
		s = s.trim().toLowerCase();
		if (s.endsWith("h"))
			return Duration.ofHours(Long.parseLong(s.substring(0, s.length() - 1)));
		if (s.endsWith("m"))
			return Duration.ofMinutes(Long.parseLong(s.substring(0, s.length() - 1)));
		if (s.endsWith("s"))
			return Duration.ofSeconds(Long.parseLong(s.substring(0, s.length() - 1)));
		return Duration.ofSeconds(Long.parseLong(s));
	}

	/**
	 * Calls {@link MetricsMain#extractToJson} on the finished JFR file, then
	 * prints the resulting JSON to the REPL console inside a highlighted box.
	 */
	private static void extractAndPrintJfrMetrics(Path jfrFile, String modelStem, String modelFilename) {
		try {
			print("\n" + Color.CYAN_BOLD + "  ┌─────────────────────────────────────────────────┐");
			print("  │              JFR Metrics Summary                │");
			print("  └─────────────────────────────────────────────────┘" + Color.RESET);
			String json = MetricsMain.extractToJson(jfrFile, modelStem, modelFilename);
			print(Color.DIM + json + Color.RESET);
			print(Color.GREEN + "  ✔ Metrics written → target/metrics/metrics.json" + Color.RESET);
			print(Color.DIM + "  JFR file         → " + jfrFile.toAbsolutePath() + Color.RESET + "\n");
		} catch (Exception e) {
			print(Color.RED + "  ✘ Could not extract JFR metrics: " + e.getMessage() + Color.RESET);
		}
	}

	// ── Local mode (unchanged from original) ──────────────────────────────────

	private static void runLocalRepl() throws Exception {
		print(Color.CYAN + "▶ Starting local in-process " + nodeCount + "-node pipeline..." + Color.RESET);

		LlamaConfig config;
		Tokenizer tokenizer;
		try (GgufReader reader = GgufReader.open(Path.of(modelPath))) {
			config = LlamaConfig.from(reader);
			tokenizer = GgufTokenizer.load(reader);
		}

		long vramPerLayerBytes = estimateVramPerLayer(config.hiddenDim());
		long nodeVramBytes = config.numLayers() * vramPerLayerBytes * 2;

		List<NodeDescriptor> nodes = new ArrayList<>();
		for (int i = 0; i < nodeCount; i++) {
			nodes.add(new NodeDescriptor("node-" + i, "localhost", 9092 + i, nodeVramBytes, nodeVramBytes,
					NodeStatus.READY, 1.0, Instant.now(), Instant.now()));
		}

		ShardMap shardMap = ShardPlanner.create().plan("model", config.numLayers(), vramPerLayerBytes, nodes);

		List<ForwardPassHandler> handlers = new ArrayList<>();
		GpuContext gpuCtx = prepareGpuContext();
		// One CudaMatVec per process — shares the same GpuContext / cuBLAS handle across shards.
		CudaMatVec cudaMv = (gpuCtx != null) ? new CudaMatVec(gpuCtx) : null;
		for (var assignment : shardMap.assignments()) {
			var context = ShardContext.from(assignment, config.vocabSize(), config.hiddenDim(), config.numHeads());
			if (cudaMv != null)
				handlers.add(ForwardPassHandlerLoader.load(Path.of(modelPath), context, cudaMv));
			else
				handlers.add(ForwardPassHandlerLoader.load(Path.of(modelPath), context));
		}

		var pipeline = LocalInferencePipeline.from(shardMap, new ArrayList<>(handlers), config.vocabSize(),
				config.hiddenDim(), config.numHeads());
		var kvCache = new KVCacheManager(new GpuKVCache(512L * 1024 * 1024), new CpuKVCache(4096));
		var loop = new GenerationLoop(tokenizer, Sampler.create(), pipeline, kvCache);

		startRepl(loop, tokenizer);
	}

	private static GpuContext prepareGpuContext() {
		GpuContext gpuCtx = null;
		if (useGpu && CudaAvailability.isAvailable())
			gpuCtx = GpuContext.init(0);
		else
			return null;
		final GpuContext gpuCtxRef = gpuCtx;
		Runtime.getRuntime().addShutdownHook(Thread.ofVirtual().unstarted(() -> {
			if (gpuCtxRef != null)
				gpuCtxRef.close();
		}));
		return gpuCtx;
	}
	// ── Cluster mode (unchanged from original) ─────────────────────────────────

	private static void runClusterRepl() throws Exception {
		String modeLabel = pType == ParallelismType.TENSOR ? "tensor-parallel" : "pipeline-parallel";
		print(Color.CYAN_BOLD + "▶ Starting 3-node " + modeLabel + " cluster (forked JVMs)..." + Color.RESET);

		int totalLayers;
		int numHeads;
		int vocabSize;
		try (GgufReader cfgReader = GgufReader.open(Path.of(modelPath))) {
			LlamaConfig cfg = LlamaConfig.from(cfgReader);
			totalLayers = cfg.numLayers();
			numHeads = cfg.numHeads();
			vocabSize = cfg.vocabSize();
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

		var pipeline = (pType == ParallelismType.TENSOR) ? harness.pipeline()
				: new ProcessPipelineClient(harness.nodeAddresses(), vocabSize, dtype);

		Tokenizer tokenizer;
		try (GgufReader reader = GgufReader.open(Path.of(modelPath))) {
			tokenizer = GgufTokenizer.load(reader);
		}

		var kvCache = new KVCacheManager(new GpuKVCache(512L * 1024 * 1024), new CpuKVCache(4096));
		var loop = new GenerationLoop(tokenizer, Sampler.create(), pipeline, kvCache);

		startRepl(loop, tokenizer);
	}

	// ── Standard REPL loop ────────────────────────────────────────────────────

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
			InferenceRequest request = InferenceRequest.ofSession(history.sessionId(), modelType, history.getMessages(),
					params, RequestPriority.NORMAL);

			System.out.print(Color.GREEN_BOLD + "bot> " + Color.RESET);
			System.out.flush();

			long start = System.currentTimeMillis();
			var consumer = streamingConsumer(verbose);
			GenerationResult result = loop.generate(request, consumer);
			history.addAssistant(result.text());

			long elapsed = System.currentTimeMillis() - start;
			System.out.println();
			System.out.printf(Color.GREEN + "     [%d tokens · %d ms · %s]" + Color.RESET + "%n",
					result.generatedTokens(), elapsed, dtype);
			System.out.println();
		}

		loop.evictSession(history.sessionId());
		print(Color.YELLOW + "\nbye." + Color.RESET);
		System.exit(0);
	}

	// ── Helpers ───────────────────────────────────────────────────────────────

	private static TokenConsumer streamingConsumer(boolean verbose) {
		return new TokenConsumer() {
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
	}

	private static void banner() {
		System.out.println(String.format("  %sJuno interactive console  ·  model: %s%s%n", Color.YELLOW_BOLD_BRIGHT,
				Path.of(modelPath).getFileName(), Color.RESET));
		System.out.println(Color.RED_BOLD + "░▀▀█" + Color.GREEN_BOLD + "░█░█" + Color.RESET);
		System.out.println(Color.RED + "░░░█" + Color.GREEN + "░█░█" + Color.RESET);
		System.out.println(Color.RED + "░▀▀░" + Color.GREEN + "░▀▀▀" + Color.RESET);
		System.out.println(Color.BLUE_BOLD + "░█▀█" + Color.YELLOW_BOLD + "░█▀█" + Color.RESET);
		System.out.println(Color.BLUE + "░█░█" + Color.YELLOW + "░█░█" + Color.RESET);
		System.out.println(Color.BLUE + "░▀░▀" + Color.YELLOW + "░▀▀▀" + Color.RESET + "\n");

		if (loraMode) {
			System.out.println(String.format("  %s⚙ LoRA mode  ·  rank=%d  α=%.1f  lr=%s  steps=%d%s%n",
					Color.PURPLE_BOLD, loraRank, loraAlpha, loraLr, loraSteps, Color.RESET));
		} else {
			System.out.println(String.format(
					"  %sdtype=%s · byteOrder=%s · max_tokens=%d · temperature=%.2f · top_k=%d · top_p=%.2f · %s nodes=%d%s%n",
					Color.GREEN_BOLD_BRIGHT, dtype, byteOrder, maxTokens, temperature, topK, topP, localMode ? "local" : "cluster",
					nodeCount, Color.RESET));
		}
		if (jfrDuration != null) {
			System.out.println(
					String.format("  %s⏱ JFR active · duration=%s%s%n", Color.YELLOW, jfrDuration, Color.RESET));
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
		default -> ParallelismType.PIPELINE;
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

	private static double parseDouble(String s, double def) {
		try {
			return Double.parseDouble(s);
		} catch (NumberFormatException e) {
			return def;
		}
	}

	private static long estimateVramPerLayer(int hiddenDim) {
		long params = 4L * hiddenDim * hiddenDim;
		return (long) (params * 2.0);
	}
}