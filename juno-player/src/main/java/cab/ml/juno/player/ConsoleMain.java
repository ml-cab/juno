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
import java.util.Locale;
import java.util.Random;
import java.util.logging.Logger;

import jdk.jfr.Configuration;
import jdk.jfr.Recording;
import jdk.jfr.RecordingState;

import cab.ml.juno.metrics.MetricsMain;

import cab.ml.juno.coordinator.AiDisclosure;
import cab.ml.juno.coordinator.GenerationLoop;
import cab.ml.juno.coordinator.GenerationResult;
import cab.ml.juno.health.HealthMain;
import cab.ml.juno.health.HealthReporter;
import cab.ml.juno.health.HealthThresholds;
import cab.ml.juno.coordinator.InferenceRequest;
import cab.ml.juno.coordinator.RequestPriority;
import cab.ml.juno.coordinator.TokenConsumer;
import cab.ml.juno.health.HealthReporter;
import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.lora.LoraAdamOptimizer;
import cab.ml.juno.lora.LoraAdapterSet;
import cab.ml.juno.metrics.MetricsMain;
import cab.ml.juno.node.ActivationDtype;
import cab.ml.juno.node.CudaAvailability;
import cab.ml.juno.node.RocmAvailability;
import cab.ml.juno.node.ForwardPassHandler;
import cab.ml.juno.node.CudaMatVec;
import cab.ml.juno.node.MatVec;
import cab.ml.juno.node.ForwardPassHandlerLoader;
import cab.ml.juno.node.DoraInitializer;
import cab.ml.juno.node.GgufReader;
import cab.ml.juno.node.GpuContext;

import cab.ml.juno.node.LlamaConfig;
import cab.ml.juno.node.LocalInferencePipeline;
import cab.ml.juno.lora.LoraAdamOptimizer;
import cab.ml.juno.lora.LoraAdapterSet;
import cab.ml.juno.lora.LoraGradients;
import cab.ml.juno.node.LoraCheckpointEvent;
import cab.ml.juno.node.LoraGradientBatch;
import cab.ml.juno.node.LoraGradientResult;
import cab.ml.juno.node.LoraInitializer;
import cab.ml.juno.node.LoraMetricsIdentity;
import cab.ml.juno.node.LoraPlaybackEvent;
import cab.ml.juno.node.LoraProjection;
import cab.ml.juno.node.LoraTrainEvent;
import cab.ml.juno.node.LoraTrainingHandler;
import cab.ml.juno.node.LoraTrainingHandlerFactory;
import cab.ml.juno.node.MatVec;
import cab.ml.juno.node.QaLoraInitializer;
import cab.ml.juno.node.ShardContext;
import cab.ml.juno.registry.ModelDescriptor;
import cab.ml.juno.registry.ModelRegistry;
import cab.ml.juno.registry.ModelStatus;
import cab.ml.juno.registry.ParallelismType;
import cab.ml.juno.registry.QuantizationType;
import cab.ml.juno.registry.ShardAssignment;
import cab.ml.juno.registry.ShardMap;
import cab.ml.juno.registry.ShardPlanner;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.GgufTokenizer;
import cab.ml.juno.tokenizer.Tokenizer;
import cab.ml.juno.vision.LlavaHandlerFactory;
import jdk.jfr.Configuration;
import jdk.jfr.Recording;
import jdk.jfr.RecordingState;

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
 * Command-line arguments: --model-path PATH Path to GGUF file (required)
 * --mmproj-path PATH Separate mmproj GGUF holding the CLIP vision encoder
 * (required for /v1/vision/chat on real LLaVA/Qwen-VL/SmolVLM releases) --cpu
 * Force computation on CPU --dtype FLOAT32|FLOAT16 Activation wire format
 * (default: FLOAT16) --max-tokens N Max generated tokens (default: 200)
 * --temperature F Sampling temperature (default: 0.7) --local Use in-process
 * nodes instead of forking --nodes N Number of in-process nodes (default: 3)
 * --lora Enable LoRA fine-tuning mode (forces --local --nodes 1) --lora-path
 * PATH .lora checkpoint file (default: <model>.lora) --lora-rank N LoRA rank
 * (default: 8) --lora-alpha F LoRA alpha scaling (default: same as rank)
 * --lora-lr F Adam learning rate for LoRA (default: 1e-4) --lora-max-iters N
 * Max training passes per /train (default: 50) --verbose Show more logging
 * --help Show this help
 */
public final class ConsoleMain {

	@SuppressWarnings("unused")
	private static final Logger log = Logger.getLogger(ConsoleMain.class.getName());

	/**
	 * Configures java.util.logging verbosity for this JVM.
	 *
	 * Must be called from {@link #main(String[])} AFTER {@link #parseArgs(String[])}
	 * has run, never from a static initializer. A static initializer executes at
	 * class-load time, which is before main()'s body — including before
	 * parseArgs() has parsed --verbose and before JUNO_VERBOSE would be set as a
	 * system property — so it would always observe verbose=false regardless of
	 * the CLI flag. That ordering bug previously made --verbose a no-op in
	 * --local mode (single in-process JVM), while cluster mode happened to work
	 * because verbosity there is read by {@link ClusterHarness} at fork time,
	 * well after parseArgs() has already run.
	 */
	private static void configureLogging() {
		boolean effectiveVerbose = verbose || Boolean.getBoolean("JUNO_VERBOSE")
				|| "true".equalsIgnoreCase(System.getenv("JUNO_VERBOSE"));
		if (!effectiveVerbose) {
			java.util.logging.LogManager.getLogManager().reset();
			java.util.logging.Logger.getLogger("").setLevel(java.util.logging.Level.OFF);
			for (String ns : new String[] { "io.grpc", "io.netty", "cab.ml.juno", "com.google", "org.slf4j", "" }) {
				java.util.logging.Logger.getLogger(ns).setLevel(java.util.logging.Level.OFF);
			}
		}
	}

	// ── Standard arguments ────────────────────────────────────────────────────
	private static String modelPath = null;
	/**
	 * Path to a separate mmproj GGUF holding the CLIP vision encoder (see
	 * {@link cab.ml.juno.vision.VisionModelPaths}). Real-world multimodal GGUF
	 * releases (LLaVA, Qwen-VL, SmolVLM, ...) never merge the vision encoder
	 * into the base LLM file; without this path {@code --api-port} vision
	 * routes are never registered for those models, even though
	 * {@code --model-path} points at a genuine I2T model.
	 */
	private static String mmprojPath = null;
	private static ActivationDtype dtype = ActivationDtype.FLOAT16;
	private static int maxTokens = 200;
	private static float temperature = 0.7f;
	private static int topK = 50;
	private static float topP = 0.9f;
	private static boolean localMode = false;
	private static int nodeCount = 3;
	private static boolean verbose = false;
	private static boolean help = false;
	private static ParallelismType pType = ParallelismType.PIPELINE;
	private static String jfrDuration = null;
	// ── Health server flag ────────────────────────────────────────────────────
	/** When true, start a health sidecar alongside the normal run mode. */
	private static boolean healthMode = false;
	private static int healthPort = cab.ml.juno.health.HealthMain.DEFAULT_PORT;
	/**
	 * Active reporters — wired from runLocalRepl(); used to record per-inference
	 * latency.
	 */
	private static final java.util.List<HealthReporter> activeReporters = new java.util.ArrayList<>();
	/** Optional local REST API port (OpenAI-compatible endpoint included). */
	private static int apiPort = -1;
	/** Prefill strategy: batched (default) or single (legacy sequential loop). */
	private static cab.ml.juno.coordinator.PrefillMode prefillMode = cab.ml.juno.coordinator.PrefillMode.BATCHED;
	// ── Byte-order argument ───────────────────────────────────────────────────
	/** Activation codec byte order: {@code "BE"} (default) or {@code "LE"}. */
	private static String byteOrder = "BE";
	// ── GPU arguments ─────────────────────────────────────────────────────────
	private static boolean useGpu = true; // use CPU
	// ── LoRA arguments ────────────────────────────────────────────────────────
	private static boolean loraMode = false;
	private static String loraPath = null; // auto-derived if null
	private static String loraPlayPath = null; // --lora-play: apply .lora at inference in non-lora modes
	private static int loraRank = 8;
	private static float loraAlpha = -1f; // sentinel: default to loraRank
	private static double loraLr = 1e-4;
	private static int loraMaxIters = 50;
	private static int loraMaxItersQa = 50;
	private static float loraLossTargetText = 1.8f;
	private static float loraLossTargetQa = 1.2f;
	private static float loraEarlyStop = 0.25f; // stop training when loss drops below this
	private static String loraTargets = "qv";
	private static int loraGradientAccumulation = 1;
	private static float loraMaxGradNorm = 1.0f;
	private static String loraLrSchedule = "constant";
	private static int loraWarmupSteps = 0;
	private static double loraMinLr = 0.0;
	private static double loraWeightDecay = 0.01;
	private static double loraPlusRatio = 1.0;
	private static float loraDropout = 0f;
	private static long loraSeed = 42L;
	private static float loraValidationSplit = 0f;
	private static int loraValidationPatience = 0;
	private static float loraValidationMinDelta = 0f;
	private static String loraAdapterMode = "lora";
	private static String loraScaling = "standard";
	private static String loraInit = "kaiming-uniform";
	private static int loraChunkTokens = LoraCorpusLimit.DEFAULT_CHUNK_TOKENS;
	private static int loraMaxTrainTokens = 0;
	private static String loraTrainDevice = cab.ml.juno.node.LoraTrainDevice.AUTO;
	private static int loraMicrobatch = cab.ml.juno.node.LoraMicrobatch.DEFAULT;
	/** Resolved after handler open: cpu|cuda|rocm (for JFR identity). */
	private static String loraResolvedTrainDevice = "cpu";
	private static LlamaConfig loraModelConfig; // set in runLoraRepl for /reset

	/** Active programmatic JFR session (local / lora); stopped and extracted on exit. */
	private static volatile Recording activeJfrRecording;
	private static volatile Path activeJfrFile;
	private static volatile String activeJfrModelStem;
	private static volatile String activeJfrModelFilename;
	private static final Object jfrLock = new Object();
	private static volatile boolean jfrExtracted;
	private static LoraTrainingConfig currentTrainingConfig() {
		LoraCliOptions o = new LoraCliOptions();
		o.rank = loraRank;
		o.alpha = loraAlpha;
		o.lr = loraLr;
		o.targets = loraTargets;
		o.gradientAccumulation = loraGradientAccumulation;
		o.maxGradNorm = loraMaxGradNorm;
		o.lrSchedule = loraLrSchedule;
		o.warmupSteps = loraWarmupSteps;
		o.minLr = loraMinLr;
		o.weightDecay = loraWeightDecay;
		o.loraPlusRatio = loraPlusRatio;
		o.dropout = loraDropout;
		o.seed = loraSeed;
		o.validationSplit = loraValidationSplit;
		o.validationPatience = loraValidationPatience;
		o.validationMinDelta = loraValidationMinDelta;
		o.mode = loraAdapterMode;
		o.scaling = loraScaling;
		o.init = loraInit;
		o.chunkTokens = loraChunkTokens;
		o.maxTrainTokens = loraMaxTrainTokens;
		o.architecture = loraModelConfig != null ? loraModelConfig.architecture() : "";
		o.trainDevice = loraResolvedTrainDevice != null && !loraResolvedTrainDevice.isBlank()
				? loraResolvedTrainDevice
				: loraTrainDevice;
		o.microbatch = loraMicrobatch;
		return o.toTrainingConfig();
	}

	private static SamplingParams samplingParamsFromCli() {
		SamplingParams params = SamplingParams.defaults().withMaxTokens(maxTokens).withTemperature(temperature)
				.withTopK(topK).withTopP(topP);
		return temperature < 1e-6f ? params.withGreedy(true) : params;
	}

	public static void main(String[] args) throws Exception {
		AnsiSupport.enable();
		applyLoraEnvDefaults();
		parseArgs(args);
		if (help) {
			printHelp();
			System.exit(0);
		}
		configureLogging();

		// ── Health sidecar — start in background then continue normally ──────
		if (healthMode) {
			cab.ml.juno.health.HealthMain.startBackground(
				healthPort, cab.ml.juno.health.HealthThresholds.defaults());
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
		if (mmprojPath != null && !Path.of(mmprojPath).toFile().exists()) {
			System.err.println("ERROR: mmproj file not found: " + mmprojPath);
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
		// For local, cluster, and lora modes, ConsoleMain manages JFR programmatically via
		// startLocalJfr() / startClusterJfr() / startLoraJfr() — no JVM flag is involved.
		if (jfrDuration != null && !localMode && !loraMode)
			System.setProperty("juno.jfr.duration", jfrDuration);

		banner();

		if (loraMode && jfrDuration != null) {
			startLoraJfr();
		} else if (loraMode) {
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

	private static void applyLoraEnvDefaults() {
		LoraCliOptions env = LoraCliOptions.fromEnvDefaults();
		loraTargets = env.targets;
		loraGradientAccumulation = env.gradientAccumulation;
		loraMaxGradNorm = env.maxGradNorm;
		loraLrSchedule = env.lrSchedule;
		loraWarmupSteps = env.warmupSteps;
		loraMinLr = env.minLr;
		loraWeightDecay = env.weightDecay;
		loraPlusRatio = env.loraPlusRatio;
		loraDropout = env.dropout;
		loraSeed = env.seed;
		loraValidationSplit = env.validationSplit;
		loraValidationPatience = env.validationPatience;
		loraValidationMinDelta = env.validationMinDelta;
		loraAdapterMode = env.mode;
		loraScaling = env.scaling;
		loraInit = env.init;
		loraChunkTokens = env.chunkTokens;
		loraMaxTrainTokens = env.maxTrainTokens;
		loraTrainDevice = env.trainDevice;
		loraMicrobatch = env.microbatch;
	}

	private static void parseArgs(String[] args) {
		for (int i = 0; i < args.length; i++) {
			switch (args[i]) {
			case "--model-path":
				if (i + 1 < args.length)
					modelPath = args[++i];
				break;
			case "--mmproj-path":
				if (i + 1 < args.length)
					mmprojPath = args[++i];
				break;
			case "--prefill":
				if (i + 1 < args.length)
					prefillMode = parsePrefillMode(args[++i]);
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
					topK = parseInt(args[++i], 50);
				break;
			case "--top-p":
				if (i + 1 < args.length)
					topP = parseFloat(args[++i], 0.9f);
				break;
			case "--temperature":
				if (i + 1 < args.length)
					temperature = parseFloat(args[++i], 0.7f);
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
			case "--lora-play":
				if (i + 1 < args.length)
					loraPlayPath = args[++i];
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
			case "--lora-max-iters":
				if (i + 1 < args.length) {
					int n = parseInt(args[++i], 50);
					loraMaxIters = n;
					loraMaxItersQa = n;
				}
				break;
			case "--lora-loss-target-text":
				if (i + 1 < args.length)
					loraLossTargetText = parseFloat(args[++i], 1.8f);
				break;
			case "--lora-loss-target-qa":
				if (i + 1 < args.length)
					loraLossTargetQa = parseFloat(args[++i], 1.2f);
				break;
			case "--lora-steps":
				if (i + 1 < args.length)
					loraMaxIters = parseInt(args[++i], 50);
				break;
			case "--lora-steps-qa":
				if (i + 1 < args.length)
					loraMaxItersQa = parseInt(args[++i], 50);
				break;
			case "--lora-early-stop":
				if (i + 1 < args.length)
					loraEarlyStop = parseFloat(args[++i], 0.25f);
				break;
			case "--lora-targets":
				if (i + 1 < args.length) {
					loraTargets = args[++i];
					LoraProjection.parseTargets(loraTargets); // validate
				}
				break;
			case "--lora-gradient-accumulation":
				if (i + 1 < args.length) {
					loraGradientAccumulation = parseInt(args[++i], 1);
					if (loraGradientAccumulation < 1)
						throw new IllegalArgumentException("--lora-gradient-accumulation must be >= 1");
				}
				break;
			case "--lora-max-grad-norm":
				if (i + 1 < args.length) {
					loraMaxGradNorm = parseFloat(args[++i], 1.0f);
					if (loraMaxGradNorm < 0f)
						throw new IllegalArgumentException("--lora-max-grad-norm must be >= 0");
				}
				break;
			case "--lora-lr-schedule":
				if (i + 1 < args.length) {
					loraLrSchedule = args[++i];
					currentTrainingConfig(); // validate
				}
				break;
			case "--lora-warmup-steps":
				if (i + 1 < args.length)
					loraWarmupSteps = parseInt(args[++i], 0);
				break;
			case "--lora-min-lr":
				if (i + 1 < args.length)
					loraMinLr = parseDouble(args[++i], 0.0);
				break;
			case "--lora-weight-decay":
				if (i + 1 < args.length)
					loraWeightDecay = parseDouble(args[++i], 0.01);
				break;
			case "--lora-plus-ratio":
				if (i + 1 < args.length)
					loraPlusRatio = parseDouble(args[++i], 1.0);
				break;
			case "--lora-dropout":
				if (i + 1 < args.length)
					loraDropout = parseFloat(args[++i], 0f);
				break;
			case "--lora-seed":
				if (i + 1 < args.length)
					loraSeed = Long.parseLong(args[++i]);
				break;
			case "--lora-validation-split":
				if (i + 1 < args.length)
					loraValidationSplit = parseFloat(args[++i], 0f);
				break;
			case "--lora-validation-patience":
				if (i + 1 < args.length)
					loraValidationPatience = parseInt(args[++i], 0);
				break;
			case "--lora-validation-min-delta":
				if (i + 1 < args.length)
					loraValidationMinDelta = parseFloat(args[++i], 0f);
				break;
			case "--lora-mode":
				if (i + 1 < args.length)
					loraAdapterMode = args[++i];
				break;
			case "--lora-scaling":
				if (i + 1 < args.length)
					loraScaling = args[++i];
				break;
			case "--lora-init":
				if (i + 1 < args.length)
					loraInit = args[++i];
				break;
			case "--lora-chunk-tokens":
				if (i + 1 < args.length) {
					loraChunkTokens = parseInt(args[++i], LoraCorpusLimit.DEFAULT_CHUNK_TOKENS);
					LoraCorpusLimit.validateChunkTokens(loraChunkTokens);
				}
				break;
			case "--lora-max-train-tokens":
				if (i + 1 < args.length) {
					loraMaxTrainTokens = parseInt(args[++i], 0);
					LoraCorpusLimit.validateMaxTrainTokens(loraMaxTrainTokens);
				}
				break;
			case "--lora-train-device":
				if (i + 1 < args.length)
					loraTrainDevice = cab.ml.juno.node.LoraTrainDevice.normalize(args[++i]);
				break;
			case "--lora-microbatch":
				if (i + 1 < args.length)
					loraMicrobatch = cab.ml.juno.node.LoraMicrobatch.normalize(args[++i]);
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
			case "--health":
				healthMode = true;
				break;
			case "--api-port":
				if (i + 1 < args.length)
					apiPort = parseInt(args[++i], -1);
				break;
			default:
				System.err.println("Unknown option: " + args[i]);
				help = true;
				return;
			}
		}
	}

	private static void printHelp() {
		System.out.println();
		System.out.println("Usage: java -jar juno-player.jar [options]");
		System.out.println();
		System.out.println("Required:");
		System.out.println("  --model-path PATH          Path to GGUF model file");
		System.out.println();
		System.out.println("Vision (image-to-text) models:");
		System.out.println("  --mmproj-path PATH         Path to a separate mmproj GGUF holding the CLIP");
		System.out.println("                             vision encoder. Required for /v1/vision/chat —");
		System.out.println("                             real LLaVA/Qwen-VL/SmolVLM GGUF releases keep the");
		System.out.println("                             vision encoder in a separate file from the base LLM.");
		System.out.println();
		System.out.println("Prefill strategy:");
		System.out.println("  --prefill single|batched   Prefill strategy (default: batched)");
		System.out.println("                             batched: windowed GEMM — fixes vision 10-min stall");
		System.out.println("                             single:  original per-token loop (escape hatch)");
		System.out.println();
		System.out.println("Inference options:");
		System.out.println("  --gpu                      Use GPU (default, no need to set)");
		System.out.println("  --cpu                      Force to use CPU");
		System.out.println("  --pType pipeline|tensor    Parallelism type (default: pipeline)");
		System.out.println("  --dtype FLOAT32|FLOAT16    Activation wire format (default: FLOAT16)");
		System.out.println("  --max-tokens N             Max generated tokens (default: 200)");
		System.out.println("  --temperature F            Sampling temperature (default: 0.7)");
		System.out.println("  --top-k N                  Top-K sampling cutoff (default: 50)");
		System.out.println("  --top-p F                  Nucleus sampling top-p (default: 0.9)");
		System.out.println("  --byteOrder BE|LE          Activation codec byte order (default: BE)");
		System.out.println("                             BE = big-endian (default, hardware-validated)");
		System.out.println("                             LE = little-endian (native x86 order)");
		System.out.println("  --local                    Use in-process nodes (no forking)");
		System.out.println("  --nodes N                  Number of in-process nodes (default: 3)");
		System.out.println();
		System.out.println("LoRA fine-tuning (forces --local --nodes 1):");
		System.out.println("  --lora                     Enable LoRA fine-tuning mode");
		System.out.println("  --lora-path PATH           Adapter checkpoint file (default: <model>.lora)");
		System.out.println("  --lora-play PATH           Apply a .lora file at inference in local/cluster mode (read-only, no training)");
		System.out.println("  --lora-rank N              Low-rank bottleneck dimension (default: 8)");
		System.out.println("  --lora-alpha F             Scale factor alpha (default: same as rank)");
		System.out.println("  --lora-mode lora|dora|qa-lora  Adapter algorithm (default: lora)");
		System.out.println("  --lora-group-width N   QA-LoRA group width (0=auto from Q4_K/Q5_K=32, Q6_K=16)");
		System.out.println("  --lora-merge f32-preserve|source-type-projected|sidecar-only");
		System.out.println("  --lora-scaling standard|rslora  Scale formula (default: standard)");
		System.out.println("  --lora-init kaiming-uniform|legacy-normal  A init (default: kaiming-uniform)");
		System.out.println("  --lora-lr F                Adam learning rate (default: 1e-4)");
		System.out.println("  --lora-max-iters N         Max training passes per /train (default: 50)");
		System.out.println("  --lora-loss-target-text F  Stop /train when loss <= F (default: 1.8)");
		System.out.println("  --lora-loss-target-qa F    Stop /train-qa when loss <= F (default: 1.2)");
		System.out.println();
		System.out.println("  LoRA REPL commands:");
		System.out.println("    /train <text>            Fine-tune on inline text");
		System.out.println("    /train-file <path>       Fine-tune on a text file (default chunk 32; recommend 128)");
		System.out.println("    /train-qa <q> A: <a>     Fine-tune on one Q&A fact (chat template)");
		System.out.println("    /train-file-qa <path.json>  Fine-tune on JSON array of {\"Q\",\"A\"} pairs");
		System.out.println("    /save                    Save adapter checkpoint to --lora-path");
		System.out.println("    /reset                   Reinitialize adapters and delete .lora checkpoint");
		System.out.println("    /status                  Show adapter info and training stats");
		System.out.println("    /merge-hint              Explain how to bake LoRA into GGUF weights");
		System.out.println("    Regular chat input       Inference with current adapter applied");
		System.out.println();
		System.out.println("  LoRA training controls:");
		System.out.println("  --lora-max-iters N         Max training passes (default: 50)");
		System.out.println("  --lora-loss-target-text F  /train loss target (default: 1.8)");
		System.out.println("  --lora-loss-target-qa F    /train-qa loss target (default: 1.2)");
		System.out.println("  --lora-steps N             Alias for --lora-max-iters (/train cap)");
		System.out.println("  --lora-steps-qa N        Max passes for /train-qa (default: 50)");
		System.out.println("  --lora-early-stop F      Overfit guard: stop when loss < F (default: 0.25).");
		System.out.println("                           Prevents catastrophic overfitting. Set 0 to disable.");
		System.out.println("  --lora-targets SPEC      qv | all | comma keys (default: qv)");
		System.out.println("  --lora-gradient-accumulation N  Chunks per optimizer update (default: 1)");
		System.out.println("  --lora-max-grad-norm F   Global grad clip; 0=off (default: 1.0)");
		System.out.println("  --lora-lr-schedule M     constant|cosine (default: constant)");
		System.out.println("  --lora-warmup-steps N    Cosine warmup updates (default: 0)");
		System.out.println("  --lora-min-lr F          Cosine floor LR (default: 0)");
		System.out.println("  --lora-weight-decay F    AdamW A-only decay (default: 0.01)");
		System.out.println("  --lora-plus-ratio F      B/A LR ratio; 1=off (default: 1.0)");
		System.out.println("  --lora-dropout F         Train-only dropout [0,1) (default: 0)");
		System.out.println("  --lora-seed N            RNG seed (default: 42)");
		System.out.println("  --lora-validation-split F  Held-out unit fraction (default: 0)");
		System.out.println("  --lora-validation-patience N  Early-stop patience (default: 0=off)");
		System.out.println("  --lora-validation-min-delta F  Min val improvement (default: 0)");
		System.out.println("  --lora-chunk-tokens N     Truncated-BPTT window (default: 32; recommend 128 for /train-file)");
		System.out.println("  --lora-max-train-tokens N Cap supervised tokens per train; 0=unlimited (default: 0)");
		System.out.println("  --lora-train-device M    auto|gpu|cpu (default: auto; gpu fails closed if unavailable)");
		System.out.println("  --lora-microbatch N     Frozen GEMM width 1..128 (default: 8; 1=FP16 sequential)");
		System.out.println();
		System.out.println("Other:");
		System.out.println("  --health                   Start the standalone health-monitor HTTP server");
		System.out.println("  --api-port N               Start REST API (local/cluster: chat; lora: train-file-qa)");
		System.out.println("                             lora: POST /v1/lora/train-file-qa , POST /v1/lora/save");
		System.out.println("    --port N                   Health listen port (default: 8081)");
		System.out.println("    --stale-ms N               Node stale threshold in ms (default: 15000)");
		System.out.println("    --warn F                   VRAM warning threshold 0.0-1.0 (default: 0.90)");
		System.out.println("    --critical F               VRAM critical threshold 0.0-1.0 (default: 0.98)");
		System.out.println("  --jfr DURATION             Java Flight Recording duration (e.g. 5m)");
		System.out.println("                             Programmatic recording; on exit extracts");
		System.out.println("                             target/metrics/metrics.json (local/lora/cluster)");
		System.out.println("  --verbose, -v              Full LoRA/train traces (default: progress bar)");
		System.out.println("  --help, -h                 Show this help");
	}

	// ── LoRA mode ─────────────────────────────────────────────────────────────

	/**
	 * LoRA fine-tuning REPL. Runs a single in-process LoraTrainingHandler that
	 * serves both inference (with LoRA delta) and training (/train commands).
	 *
	 * Adapters are persisted in a separate .lora file alongside the model. The base
	 * GGUF is never modified.
	 */
	private static void runLoraRepl() throws Exception {
		print(Color.DIM + "  Adapter file: " + loraPath + Color.RESET);

		LlamaConfig config;
		Tokenizer tokenizer;
		LoraAdapterSet adapters;
		Path adapterFile = Path.of(loraPath);
		LoraTrainingConfig trainCfg = currentTrainingConfig();
		try (GgufReader reader = GgufReader.open(Path.of(modelPath))) {
			config = LlamaConfig.from(reader);
			loraModelConfig = config;
			tokenizer = GgufTokenizer.load(reader);
			if (Files.exists(adapterFile)) {
				long tLoad = System.currentTimeMillis();
				adapters = LoraAdapterSet.load(adapterFile);
				long loadMs = System.currentTimeMillis() - tLoad;
				LoraInitializer.validate(adapters, config);
				DoraInitializer.verifyFingerprints(reader, adapters);
				DoraInitializer.attachMissingDoraState(reader, config, adapters);
				QaLoraInitializer.verifyFingerprints(reader, adapters);
				commitCheckpointEvent(adapters, "load", 2, adapters.size(), loadMs, Files.size(adapterFile));
				print(Color.GREEN + "  ✔ Loaded checkpoint: " + adapters.size() + " adapters from " + loraPath
						+ Color.RESET);
				if (!adapters.all().isEmpty()) {
					var sample = adapters.all().get(0);
					print(Color.DIM + "  checkpoint mode=" + sample.mode + " scaling=" + sample.scaling + " init="
							+ sample.initialization + " alpha=" + sample.alpha + " scale=" + sample.scale
							+ Color.RESET);
				} else if (!adapters.allQa().isEmpty()) {
					var sample = adapters.allQa().get(0);
					print(Color.DIM + "  checkpoint mode=QA_LORA groupWidth=" + sample.groupWidth + " groups="
							+ sample.groupCount + " alpha=" + sample.alpha + " scale=" + sample.scale + Color.RESET);
				}
				print(Color.YELLOW
						+ "  ⚠ Continuing on prior weights. If every reply is the same answer, run /reset before /train-qa."
						+ Color.RESET);
			} else if (trainCfg.mode() == cab.ml.juno.lora.LoraMode.DORA) {
				adapters = DoraInitializer.create(reader, config, trainCfg.targets(), trainCfg.adapterConfig(),
						new Random(loraSeed));
				print(Color.YELLOW + "  ✦ New DoRA adapters initialised (" + adapters.size() + " total · targets="
						+ loraTargets + " · /save to persist)" + Color.RESET);
			} else if (trainCfg.mode() == cab.ml.juno.lora.LoraMode.QA_LORA) {
				adapters = QaLoraInitializer.create(reader, config, trainCfg.targets(), trainCfg.adapterConfig(),
						new Random(loraSeed), trainCfg.groupWidth(), trainCfg.mergeCapability());
				print(Color.YELLOW + "  ✦ New QA-LoRA adapters initialised (" + adapters.size()
						+ " total · targets=" + loraTargets + " · /save to persist)" + Color.RESET);
			} else {
				adapters = LoraInitializer.create(config, trainCfg.targets(), trainCfg.adapterConfig(),
						new Random(loraSeed));
				print(Color.YELLOW + "  ✦ New adapters initialised (" + adapters.size() + " total · targets="
						+ loraTargets + " scaling=" + trainCfg.scaling() + " init=" + trainCfg.initialization()
						+ " · /save to persist)" + Color.RESET);
			}
		}
		loraModelConfig = config;

		// Single-node ShardContext covering the full model
		ShardAssignment assignment = new ShardAssignment("lora-node", "localhost", 0, 0, config.numLayers(), true,
				true);
		ShardMap shardMap = new ShardMap("model", config.numLayers(), List.of(assignment), Instant.now());
		ShardContext ctx = ShardContext.from(assignment, config.vocabSize(), config.hiddenDim(), config.numHeads());

		print(Color.DIM + "  Loading model weights…" + Color.RESET);
		cab.ml.juno.node.LoraTrainNotices.clear();
		cab.ml.juno.node.LoraMicrobatch.apply(loraMicrobatch);
		cab.ml.juno.node.MatVec loraBackend = cab.ml.juno.node.LoraTrainDevice.selectBackend(loraTrainDevice);
		loraResolvedTrainDevice = cab.ml.juno.node.LoraTrainDevice.labelFor(loraBackend);
		if (cab.ml.juno.node.LoraTrainDevice.AUTO.equals(loraTrainDevice)
				&& "cpu".equals(loraResolvedTrainDevice))
			cab.ml.juno.node.LoraTrainNotices.add(autoCpuFallbackNotice());
		LoraTrainingHandler handler = LoraTrainingHandlerFactory.create(Path.of(modelPath), ctx, adapters, loraBackend);
		print(Color.GREEN + "  ✔ Model loaded  (" + config + ")" + Color.RESET);
		print(Color.DIM + "  " + formatLoraTrainStatus(loraTrainDevice, loraResolvedTrainDevice,
				cab.ml.juno.node.LoraMicrobatch.current()) + Color.RESET);
		for (String notice : cab.ml.juno.node.LoraTrainNotices.drain()) {
			print(Color.YELLOW + "  ⚠ " + notice + Color.RESET);
		}
		print("");

		if (verbose) {
			String detectedModelType = ChatModelType.fromPath(modelPath);
			print(Color.DIM + "  [TRACE] model type (chat template key) : " + detectedModelType + Color.RESET);
			print(Color.DIM + "  [TRACE] model path                     : " + modelPath + Color.RESET);
			print(Color.DIM + "  [TRACE] LoRA rank=" + loraRank + "  alpha=" + loraAlpha
					+ "  lr=" + loraLr + "  targets=" + loraTargets + "  accum=" + loraGradientAccumulation
					+ "  max-grad-norm=" + loraMaxGradNorm + "  chunk-tokens=" + loraChunkTokens
					+ "  max-train-tokens=" + loraMaxTrainTokens + "  loss-target-text=" + loraLossTargetText
					+ "  loss-target-qa=" + loraLossTargetQa + "  max-iters=" + loraMaxIters
					+ "  max-iters-qa=" + loraMaxItersQa + "  early-stop=" + loraEarlyStop
					+ "  temperature=" + temperature + Color.RESET);
			print("");
		}

		// Wrap in LocalInferencePipeline for standard inference path
		var pipeline = LocalInferencePipeline.from(shardMap, List.of(handler), config.vocabSize(), config.hiddenDim(),
				config.numHeads());
		var kvCache = new KVCacheManager(new GpuKVCache(512L * 1024 * 1024), new CpuKVCache(4096));
		var loop = new GenerationLoop(tokenizer, Sampler.create(), pipeline, kvCache, prefillMode);

		LoraAdamOptimizer optimizer = LoraAdamOptimizer.defaults(loraLr);
		int[] totalStepsTrained = { 0 };
		boolean[] dirty = { false }; // unsaved changes?
		final Object trainLock = new Object();

		SamplingParams params = samplingParamsFromCli();

		ChatHistory history = new ChatHistory();

		LoraApiServer loraApi = null;
		if (apiPort > 0) {
			String modelTypeKey = ChatModelType.fromPath(modelPath);
			loraApi = new LoraApiServer(new LoraApiServer.Backend() {
				@Override
				public LoraTrainingLoop.TrainingResult trainFileQa(List<LoraQaFile.Pair> pairs) throws Exception {
					synchronized (trainLock) {
						List<LoraTrainingLoop.TrainUnit> units = new ArrayList<>();
						for (LoraQaFile.Pair pair : pairs) {
							for (var v : LoraTrainingSequences.buildQaVariants(tokenizer, pair.q(), pair.a(),
									modelTypeKey))
								units.add(new LoraTrainingLoop.TrainUnit(v.tokens(), v.lossMask()));
						}
						int stepsBefore = optimizer.step();
						LoraTrainingConfig cfg = currentTrainingConfig();
						LoraTrainingLoop.TrainingResult result = LoraTrainingLoop.train(units, cfg, adapters, optimizer,
								(tokens, mask, tctx) -> handler.computeGradients(tokens, mask, tctx),
								(tokens, mask) -> handler.evaluateLoss(tokens, mask), loraLossTargetQa, loraMaxItersQa,
								loraEarlyStop, loraChunkTokens);
						int updates = optimizer.step() - stepsBefore;
						totalStepsTrained[0] += updates;
						if (updates > 0)
							dirty[0] = true;
						return result;
					}
				}

				@Override
				public Path save() {
					synchronized (trainLock) {
						saveAdapters(adapters, adapterFile, totalStepsTrained[0]);
						dirty[0] = false;
						return adapterFile.toAbsolutePath();
					}
				}
			});
			loraApi.start(apiPort);
			print(Color.GREEN + "  LoRA API on http://localhost:" + apiPort
					+ "  (POST /v1/lora/train-file-qa , POST /v1/lora/save)" + Color.RESET);
		}

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
				synchronized (trainLock) {
					handleLoraCommand(line, adapters, optimizer, handler, tokenizer, adapterFile, totalStepsTrained,
							dirty, history, loop);
				}
				continue;
			}

			if (line.equalsIgnoreCase("exit") || line.equalsIgnoreCase("quit")) {
				if (dirty[0]) {
					System.out
							.print(Color.YELLOW + "  Unsaved adapter changes. Save before exit? [y/N] " + Color.RESET);
					System.out.flush();
					String yn = stdin.readLine();
					if (yn != null && yn.strip().equalsIgnoreCase("y")) {
						synchronized (trainLock) {
							saveAdapters(adapters, adapterFile, totalStepsTrained[0]);
							dirty[0] = false;
						}
					}
				}
				break;
			}

			// ── Regular inference ──────────────────────────────────────────
			history.addUser(line);
			String modelType = ChatModelType.fromPath(modelPath);
			// ── [TRACE] confirm the template key used for this inference request ──
			if (verbose) {
				print(Color.DIM + "  [TRACE] inference model type (chat template): " + modelType + Color.RESET);
			}
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
			System.out.printf(Color.GREEN + "     [%d tokens · %d ms · LoRA rank=%d · temperature=%.2f]" + Color.RESET
							+ "%n",
					result.generatedTokens(), elapsed, loraRank, temperature);
			System.out.println();
			activeReporters.forEach(r -> r.recordLatency(elapsed));
		}

		if (loraApi != null)
			loraApi.stop();
		loop.evictSession(history.sessionId());
		print(Color.YELLOW + "\nbye." + Color.RESET);
		stopAndExtractActiveJfr();
		System.exit(0);
	}

	private static void handleLoraCommand(String line, LoraAdapterSet adapters, LoraAdamOptimizer optimizer,
			LoraTrainingHandler handler, Tokenizer tokenizer, Path adapterFile, int[] totalSteps, boolean[] dirty,
			ChatHistory history, GenerationLoop loop) throws Exception {

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

		case "/train-file-qa" -> {
			if (parts.length < 2 || parts[1].isBlank()) {
				print(Color.RED + "  Usage: /train-file-qa <path.json>" + Color.RESET);
				print(Color.RED + "  JSON array of {\"Q\":\"...\",\"A\":\"...\"} objects" + Color.RESET);
				return;
			}
			Path p = Path.of(parts[1].strip());
			List<LoraQaFile.Pair> pairs;
			try {
				pairs = LoraQaFile.load(p);
			} catch (IllegalArgumentException e) {
				print(Color.RED + "  " + e.getMessage() + Color.RESET);
				return;
			}
			print(Color.DIM + "  Loaded: " + p.getFileName() + "  (" + pairs.size() + " Q&A pair(s))"
					+ Color.RESET);
			trainOnQaPairs(pairs, adapters, optimizer, handler, tokenizer, totalSteps, dirty,
					ChatModelType.fromPath(modelPath));
		}

		case "/save" -> saveAdapters(adapters, adapterFile, totalSteps[0]);

		case "/reset" -> {
			System.out.print(Color.YELLOW
					+ "  Reset adapters and delete checkpoint on disk? All training will be lost. [y/N] "
					+ Color.RESET);
			System.out.flush();
			String yn = new BufferedReader(new InputStreamReader(System.in)).readLine();
			if (yn != null && yn.strip().equalsIgnoreCase("y")) {
				LoraTrainingConfig cfg = currentTrainingConfig();
				LoraAdapterSet fresh;
				try (GgufReader reader = GgufReader.open(Path.of(modelPath))) {
					if (cfg.mode() == cab.ml.juno.lora.LoraMode.DORA)
						fresh = DoraInitializer.create(reader, loraModelConfig, cfg.targets(), cfg.adapterConfig(),
								new Random(loraSeed));
					else if (cfg.mode() == cab.ml.juno.lora.LoraMode.QA_LORA)
						fresh = QaLoraInitializer.create(reader, loraModelConfig, cfg.targets(), cfg.adapterConfig(),
								new Random(loraSeed), cfg.groupWidth(), cfg.mergeCapability());
					else
						fresh = LoraInitializer.create(loraModelConfig, cfg.targets(), cfg.adapterConfig(),
								new Random(loraSeed));
				}
				int n = adapters.resetFrom(fresh, new Random(loraSeed));
				optimizer.reset();
				totalSteps[0] = 0;
				// Drop prior chat turns + KV blocks; otherwise the model echoes
				// memorized answers still present in the conversation context.
				String oldSession = history.clear();
				loop.evictSession(oldSession);
				dirty[0] = false;
				boolean deleted = false;
				try {
					deleted = Files.deleteIfExists(adapterFile);
				} catch (IOException e) {
					print(Color.RED + "  ✔ Memory reset, but failed to delete " + adapterFile + ": " + e.getMessage()
							+ Color.RESET);
					print(Color.YELLOW + "  Remove the file manually or the old checkpoint will reload next time."
							+ Color.RESET);
				}
				print(Color.GREEN + "  ✔ Adapters reinitialised (" + n + " · targets=" + loraTargets + ")"
						+ Color.RESET);
				if (deleted)
					print(Color.DIM + "  Deleted checkpoint: " + adapterFile + Color.RESET);
				else if (!Files.exists(adapterFile))
					print(Color.DIM + "  No checkpoint file to delete." + Color.RESET);
				print(Color.DIM + "  Chat history cleared (fresh session). Use /save to write a new checkpoint."
						+ Color.RESET);
			}
		}

		case "/status" -> {
			long adapterBytes = adapters.all().stream().mapToLong(a -> (a.a().length + a.b().length) * 4L).sum();
			var sample = adapters.all().isEmpty() ? null : adapters.all().get(0);
			print("");
			print(Color.CYAN_BOLD + "  LoRA status" + Color.RESET);
			print("  ─────────────────────────────────");
			print("  adapters  : " + adapters.size() + "  (targets=" + loraTargets + ")");
			if (sample != null) {
				print("  mode      : " + sample.mode + "  ·  scaling=" + sample.scaling + "  ·  init="
						+ sample.initialization);
				print("  rank      : " + sample.rank);
				print("  alpha     : " + sample.alpha + "  (effective scale = " + sample.scale + ")");
			} else {
				print("  rank      : " + loraRank);
				print("  alpha     : " + loraAlpha);
			}
			print("  accum     : " + loraGradientAccumulation + "  ·  max-grad-norm=" + loraMaxGradNorm);
			print("  parameters: " + (adapterBytes / 4) + "  (" + (adapterBytes / 1024) + " KB)");
			print("  trained   : " + totalSteps[0] + " optimizer updates");
			print("  checkpoint: " + adapterFile + (dirty[0] ? "  " + Color.YELLOW + "[unsaved]" + Color.RESET : ""));
			print("  lr        : " + loraLr + "  ·  optimizer step = " + optimizer.step());
			print("");
		}

		case "/merge-hint" -> {
			print("");
			print(Color.CYAN_BOLD + "  Merging LoRA into base weights" + Color.RESET);
			print("  ─────────────────────────────────────────────────────────────");
			print("  Juno now includes a native merge tool. After /save, run:");
			print("");
			print("    ./juno merge --model-path " + modelPath);
			print("");
			print("  This copies the GGUF, bakes W_merged = W + (alpha/rank)·B·A");
			print("  for every adapted projection, re-quantises to the original");
			print("  format (Q4_K, Q8_0, F16, …), and writes a standalone GGUF");
			print("  that needs no .lora sidecar at inference time.");
			print("");
			print("  Optional flags:");
			print("    --lora-path PATH     (default: " + adapterFile + ")");
			print("    --output PATH        (default: <model>-merged.gguf)");
			print("    --heap SIZE          (default: 4g — match your model size)");
			print("");
			print("  Run merged model with:  ./juno local --model-path <model>-merged.gguf");
			print("");
		}

		case "/help" -> {
			print("");
			print(Color.CYAN_BOLD + "  LoRA REPL commands" + Color.RESET);
			print("  /train-qa <q> A: <a>  Fine-tune on a Q&A pair in the correct chat format  ← USE THIS");
			print("  /train-file-qa <path.json>  Fine-tune on many Q&A pairs from a JSON file (one loop)");
			print("  /train <text>          Fine-tune on raw text (no chat template applied)");
			print("  /train-file <path>     Fine-tune on a text file (default chunk 32; recommend --lora-chunk-tokens 128)");
			print("  /save                  Save adapter to " + adapterFile);
			print("  /reset                 Reinitialise adapters, clear chat history, delete checkpoint");
			print("  /status                Show adapter info and training statistics");
			print("  /merge-hint            Explain how to bake adapters into model weights");
			print("  Regular input          Chat using the current adapter for inference");
			print("");
			print("  " + Color.YELLOW + "TIP:" + Color.RESET
					+ " Use /train-qa or /train-file-qa for factual recall (names, dates, preferences).");
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
			LoraTrainingHandler handler, Tokenizer tokenizer, int[] totalSteps, boolean[] dirty, String modelType)
			throws Exception {
		trainOnQaPairs(List.of(new LoraQaFile.Pair(question, answer)), adapters, optimizer, handler, tokenizer,
				totalSteps, dirty, modelType);
	}

	/**
	 * Fine-tune on one or more Q&amp;A pairs in a single training loop. Each pair is
	 * expanded to four chat-templated phrasings with completion-only loss.
	 */
	private static void trainOnQaPairs(List<LoraQaFile.Pair> pairs, LoraAdapterSet adapters,
			LoraAdamOptimizer optimizer, LoraTrainingHandler handler, Tokenizer tokenizer, int[] totalSteps,
			boolean[] dirty, String modelType) throws Exception {
		if (pairs == null || pairs.isEmpty()) {
			print(Color.YELLOW + "  No Q&A pairs to train." + Color.RESET);
			return;
		}

		// Echo pairs BEFORE training starts — catches typos.
		print("");
		for (int i = 0; i < pairs.size(); i++) {
			LoraQaFile.Pair p = pairs.get(i);
			print(Color.CYAN_BOLD + "  [" + (i + 1) + "] Q: " + Color.RESET + p.q());
			print(Color.CYAN_BOLD + "      A: " + Color.RESET + p.a());
		}
		print(Color.DIM + "  (check spelling above — typos in Q won't match inference phrasing)" + Color.RESET);
		print("");

		var first = LoraTrainingSequences.buildQa(tokenizer, pairs.get(0).q(), pairs.get(0).a(), modelType);
		int answerPreds = first.predictionCount();
		int totalPreds = Math.max(0, first.tokens().length - 1);
		int variantCount = pairs.size() * 4;

		print(Color.DIM + "  Formatted as " + variantCount + " Q&A variant(s) from " + pairs.size()
				+ " pair(s)  ·  model type: " + modelType + "  ·  completion-only loss (sample "
				+ answerPreds + "/" + totalPreds + " positions)" + Color.RESET);
		if (verbose) {
			print(Color.DIM + "  loss-target=" + loraLossTargetQa + "  max-iters=" + loraMaxItersQa
					+ "  early-stop=" + loraEarlyStop
					+ "  (tune with --lora-loss-target-qa F  --lora-max-iters N  --lora-early-stop F)" + Color.RESET);
		}
		print("");

		List<LoraTrainingLoop.TrainUnit> units = new ArrayList<>();
		for (LoraQaFile.Pair pair : pairs) {
			for (var v : LoraTrainingSequences.buildQaVariants(tokenizer, pair.q(), pair.a(), modelType))
				units.add(new LoraTrainingLoop.TrainUnit(v.tokens(), v.lossMask()));
		}
		trainOnUnits(units, adapters, optimizer, handler, totalSteps, dirty, loraLossTargetQa, loraMaxItersQa, "qa",
				loraChunkTokens);
	}

		/**
	 * Tokenize {@code text}, split into chunks, and run loss-target training: repeat
	 * one gradient step per chunk until loss drops below {@code lossTarget} or
	 * {@code maxIters} passes are exhausted.
	 */
	private static void trainOnText(String text, LoraAdapterSet adapters, LoraAdamOptimizer optimizer,
			LoraTrainingHandler handler, Tokenizer tokenizer, int[] totalSteps, boolean[] dirty) throws Exception {
		trainOnText(text, adapters, optimizer, handler, tokenizer, totalSteps, dirty, loraLossTargetText,
				loraMaxIters, "text");
	}

	private static void trainOnText(String text, LoraAdapterSet adapters, LoraAdamOptimizer optimizer,
			LoraTrainingHandler handler, Tokenizer tokenizer, int[] totalSteps, boolean[] dirty, float lossTarget,
			int maxIters, String logLabel) throws Exception {

		// encode() already prepends BOS when the GGUF says so — do not prepend again.
		// A second BOS makes training OOD vs inference and yields garbage after /train.
		int[] allTokens = tokenizer.encode(text);
		if (allTokens.length < 2) {
			print(Color.YELLOW + "  Input too short to train on (need ≥ 2 tokens)." + Color.RESET);
			return;
		}
		var seq = new LoraTrainingSequences.MaskedSequence(allTokens, LoraTrainingSequences.allTrueMask(allTokens.length),
				text);
		trainOnMasked(seq, adapters, optimizer, handler, totalSteps, dirty, lossTarget, maxIters, logLabel);
	}

	/**
	 * Build a document-level unit (optionally corpus-capped) and run loss-target
	 * training with optional skip when the loaded adapters are already at the
	 * target (avoids one more Adam step that deepens mode collapse).
	 */
	private static void trainOnMasked(LoraTrainingSequences.MaskedSequence seq, LoraAdapterSet adapters,
			LoraAdamOptimizer optimizer, LoraTrainingHandler handler, int[] totalSteps, boolean[] dirty,
			float lossTarget, int maxIters, String logLabel) throws Exception {
		List<LoraTrainingLoop.TrainUnit> units = LoraCorpusLimit.limitDocument(seq.tokens(), seq.lossMask(),
				loraChunkTokens, loraMaxTrainTokens, loraSeed);
		if (units.isEmpty()) {
			print(Color.YELLOW + "  No trainable (completion) tokens in this example." + Color.RESET);
			return;
		}
		if (loraMaxTrainTokens > 0 && units.size() > 1) {
			int cappedPred = 0;
			for (var u : units)
				for (boolean m : u.lossMask())
					if (m)
						cappedPred++;
			print(Color.DIM + "  Corpus cap: max-train-tokens=" + loraMaxTrainTokens + " → " + units.size()
					+ " chunk unit(s), " + cappedPred + " supervised (seed=" + loraSeed + ")" + Color.RESET);
		}
		trainOnUnits(units, adapters, optimizer, handler, totalSteps, dirty, lossTarget, maxIters, logLabel,
				loraChunkTokens);
	}

	private static void trainOnUnits(List<LoraTrainingLoop.TrainUnit> units, LoraAdapterSet adapters,
			LoraAdamOptimizer optimizer, LoraTrainingHandler handler, int[] totalSteps, boolean[] dirty,
			float lossTarget, int maxIters, String logLabel, int chunkTokens) throws Exception {
		LoraTrainingConfig cfg = currentTrainingConfig();
		int predCount = 0;
		int tokenCount = 0;
		for (var u : units) {
			tokenCount += u.tokens().length;
			for (boolean m : u.lossMask())
				if (m)
					predCount++;
		}

		print("");
		System.out.printf(
				"  %sTraining%s  rank=%d · lr=%s · schedule=%s · dropout=%s · plus=%s · decay=%s · targets=%s · accum=%d · max-norm=%s · chunk=%d · target=%.2f · max %d pass(es) · %d unit(s) · %d tokens · %d supervised%n",
				Color.CYAN_BOLD, Color.RESET, loraRank, loraLr, loraLrSchedule, loraDropout, loraPlusRatio,
				loraWeightDecay, loraTargets, loraGradientAccumulation, loraMaxGradNorm, chunkTokens, lossTarget,
				maxIters, units.size(), tokenCount, predCount);
		print("  " + "─".repeat(62));

		// Probe with forward-only eval — skip updates if already at target.
		float probeLoss = LoraTrainingLoop.evaluateUnits(units, (tokens, mask) -> handler.evaluateLoss(tokens, mask),
				chunkTokens);
		if (!Float.isNaN(probeLoss) && probeLoss <= lossTarget) {
			System.out.printf(
					"  %sAlready at target%s  loss=%.4f ≤ %.2f on current adapters — skipped updates.%n",
					Color.YELLOW, Color.RESET, probeLoss, lossTarget);
			System.out.printf(
					"  %s  If every chat reply is the same memorized answer, run /reset then /train-qa again.%s%n%n",
					Color.YELLOW, Color.RESET);
			return;
		}

		int stepsBefore = optimizer.step();
		long trainStart = System.currentTimeMillis();
		long[] lastPassAt = { trainStart };
		float[] baselineLoss = { Float.NaN };
		long[] baselineAtMs = { 0L };
		LoraTrainingLoop.TrainingResult result = LoraTrainingLoop.train(units, cfg, adapters, optimizer,
				(tokens, mask, ctx) -> handler.computeGradients(tokens, mask, ctx),
				(tokens, mask) -> handler.evaluateLoss(tokens, mask), lossTarget, maxIters, loraEarlyStop, chunkTokens,
				(pass, trainLoss, valLoss, optUpdates) -> {
					long now = System.currentTimeMillis();
					long passMs = Math.max(1L, now - lastPassAt[0]);
					lastPassAt[0] = now;
					if (verbose) {
						if (Float.isFinite(valLoss))
							System.out.printf(
									"  [train-%s] iter=%2d  loss=%.4f  val=%.4f  target=%.2f  updates=%d  %dms%n",
									logLabel, pass, trainLoss, valLoss, lossTarget, optUpdates, passMs);
						else
							System.out.printf("  [train-%s] iter=%2d  loss=%.4f  target=%.2f  updates=%d  %dms%n",
									logLabel, pass, trainLoss, lossTarget, optUpdates, passMs);
						return;
					}
					if (pass == LoraTrainProgressBar.BASELINE_PASS && Float.isFinite(trainLoss)
							&& !Float.isFinite(baselineLoss[0])) {
						baselineLoss[0] = trainLoss;
						baselineAtMs[0] = now;
					}
					long sinceBaseline = baselineAtMs[0] > 0L ? Math.max(0L, now - baselineAtMs[0]) : 0L;
					System.out.print(LoraTrainProgressBar.render(pass, trainLoss, lossTarget, baselineLoss[0], passMs,
							sinceBaseline));
					System.out.flush();
				});
		long totalMs = System.currentTimeMillis() - trainStart;
		if (!verbose) {
			if (result.passCount() > 0) {
				long avgPassMs = Math.max(1L, totalMs / Math.max(1, result.passCount()));
				float base = Float.isFinite(baselineLoss[0]) ? baselineLoss[0] : result.finalTrainLoss();
				// Training finished: ETA 0; percent from loss vs target (100% when target met).
				System.out.print(LoraTrainProgressBar.render(result.passCount(), result.finalTrainLoss(), lossTarget,
						base, avgPassMs, 0L));
			}
			System.out.println();
		}

		float lastLoss = result.finalTrainLoss();
		String doneLabel = switch (result.stopReason()) {
		case TARGET_REACHED -> Color.GREEN + "target reached" + Color.RESET;
		case PATIENCE_EXHAUSTED -> Color.YELLOW + "validation patience exhausted" + Color.RESET;
		case LOW_LOSS_GUARD -> Color.YELLOW + "stopped early (overfit guard)" + Color.RESET;
		case MAX_ITERATIONS -> Color.YELLOW + "max iters reached" + Color.RESET;
		default -> Color.YELLOW + result.stopReason().name() + Color.RESET;
		};
		System.out.printf(
				"  %s  train-loss=%.4f  val-loss=%s  best-pass=%s  passes=%d  opt-updates=%d  lrA=%s  lrB=%s  %ds total  · /save to persist%n",
				doneLabel, lastLoss,
				Float.isFinite(result.finalValidationLoss()) ? String.format("%.4f", result.finalValidationLoss()) : "n/a",
				result.bestPass() >= 0 ? Integer.toString(result.bestPass()) : "n/a", result.passCount(),
				result.optimizerUpdateCount(),
				Double.isFinite(optimizer.lastLearningRateA()) ? String.format("%.2e", optimizer.lastLearningRateA())
						: "n/a",
				Double.isFinite(optimizer.lastLearningRateB()) ? String.format("%.2e", optimizer.lastLearningRateB())
						: "n/a",
				totalMs / 1000);
		if (result.validationWarning() != null)
			print(Color.YELLOW + "  " + result.validationWarning() + Color.RESET);
		if (Float.isFinite(lastLoss) && lastLoss < 0.1f) {
			System.out.printf("  %sWARNING: loss=%.4f — adapter is severely overfit.%s%n", Color.RED, lastLoss,
					Color.RESET);
			System.out.printf("  %s  The model will generate garbage until you /reset adapters.%s%n%n", Color.RED,
					Color.RESET);
		} else if (Float.isFinite(lastLoss) && lastLoss < 0.5f
				&& result.stopReason() != LoraTrainingLoop.StopReason.LOW_LOSS_GUARD) {
			System.out.printf("  %sloss < 0.5 — consider stopping here. More passes risk overfitting.%s%n%n",
					Color.YELLOW, Color.RESET);
		} else if (result.stopReason() == LoraTrainingLoop.StopReason.MAX_ITERATIONS) {
			System.out.printf("  %sloss=%.4f still above target=%.2f — raise --lora-max-iters or lower target.%s%n%n",
					Color.YELLOW, lastLoss, lossTarget, Color.RESET);
		} else {
			System.out.println();
		}

		totalSteps[0] += Math.max(0, result.optimizerUpdateCount() - stepsBefore);
		if (result.passCount() > 0)
			dirty[0] = true;
	}

	/** Normalize, clip, Adam-step, and emit one JFR event for an accumulation group. */
	private static float applyLoraOptimizerUpdate(LoraAdapterSet adapters, LoraAdamOptimizer optimizer,
			LoraGradientBatch batch, int numTokens) {
		LoraMetricsIdentity identity = currentTrainingConfig().metricsIdentity();
		LoraTrainEvent event = new LoraTrainEvent();
		event.begin();
		identity.apply(event);
		event.step = optimizer.step() + 1;
		event.numTokens = numTokens;
		event.chunkCount = batch.chunkCount();
		event.predictionCount = batch.predictionCount();
		event.forwardMs = batch.forwardMs();
		event.backwardMs = batch.backwardMs();
		batch.timing().apply(event);

		LoraGradients.PrepResult prep = LoraGradients.prepare(adapters, batch.predictionCount(), loraMaxGradNorm);
		event.globalGradNorm = (float) prep.globalNorm();
		event.clipScale = prep.scale();
		event.clipped = prep.clipped();

		long t0 = System.currentTimeMillis();
		optimizer.step(adapters);
		event.optimizerMs = System.currentTimeMillis() - t0;

		float mean = batch.meanLoss();
		event.loss = mean;
		event.totalMs = event.forwardMs + event.backwardMs + event.optimizerMs;
		event.commit();

		if (identity.algorithm.equals("dora"))
			LoraTrainingLoop.commitNormRefresh(identity, adapters, "post-step", 0L);

		return mean;
	}

	private static void saveAdapters(LoraAdapterSet adapters, Path path, int stepsTrained) {
		try {
			Files.createDirectories(path.getParent() != null ? path.getParent() : Path.of("."));
			long t0 = System.currentTimeMillis();
			adapters.save(path);
			long ms = System.currentTimeMillis() - t0;
			long bytes = Files.size(path);
			commitCheckpointEvent(adapters, "save", 2, adapters.size(), ms, bytes);
			long kb = bytes / 1024;
			print(Color.GREEN + "  ✔ Saved → " + path + "  (" + adapters.size() + " adapters · " + kb + " KB" + "  · "
					+ stepsTrained + " steps trained)" + Color.RESET);
		} catch (IOException e) {
			print(Color.RED + "  ✘ Save failed: " + e.getMessage() + Color.RESET);
		}
	}

	private static void commitCheckpointEvent(LoraAdapterSet adapters, String operation, int version, int entryCount,
			long durationMs, long bytes) {
		LoraMetricsIdentity identity = LoraMetricsIdentity.fromAdapterSet(adapters,
				loraModelConfig != null ? loraModelConfig.architecture() : "",
				loraResolvedTrainDevice);
		LoraCheckpointEvent ev = new LoraCheckpointEvent();
		ev.begin();
		identity.apply(ev);
		ev.operation = operation;
		ev.version = version;
		ev.entryCount = entryCount;
		ev.durationMs = durationMs;
		ev.bytes = bytes;
		ev.commit();
	}

	private static void commitPlaybackEvent(LoraAdapterSet adapters, long loadMs) {
		LoraMetricsIdentity identity = LoraMetricsIdentity.fromAdapterSet(adapters,
				loraModelConfig != null ? loraModelConfig.architecture() : "",
				loraResolvedTrainDevice);
		LoraPlaybackEvent ev = new LoraPlaybackEvent();
		ev.begin();
		identity.apply(ev);
		ev.adapterCount = adapters != null ? adapters.size() : 0;
		ev.loadMs = loadMs;
		ev.commit();
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

	/** Plain-language LoRA train device / microbatch status after weight load. */
	static String formatLoraTrainStatus(String requested, String resolved, int microbatch) {
		String deviceLabel = switch (resolved) {
		case "cuda" -> "CUDA";
		case "rocm" -> "ROCm";
		default -> "CPU";
		};
		String autoNote = cab.ml.juno.node.LoraTrainDevice.AUTO.equals(requested) ? " (auto-selected)" : "";
		return "Training on " + deviceLabel + autoNote + " · microbatch size " + microbatch;
	}

	/** Why {@code --lora-train-device=auto} resolved to CPU, plus the speed consequence. */
	private static String autoCpuFallbackNotice() {
		if (!useGpu)
			return cab.ml.juno.node.LoraTrainNotices.AUTO_CPU_DISABLED;
		boolean gpuAvailable = CudaAvailability.isAvailable() || RocmAvailability.isAvailable();
		if (!gpuAvailable)
			return cab.ml.juno.node.LoraTrainNotices.AUTO_CPU_UNAVAILABLE;
		int dev = Math.max(0, Integer.getInteger("juno.gpu.device", Integer.getInteger("juno.cuda.device", 0)));
		int devCount = CudaAvailability.isAvailable() ? CudaAvailability.deviceCount()
				: RocmAvailability.deviceCount();
		if (dev >= devCount)
			return cab.ml.juno.node.LoraTrainNotices.AUTO_CPU_BAD_DEVICE;
		return cab.ml.juno.node.LoraTrainNotices.AUTO_CPU_UNAVAILABLE;
	}

	// ── JFR local mode ────────────────────────────────────────────────────────

	/**
	 * Starts a programmatic JFR recording for LoRA mode (parity with
	 * {@link #startLocalJfr}), runs the LoRA REPL, and extracts metrics on exit.
	 */
	private static void startLoraJfr() throws Exception {
		startProgrammaticJfr();
		runLoraRepl(); // extracts via stopAndExtractActiveJfr() before System.exit
	}

	/**
	 * Starts a programmatic JFR recording, runs the local REPL, and once the
	 * recording period expires automatically extracts and prints metrics JSON.
	 *
	 * <p>Uses {@code jdk.jfr.Recording} so the JFR lifecycle is fully managed
	 * in-process — no JVM flags required. Metrics are extracted synchronously on
	 * REPL quit; a platform-thread shutdown hook covers Ctrl-C / abrupt exit.
	 */
	private static void startLocalJfr() throws Exception {
		startProgrammaticJfr();
		runLocalRepl(); // extracts via stopAndExtractActiveJfr() before System.exit
	}

	private static void startProgrammaticJfr() throws Exception {
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

		activeJfrRecording = rec;
		activeJfrFile = jfrFile;
		activeJfrModelStem = modelStem;
		activeJfrModelFilename = modelName;
		jfrExtracted = false;

		print(Color.YELLOW + "  ⏱ JFR recording started — duration=" + jfrDuration
				+ "  output=" + jfrFileName + Color.RESET + "\n");

		// Platform thread: virtual-thread shutdown hooks can be skipped during JVM teardown.
		Runtime.getRuntime().addShutdownHook(new Thread(() -> stopAndExtractActiveJfr(), "juno-jfr-extract"));
	}

	/**
	 * Stop the active recording (if any), dump the file, and print metrics once.
	 * Safe to call from the REPL quit path and from a shutdown hook.
	 */
	private static void stopAndExtractActiveJfr() {
		synchronized (jfrLock) {
			if (jfrExtracted)
				return;
			Recording rec = activeJfrRecording;
			Path jfrFile = activeJfrFile;
			String stem = activeJfrModelStem;
			String filename = activeJfrModelFilename;
			if (rec == null || jfrFile == null)
				return;
			jfrExtracted = true;
			try {
				if (rec.getState() == RecordingState.RUNNING)
					rec.stop();
			} catch (Exception e) {
				System.err.println("JFR stop failed: " + e.getMessage());
			}
			try {
				Thread.sleep(200);
			} catch (InterruptedException ignored) {
				Thread.currentThread().interrupt();
			}
			try {
				extractAndPrintJfrMetrics(jfrFile, stem != null ? stem : "model",
						filename != null ? filename : "model.gguf");
			} catch (Exception e) {
				System.err.println("JFR metrics extraction failed: " + e.getMessage());
			} finally {
				try {
					rec.close();
				} catch (Exception ignored) {
				}
				activeJfrRecording = null;
			}
		}
	}

	/**
	 * Starts a programmatic JFR recording on the coordinator, injects
	 * {@code -XX:StartFlightRecording} into every forked node JVM via
	 * {@link ClusterHarness#withJfr}, runs the cluster REPL, and on exit aggregates
	 * all four JFR files (coordinator + 3 nodes) before printing the merged metrics
	 * summary.
	 *
	 * <p>
	 * A <em>single</em> shutdown hook owns the full teardown sequence so that
	 * ordering is guaranteed:
	 * <ol>
	 * <li>Stop coordinator's {@link Recording} (flushes its JFR file).</li>
	 * <li>{@link ClusterHarness#stop()} — destroys node processes; their
	 * {@code dumponexit=true} flag writes each node's JFR file.</li>
	 * <li>Brief sleep to let OS flush all files to disk.</li>
	 * <li>Merge-extract from coordinator + node files → print JSON summary.</li>
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

		print(Color.YELLOW + "  ⏱ JFR recording started — duration=" + jfrDuration + "  output=" + coordinatorJfrName
				+ Color.RESET + "\n");

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
				: ClusterHarness.threeNodes(modelPath, totalLayers)).withJfr(jfrDuration, timestamp);

		if (loraPlayPath != null && !loraPlayPath.isBlank()) {
			harness.withLoraPlay(loraPlayPath);
			print(Color.CYAN + "  ⚙ LoRA inference overlay will be applied on every node: " + loraPlayPath
					+ Color.RESET);
		}
		if (healthMode) {
			harness.withHealthUrl("http://localhost:" + healthPort);
		}

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
			} catch (Exception ignored) {
			}

			// 2. Stop cluster → destroys node processes → dumponexit fires on each node.
			print("\n" + Color.YELLOW + "⏹ Shutting down cluster..." + Color.RESET);
			try {
				harnessRef.stop();
			} catch (Exception ignored) {
			}
			print(Color.YELLOW + "✔ Cluster stopped." + Color.RESET);

			// 3. Wait for coordinator + node JFR files to be fully flushed to disk.
			try {
				Thread.sleep(2000);
			} catch (InterruptedException ignored) {
			}

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
		var loop = new GenerationLoop(tokenizer, Sampler.create(), pipeline, kvCache, prefillMode);

		startRepl(loop, tokenizer); // calls System.exit(0) on quit — shutdown hook fires from there
	}

	/**
	 * Merges the given JFR files (coordinator + node files), extracts metrics, and
	 * prints the JSON summary to the console — same presentation as
	 * {@link #extractAndPrintJfrMetrics} but across multiple files.
	 *
	 * <p>
	 * Files that do not exist (e.g. a node that crashed before dumping) are
	 * silently skipped so a partial result is still reported.
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
	 * Calls {@link MetricsMain#extractToJson} on the finished JFR file, then prints
	 * the resulting JSON to the REPL console inside a highlighted box.
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

		// Local, in-process pipeline-parallelism simulation — every "node" is a
		// stage in this same JVM with no real VRAM difference between stages, so
		// split layers evenly rather than routing through ShardPlanner's
		// VRAM-aware greedy algorithm (correct for real clusters, but a
		// deliberately huge per-node VRAM figure to keep it from ever rejecting
		// a local run causes it to hand nearly the whole model to node 0 and one
		// layer each to every other node — see ShardMap.evenSplit javadoc).
		ShardMap shardMap = ShardMap.evenSplit("model", config.numLayers(), nodeCount);
		// Load .lora adapters for inference-only playback if --lora-play was given
		LoraAdapterSet playAdapters = null;
		if (loraPlayPath != null) {
			print(Color.CYAN + "  ⚙ Loading LoRA adapters for inference: " + loraPlayPath + Color.RESET);
			long t0 = System.currentTimeMillis();
			playAdapters = LoraAdapterSet.load(Path.of(loraPlayPath));
			commitPlaybackEvent(playAdapters, System.currentTimeMillis() - t0);
			print(Color.GREEN + "  ✔ Loaded " + playAdapters.size() + " LoRA adapters  (inference-only, no training)"
					+ Color.RESET);
		}
		List<ForwardPassHandler> handlers = new ArrayList<>();
		GpuContext gpuCtx = prepareGpuContext();
		// One MatVec per process — shares the same GpuContext / cuBLAS handle across
		// shards.
		MatVec sharedBackend = (gpuCtx != null) ? gpuCtx.createMatVec() : ForwardPassHandlerLoader.selectBackend();
		for (var assignment : shardMap.assignments()) {
			var context = ShardContext.from(assignment, config.vocabSize(), config.hiddenDim(), config.numHeads());
			handlers.add(ForwardPassHandlerLoader.load(Path.of(modelPath), context, sharedBackend, playAdapters));
		}

		// MUST run before LocalInferencePipeline.from() below: prepareVisionHandler()
		// replaces handlers.get(0) in place with a VisionAwareForwardPassHandler when
		// the loaded model is vision-capable. LocalInferencePipeline.from() captures a
		// direct reference to each handler into its NodeStage list at construction
		// time — if the wrap happens after that call, the pipeline keeps using the
		// unwrapped text handler forever and vision requests silently never reach the
		// vision-aware code path (see CHANGELOG "vision handler wiring order").
		LlavaHandlerFactory.Built visionBuilt = prepareVisionHandler(modelPath, mmprojPath, handlers, config);

		var pipeline = LocalInferencePipeline.from(shardMap, new ArrayList<>(handlers), config.vocabSize(),
				config.hiddenDim(), config.numHeads());
		var kvCache = new KVCacheManager(new GpuKVCache(512L * 1024 * 1024), new CpuKVCache(4096));
		var loop = new GenerationLoop(tokenizer, Sampler.create(), pipeline, kvCache, prefillMode);
		var scheduler = new cab.ml.juno.coordinator.RequestScheduler(1000, loop,
				cab.ml.juno.coordinator.BatchConfig.disabled());
		if (apiPort > 0) {
			ModelRegistry registry = buildLocalModelRegistry(config, modelPath);
			var apiServer = new cab.ml.juno.coordinator.InferenceApiServer(scheduler, registry, byteOrder);
			registerVisionRoutes(apiServer, scheduler, registry, visionBuilt);
			apiServer.start(apiPort);
			print(Color.GREEN + "  ✔ Local API server on http://localhost:" + apiPort
					+ " (OpenAI: /v1/chat/completions)" + Color.RESET);
			Runtime.getRuntime().addShutdownHook(Thread.ofVirtual().unstarted(apiServer::stop));
		}

		// ── Health reporters for in-process nodes ─────────────────────────────
		if (healthMode) {
			String base = "http://localhost:" + healthPort;
			List<HealthReporter> reporters = new ArrayList<>();
			for (var assignment : shardMap.assignments()) {
				HealthReporter r = new HealthReporter(assignment.nodeId(), base);
				r.startBackground();
				reporters.add(r);
			}
			activeReporters.addAll(reporters);
			Runtime.getRuntime().addShutdownHook(Thread.ofVirtual().unstarted(() -> {
				for (HealthReporter r : reporters)
					r.stop();
			}));
		}

		startRepl(loop, tokenizer);
	}

	/**
	 * Detects a vision-capable model and, if found, wraps {@code handlers.get(0)}
	 * in place with a {@link VisionAwareForwardPassHandler}.
	 *
	 * <p><b>Call this BEFORE {@link LocalInferencePipeline#from}.</b>
	 * {@code LocalInferencePipeline.from()} reads {@code handlers.get(i)} once,
	 * at construction time, and stores that exact reference in each
	 * {@code NodeStage} — it never re-reads the list afterwards. Mutating
	 * {@code handlers} after the pipeline is built has no effect on the
	 * already-constructed pipeline, which then keeps calling the unwrapped text
	 * handler for the lifetime of the process; vision requests never reach
	 * {@link VisionAwareForwardPassHandler}, and image patches registered by
	 * {@code VisionChatHandler} are silently never looked up.
	 *
	 * @return the wired vision components, or {@code null} if the loaded model
	 *         is not vision-capable (nothing was wrapped)
	 */
	private static LlavaHandlerFactory.Built prepareVisionHandler(String modelPath, String mmprojPath,
			List<ForwardPassHandler> handlers, LlamaConfig config) {
		// Build/version marker — fires in under a second, long before the prefill
		// that follows. If this string doesn't match what you expect from the
		// latest delivered files, the running jar is STALE — stop here and
		// rebuild rather than spending 20-30 minutes on a prefill whose result
		// you won't be able to trust. Added 2026-07-20 after two separate stale
		// -build incidents this session (StubForwardPassHandler missing a
		// constructor; the text-embedding-stats log line never appearing) each
		// cost a full ~25-minute cycle to even notice.
		log.info("[vision] BUILD MARKER: session42-2026-07-20 "
				+ "(includes: quick_gelu dispatch, EXIF orientation correction, "
				+ "unconditional forwardBatch/forward entry logging, per-window text-embedding-stats aggregate)");
		log.info("[vision] prepareVisionHandler — modelPath=" + modelPath + "  mmprojPath=" + mmprojPath
				+ "  handlers=" + (handlers == null ? "null" : handlers.size()));
		try {
			Path mmproj = mmprojPath != null ? Path.of(mmprojPath) : null;
			boolean isVision = LlavaHandlerFactory.isVisionArchitecture(Path.of(modelPath), mmproj);
			log.info("[vision] isVisionArchitecture=" + isVision);
			if (!isVision) {
				log.info("[vision] Not a vision model — skipping handler wrap"
						+ (mmproj == null ? " (no --mmproj-path given; pass one if this is a known I2T model)" : ""));
				return null;
			}
			log.info("[vision] Building vision handler from " + handlers.size() + " loaded handler(s)");
			LlavaHandlerFactory.Built built = LlavaHandlerFactory.buildFromHandlers(Path.of(modelPath), mmproj,
					handlers, config);
			log.info("[vision] Built — encoder patches=" + built.config().numPatches() + "  outputDim="
					+ built.encoder().outputDim() + "  imageTokenId=" + built.imageTokenId()
					+ "  handlers.get(0) now=" + handlers.get(0).getClass().getName());
			return built;
		} catch (Exception e) {
			log.severe("[vision] prepareVisionHandler FAILED: " + e.getClass().getName() + ": " + e.getMessage());
			java.util.Arrays.stream(e.getStackTrace()).limit(8).forEach(f -> log.severe("[vision]   at " + f));
			return null;
		}
	}

	/**
	 * Registers {@code POST /v1/vision/chat} and {@code /v1/vision/chat/stream}
	 * on {@code apiServer} when {@code built} is non-null. Call this AFTER
	 * {@code apiServer} exists; the handler wrap itself must already have
	 * happened via {@link #prepareVisionHandler}, before the pipeline was built.
	 */
	private static void registerVisionRoutes(cab.ml.juno.coordinator.InferenceApiServer apiServer,
			cab.ml.juno.coordinator.RequestScheduler scheduler, ModelRegistry registry,
			LlavaHandlerFactory.Built built) {
		if (built == null) {
			return;
		}
		VisionChatHandler visionChatHandler = new VisionChatHandler(scheduler, registry, built.encoder(),
				built.visionHandler());
		apiServer.addRoutes(app -> {
			app.post("/v1/vision/chat", visionChatHandler::handleBlocking);
			app.post("/v1/vision/chat/stream", visionChatHandler::handleStreaming);
			log.info("[vision] Routes registered: POST /v1/vision/chat  POST /v1/vision/chat/stream");
		});
		print(Color.GREEN + "  ✔ Vision routes registered (POST /v1/vision/chat)" + Color.RESET);
	}

	private static ModelRegistry buildLocalModelRegistry(LlamaConfig config, String modelPath) {
		ModelRegistry registry = new ModelRegistry(ShardPlanner.create());
		long vramPerLayer = 4L * config.hiddenDim() * config.hiddenDim() * 2;
		String filename = Path.of(modelPath).getFileName().toString();
		QuantizationType quant = LlamaConfig.fromFilename(filename);
		ModelDescriptor descriptor = new ModelDescriptor(filename, config.architecture(), config.numLayers(),
				config.hiddenDim(), config.vocabSize(), config.numHeads(), vramPerLayer, quant, modelPath,
				ModelStatus.LOADED, Instant.now());
		registry.putLoaded(descriptor);
		return registry;
	}

	private static GpuContext prepareGpuContext() {
		boolean gpuAvailable = CudaAvailability.isAvailable() || RocmAvailability.isAvailable();
		if (!useGpu || !gpuAvailable)
			return null;
		int dev = Math.max(0, Integer.getInteger("juno.cuda.device", 0));
		int devCount = CudaAvailability.isAvailable() ? CudaAvailability.deviceCount() : RocmAvailability.deviceCount();
		if (dev >= devCount) {
			log.warning("juno.cuda.device=" + dev + " out of range — using CPU matmul for local REPL");
			return null;
		}
		final GpuContext gpuCtx = GpuContext.shared(dev);
		Runtime.getRuntime().addShutdownHook(Thread.ofVirtual().unstarted(gpuCtx::close));
		return gpuCtx;
	}
	// ── Cluster mode (unchanged from original) ─────────────────────────────────

	private static void runClusterRepl() throws Exception {
		String modeLabel = pType == ParallelismType.TENSOR ? "tensor-parallel" : "pipeline-parallel";
		print(Color.CYAN_BOLD + "▶ Starting 3-node " + modeLabel + " cluster (forked JVMs)..." + Color.RESET);

		LlamaConfig config;
		try (GgufReader cfgReader = GgufReader.open(Path.of(modelPath))) {
			config = LlamaConfig.from(cfgReader);
		}

		ClusterHarness harness = (pType == ParallelismType.TENSOR)
				? ClusterHarness.tensorNodes(modelPath, config.numLayers(), config.numHeads())
				: ClusterHarness.threeNodes(modelPath, config.numLayers());

		if (healthMode) {
			harness.withHealthUrl("http://localhost:" + healthPort);
		}

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
				: new ProcessPipelineClient(harness.nodeAddresses(), config.vocabSize(), dtype);

		Tokenizer tokenizer;
		try (GgufReader reader = GgufReader.open(Path.of(modelPath))) {
			tokenizer = GgufTokenizer.load(reader);
		}

		var kvCache = new KVCacheManager(new GpuKVCache(512L * 1024 * 1024), new CpuKVCache(4096));
		var loop = new GenerationLoop(tokenizer, Sampler.create(), pipeline, kvCache, prefillMode);
		var scheduler = new cab.ml.juno.coordinator.RequestScheduler(1000, loop,
				cab.ml.juno.coordinator.BatchConfig.disabled());
		if (apiPort > 0) {
			ModelRegistry registry = buildLocalModelRegistry(config, modelPath);
			var apiServer = new cab.ml.juno.coordinator.InferenceApiServer(scheduler, registry, byteOrder);
			apiServer.start(apiPort);
			print(Color.GREEN + "  ✔ Cluster API server on http://localhost:" + apiPort
					+ " (OpenAI: /v1/chat/completions)" + Color.RESET);
			Runtime.getRuntime().addShutdownHook(Thread.ofVirtual().unstarted(apiServer::stop));
		}

		startRepl(loop, tokenizer);
	}

	// ── Standard REPL loop ────────────────────────────────────────────────────

	private static void startRepl(GenerationLoop loop, Tokenizer tokenizer) throws IOException {
		SamplingParams params = samplingParamsFromCli();

		ChatHistory history = new ChatHistory();
		print("");
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
			System.out.printf(Color.GREEN + "     Generated [%d tokens · %d ms · %s]" + Color.RESET + "%n",
					result.generatedTokens(), elapsed, dtype);
			System.out.println();
			activeReporters.forEach(r -> r.recordLatency(elapsed));
		}

		loop.evictSession(history.sessionId());
		print(Color.YELLOW + "\nbye." + Color.RESET);
		stopAndExtractActiveJfr();
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
				// Clear the whole prefill status line (ANSI \r alone leaves a tail).
				System.out.print("\r\033[2K" + Color.GREEN + "bot> " + Color.RESET);
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
			System.out.println(String.format(
					"  %s⚙ LoRA mode  ·  rank=%d  α=%.1f  lr=%s  loss-target=%.1f  max-iters=%d  ·  max_tokens=%d  ·  temperature=%.2f  ·  top_k=%d  ·  top_p=%.2f%s%n",
					Color.PURPLE_BOLD, loraRank, loraAlpha, loraLr, loraLossTargetText, loraMaxIters, maxTokens,
					temperature, topK, topP, Color.RESET));
		} else {
			System.out.println(String.format(
					"  %sdtype=%s · byteOrder=%s · max_tokens=%d · temperature=%.2f · top_k=%d · top_p=%.2f · %s nodes=%d%s%n",
					Color.GREEN_BOLD_BRIGHT, dtype, byteOrder, maxTokens, temperature, topK, topP,
					localMode ? "local" : "cluster", nodeCount, Color.RESET));
		}
		if (jfrDuration != null) {
			System.out.println(
					String.format("  %s⏱ JFR active · duration=%s%s%n", Color.YELLOW, jfrDuration, Color.RESET));
		}
		print("");
		print(Color.YELLOW_BOLD + AiDisclosure.DISCLOSURE_TEXT + Color.RESET);
		print("");
	}

	private static void print(String msg) {
		System.out.println(msg);
		System.out.flush();
	}

	private static ActivationDtype parseDtype(String s) {
		if (s == null)
			return ActivationDtype.FLOAT16;
		return switch (s.toUpperCase()) {
		case "FLOAT32", "F32", "FP32" -> ActivationDtype.FLOAT32;
		case "FLOAT16", "F16", "FP16" -> ActivationDtype.FLOAT16;
		case "INT8", "I8" -> ActivationDtype.INT8;
		default -> {
			System.err.println("WARNING: Unrecognized --dtype '" + s
					+ "' (expected FLOAT32|FLOAT16|INT8) — falling back to FLOAT32");
			yield ActivationDtype.FLOAT32;
		}
		};
	}

	private static cab.ml.juno.coordinator.PrefillMode parsePrefillMode(String s) {
		try {
			return cab.ml.juno.coordinator.PrefillMode.parse(s);
		} catch (IllegalArgumentException e) {
			System.err.println("WARNING: " + e.getMessage() + " — defaulting to 'batched'");
			return cab.ml.juno.coordinator.PrefillMode.BATCHED;
		}
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
}