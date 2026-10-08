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

import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;

import cab.ml.juno.coordinator.BatchConfig;
import cab.ml.juno.coordinator.GenerationLoop;
import cab.ml.juno.coordinator.GenerationResult;
import cab.ml.juno.coordinator.InferenceRequest;
import cab.ml.juno.coordinator.PrefillMode;
import cab.ml.juno.coordinator.RequestPriority;
import cab.ml.juno.coordinator.RequestScheduler;
import cab.ml.juno.coordinator.TokenConsumer;
import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.kvcache.ServeScheduleOptions;
import cab.ml.juno.node.ActivationDtype;
import cab.ml.juno.node.CudaAvailability;
import cab.ml.juno.node.ForwardPassHandler;
import cab.ml.juno.node.ForwardPassHandlerLoader;
import cab.ml.juno.node.GgufReader;
import cab.ml.juno.node.GpuContext;
import cab.ml.juno.node.LlamaConfig;
import cab.ml.juno.node.LocalInferencePipeline;
import cab.ml.juno.node.MatVec;
import cab.ml.juno.node.ShardContext;
import cab.ml.juno.player.ClusterHarness;
import cab.ml.juno.player.ProcessPipelineClient;
import cab.ml.juno.registry.ShardMap;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.GgufTokenizer;

/**
 * The real-model checks behind {@code ./juno test} ({@link ModelLiveRunner}) and
 * {@code ModelLiveRunnerIT}, written once so the command and the integration test
 * cannot drift apart. Each check is run even when an earlier one failed, and
 * reports a pass or a failure with the reason; nothing here throws for a failed
 * check.
 *
 * <ol>
 * <li>Hello greeting: the reply contains a greeting word</li>
 * <li>No raw SentencePiece markers in emitted pieces</li>
 * <li>A simple question gets a non-empty reply</li>
 * <li>Greedy decoding is deterministic</li>
 * <li>Multi-turn: the prompt token count reflects the conversation history</li>
 * <li>A FLOAT16 activation pipeline produces output</li>
 * <li>Tensor-parallel generation produces output</li>
 * <li>Tensor-parallel greedy decoding is deterministic</li>
 * <li>Long-prompt prefill: a prompt of at least 512 tokens, prefilled in-process as
 * one window on the {@code static} schedule and in 32-token chunks on the
 * {@code continuous} schedule, gives the same first greedy token as one-token-at-a-time
 * prefill ({@link PrefillMode#SINGLE}); on the GPU when one is present. On the GPU
 * the same comparison runs again at 2048 tokens, capped where the model's context
 * cannot hold that prompt and the generated tokens</li>
 * </ol>
 *
 * Checks 1-6 run on a three-node pipeline-parallel cluster of forked JVMs, 7-8 on a
 * three-node tensor-parallel cluster, 9 in this JVM: cluster prefill is one forward
 * pass per prompt token, so the batched prefill window is only reached in-process.
 */
public final class ModelLiveChecks {

	/** Which group of checks to run. */
	public enum Suite {
		/** Checks 1-6, pipeline-parallel cluster. */
		PIPELINE,
		/** Checks 7-8, tensor-parallel cluster. */
		TENSOR,
		/** Check 9, in-process long-prompt prefill. */
		PREFILL
	}

	/** One check's outcome. */
	public record LiveCheck(int number, String name, boolean passed, String detail) {
	}

	private static final Set<String> GREETING_WORDS = Set.of("how", "are", "you", "hello", "hi", "help", "doing",
			"today", "there", "welcome", "assist", "can", "i", "what", "do", "hola", "hey", "greetings", "good",
			"great", "nice", "pleased");

	private static final List<String> TEMPLATE_MARKERS = List.of("</s>", "<|endoftext|>", "<|eot_id|>",
			"<end_of_turn>", "<|user|>", "<|assistant|>", "<|system|>", "<|im_end|>", "<|im_start|>");

	private ModelLiveChecks() {
	}

	/** {@code general.architecture} of the file, lower-cased; {@code llama} when absent. */
	public static String architecture(String modelPath) throws IOException {
		try (GgufReader reader = GgufReader.open(Path.of(modelPath))) {
			String arch = reader.metaString("general.architecture");
			return arch != null ? arch.toLowerCase(Locale.ROOT).strip() : "llama";
		}
	}

	/**
	 * Runs the requested suites against {@code modelPath}, which must hold an
	 * architecture the loader supports, writing one line per check to
	 * {@code progress} as it completes.
	 */
	public static List<LiveCheck> run(String modelPath, Set<Suite> suites, PrintStream progress) throws IOException {
		LlamaConfig cfg;
		GgufTokenizer tokenizer;
		int contextLength;
		try (GgufReader reader = GgufReader.open(Path.of(modelPath))) {
			cfg = LlamaConfig.from(reader);
			tokenizer = GgufTokenizer.load(reader);
			contextLength = reader.metaInt(cfg.architecture() + ".context_length", 0);
		}
		List<LiveCheck> results = new ArrayList<>();
		if (suites.contains(Suite.PIPELINE))
			runPipeline(modelPath, cfg, tokenizer, results, progress);
		if (suites.contains(Suite.TENSOR))
			runTensor(modelPath, cfg, tokenizer, results, progress);
		if (suites.contains(Suite.PREFILL))
			record(results, progress, 9, "long-prompt prefill, both schedules",
					() -> longPromptPrefill(modelPath, cfg, tokenizer, contextLength));
		return results;
	}

	// ── Suites ────────────────────────────────────────────────────────────────

	private static void runPipeline(String modelPath, LlamaConfig cfg, GgufTokenizer tokenizer,
			List<LiveCheck> results, PrintStream progress) {
		String[] names = { "hello greeting", "no raw SentencePiece markers", "question response",
				"greedy determinism", "multi-turn context", "FLOAT16 pipeline" };
		ClusterHarness harness = ClusterHarness.threeNodes(modelPath, cfg.numLayers());
		ProcessPipelineClient client = null;
		int before = results.size();
		try {
			harness.start();
			client = new ProcessPipelineClient(harness.nodeAddresses(), cfg.vocabSize(), ActivationDtype.FLOAT32);
			GenerationLoop loop = new GenerationLoop(tokenizer, Sampler.create(), client, newKvCache(4096));
			record(results, progress, 1, names[0], () -> helloGreeting(loop));
			record(results, progress, 2, names[1], () -> noRawSentencePiece(loop));
			record(results, progress, 3, names[2], () -> questionResponse(loop));
			record(results, progress, 4, names[3], () -> greedyDeterminism(loop, "Test 4"));
			record(results, progress, 5, names[4], () -> multiTurn(loop));
			record(results, progress, 6, names[5], () -> float16Pipeline(harness, tokenizer, cfg.vocabSize()));
		} catch (IOException | InterruptedException | RuntimeException e) {
			if (e instanceof InterruptedException)
				Thread.currentThread().interrupt();
			for (int i = results.size() - before; i < names.length; i++)
				add(results, progress, new LiveCheck(i + 1, names[i], false, "pipeline cluster: " + e.getMessage()));
		} finally {
			if (client != null) {
				try {
					client.shutdown();
				} catch (Exception ignored) {
				}
			}
			stopQuietly(harness);
		}
	}

	private static void runTensor(String modelPath, LlamaConfig cfg, GgufTokenizer tokenizer, List<LiveCheck> results,
			PrintStream progress) {
		String[] names = { "tensor-parallel generation", "tensor-parallel greedy determinism" };
		ClusterHarness harness = ClusterHarness.tensorNodes(modelPath, cfg.numLayers(), cfg.numHeads());
		int before = results.size();
		try {
			harness.start();
			GenerationLoop loop = new GenerationLoop(tokenizer, Sampler.create(), harness.pipeline(),
					newKvCache(4096));
			record(results, progress, 7, names[0], () -> tensorGeneration(loop));
			record(results, progress, 8, names[1], () -> greedyDeterminism(loop, "Test 8"));
		} catch (IOException | InterruptedException | RuntimeException e) {
			if (e instanceof InterruptedException)
				Thread.currentThread().interrupt();
			for (int i = results.size() - before; i < names.length; i++)
				add(results, progress, new LiveCheck(i + 7, names[i], false, "tensor cluster: " + e.getMessage()));
		} finally {
			stopQuietly(harness);
		}
	}

	// ── Checks 1-8 ────────────────────────────────────────────────────────────

	private static String helloGreeting(GenerationLoop loop) {
		GenerationResult result = loop.generate(request("hello", 20), TokenConsumer.discard());
		String text = cleanText(result.text());
		check(!text.isEmpty(), "response is empty after template cleanup (raw: \"" + result.text() + "\")");
		long matches = GREETING_WORDS.stream().filter(text.toLowerCase(Locale.ROOT)::contains).count();
		check(matches >= 1, "response \"" + text + "\" contains no greeting word");
		return "\"" + text + "\"";
	}

	private static String noRawSentencePiece(GenerationLoop loop) {
		List<String> pieces = new ArrayList<>();
		loop.generate(request("hello", 10), (piece, tokenId, step) -> pieces.add(piece));
		for (String piece : pieces)
			check(!piece.contains("▁"), "raw ▁ marker in piece \"" + piece + "\"");
		return pieces.size() + " pieces";
	}

	private static String questionResponse(GenerationLoop loop) {
		GenerationResult result = loop.generate(request("What is 2 plus 2?", 12), TokenConsumer.discard());
		check(!result.text().strip().isEmpty(), "response is empty");
		return "\"" + cleanText(result.text()) + "\"";
	}

	private static String greedyDeterminism(GenerationLoop loop, String label) {
		SamplingParams greedy = SamplingParams.deterministic().withMaxTokens(8);
		GenerationResult r1 = loop.generate(
				InferenceRequest.of("model", List.of(ChatMessage.user("hello")), greedy, RequestPriority.NORMAL),
				TokenConsumer.discard());
		GenerationResult r2 = loop.generate(
				InferenceRequest.of("model", List.of(ChatMessage.user("hello")), greedy, RequestPriority.NORMAL),
				TokenConsumer.discard());
		check(r1.text().equals(r2.text()), "responses differ: \"" + r1.text() + "\" vs \"" + r2.text() + "\"");
		return "\"" + cleanText(r1.text()) + "\" twice";
	}

	private static String multiTurn(GenerationLoop loop) {
		List<ChatMessage> conversation = List.of(ChatMessage.user("hello"),
				ChatMessage.assistant("Hello! How can I help you today?"), ChatMessage.user("What is Java?"));
		GenerationResult result = loop.generate(InferenceRequest.of("model", conversation,
				SamplingParams.defaults().withMaxTokens(12), RequestPriority.NORMAL), TokenConsumer.discard());
		check(result.generatedTokens() != 0, "no tokens generated");
		check(result.promptTokens() > 20, "prompt tokens (" + result.promptTokens() + ") should be >20");
		return result.promptTokens() + " prompt tokens";
	}

	private static String float16Pipeline(ClusterHarness harness, GgufTokenizer tokenizer, int vocabSize) {
		ProcessPipelineClient f16 = new ProcessPipelineClient(harness.nodeAddresses(), vocabSize,
				ActivationDtype.FLOAT16);
		try {
			GenerationLoop loop = new GenerationLoop(tokenizer, Sampler.create(), f16, newKvCache(256));
			GenerationResult result = loop.generate(InferenceRequest.of("model", List.of(ChatMessage.user("hello")),
					SamplingParams.defaults().withMaxTokens(5).withTemperature(0.7f), RequestPriority.NORMAL),
					TokenConsumer.discard());
			check(result.generatedTokens() != 0, "FLOAT16 pipeline produced no tokens");
			return result.generatedTokens() + " tokens";
		} finally {
			try {
				f16.shutdown();
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
			}
		}
	}

	private static String tensorGeneration(GenerationLoop loop) {
		GenerationResult result = loop.generate(InferenceRequest.of("model", List.of(ChatMessage.user("hello")),
				SamplingParams.defaults().withMaxTokens(10).withTemperature(0.7f), RequestPriority.NORMAL),
				TokenConsumer.discard());
		check(result.generatedTokens() != 0, "tensor-parallel pipeline produced no tokens");
		String text = cleanText(result.text());
		check(!text.isEmpty(), "response is empty after cleanup (raw: \"" + result.text() + "\")");
		return "\"" + text + "\"";
	}

	// ── Check 9 ───────────────────────────────────────────────────────────────

	/** Prompt length of check 9's first leg, run on every backend. */
	static final int PROMPT_TOKENS = 512;

	/** Prompt length of check 9's second leg, run on the GPU only. */
	static final int LONG_PROMPT_TOKENS = 2048;

	/** Generated tokens per request in check 9. */
	private static final int GENERATED_TOKENS = 16;

	/**
	 * Room left in the context for the chat template's tokens and for the one
	 * numbered note by which {@link #longPrompt} may overshoot its target.
	 */
	private static final int CONTEXT_MARGIN = 64;

	/**
	 * Runs the comparison at {@link #PROMPT_TOKENS} and, on the GPU, again at
	 * {@link #longPromptTarget}. The second leg is skipped on the CPU, where three
	 * 2048-token prefills of a 7B model take hours, and the detail says so.
	 */
	private static String longPromptPrefill(String modelPath, LlamaConfig cfg, GgufTokenizer tokenizer,
			int contextLength) throws IOException {
		String detail = prefillLeg(modelPath, cfg, tokenizer, PROMPT_TOKENS, contextLength);
		if (!CudaAvailability.isAvailable())
			return detail + "; " + LONG_PROMPT_TOKENS + "-token leg not run on the CPU backend";
		int target = longPromptTarget(contextLength);
		String cap = target < LONG_PROMPT_TOKENS
				? " (" + LONG_PROMPT_TOKENS + " capped at " + target + ", context " + contextLength + ")"
				: "";
		return detail + "; " + prefillLeg(modelPath, cfg, tokenizer, target, contextLength) + cap;
	}

	/**
	 * {@link #LONG_PROMPT_TOKENS}, or less where the model's context (0 when the
	 * file does not state one) cannot also hold the generated tokens and the margin.
	 */
	static int longPromptTarget(int contextLength) {
		if (contextLength <= 0)
			return LONG_PROMPT_TOKENS;
		return Math.min(LONG_PROMPT_TOKENS, contextLength - GENERATED_TOKENS - CONTEXT_MARGIN);
	}

	/**
	 * One leg of check 9 at a user message of at least {@code minTokens} tokens.
	 * The oracle is {@link PrefillMode#SINGLE}, the path that predates batched
	 * prefill. The first greedy token is compared exactly; later tokens may part
	 * where two candidates are within float noise (the windowed and per-token paths
	 * sum in different orders), so the first differing step is reported, not
	 * asserted. On the CPU a 512-token prefill of TinyLlama takes about 90 s, and
	 * three are run.
	 */
	private static String prefillLeg(String modelPath, LlamaConfig cfg, GgufTokenizer tokenizer, int minTokens,
			int contextLength) throws IOException {
		String prompt = longPrompt(tokenizer, minTokens);
		String savedSchedule = System.getProperty(ServeScheduleOptions.ENV);
		GpuContext gpu = CudaAvailability.isAvailable() ? GpuContext.init(0) : null;
		try {
			MatVec backend = gpu != null ? gpu.createMatVec() : ForwardPassHandlerLoader.selectBackend();

			System.setProperty(ServeScheduleOptions.ENV, "static");
			GenerationResult oracle;
			GenerationResult window;
			ForwardPassHandler dense = loadSingleShard(modelPath, cfg, backend);
			try {
				LocalInferencePipeline pipeline = singleShardPipeline(cfg, dense);
				oracle = new GenerationLoop(tokenizer, Sampler.create(), pipeline, newKvCache(4096),
						PrefillMode.SINGLE, 1).generate(longPromptRequest(prompt), TokenConsumer.discard());
				window = new GenerationLoop(tokenizer, Sampler.create(), pipeline, newKvCache(4096),
						PrefillMode.BATCHED, 4096).generate(longPromptRequest(prompt), TokenConsumer.discard());
			} finally {
				dense.releaseGpuResources();
			}

			System.setProperty(ServeScheduleOptions.ENV, "continuous");
			GenerationResult chunked;
			ForwardPassHandler paged = loadSingleShard(modelPath, cfg, backend);
			RequestScheduler scheduler = null;
			try {
				GenerationLoop loop = new GenerationLoop(tokenizer, Sampler.create(), singleShardPipeline(cfg, paged),
						newKvCache(4096), PrefillMode.BATCHED, 32);
				scheduler = new RequestScheduler(16, loop, BatchConfig.disabled(),
						ServeScheduleOptions.of(ServeScheduleOptions.Mode.CONTINUOUS));
				chunked = scheduler.submitAndWait(longPromptRequest(prompt));
			} finally {
				if (scheduler != null)
					scheduler.shutdown();
				paged.releaseGpuResources();
			}

			String detail = String.format(Locale.ROOT,
					"backend=%s prompt_tokens=%d first-divergence static=%d continuous=%d",
					gpu != null ? "cuda" : "cpu", oracle.promptTokens(),
					firstDifference(oracle.tokenIds(), window.tokenIds()),
					firstDifference(oracle.tokenIds(), chunked.tokenIds()));
			check(oracle.promptTokens() >= minTokens, "prompt must be at least " + minTokens + " tokens but was "
					+ oracle.promptTokens() + " (" + detail + ")");
			check(contextLength <= 0 || oracle.promptTokens() + GENERATED_TOKENS <= contextLength,
					"prompt of " + oracle.promptTokens() + " tokens and " + GENERATED_TOKENS
							+ " generated tokens exceed the model's context of " + contextLength);
			check(window.promptTokens() == oracle.promptTokens() && chunked.promptTokens() == oracle.promptTokens(),
					"every path must prefill the same prompt: per-token " + oracle.promptTokens() + ", static "
							+ window.promptTokens() + ", continuous " + chunked.promptTokens());
			check(!oracle.tokenIds().isEmpty() && !window.tokenIds().isEmpty() && !chunked.tokenIds().isEmpty(),
					"every path must generate a token (" + detail + ")");
			check(window.tokenIds().get(0).equals(oracle.tokenIds().get(0)), "static window first token "
					+ window.tokenIds().get(0) + " differs from per-token prefill's " + oracle.tokenIds().get(0));
			check(chunked.tokenIds().get(0).equals(oracle.tokenIds().get(0)), "continuous chunked first token "
					+ chunked.tokenIds().get(0) + " differs from per-token prefill's " + oracle.tokenIds().get(0));
			check(!cleanText(window.text()).isEmpty(),
					"static response is empty after cleanup (raw: \"" + window.text() + "\")");
			return detail;
		} finally {
			if (savedSchedule == null)
				System.clearProperty(ServeScheduleOptions.ENV);
			else
				System.setProperty(ServeScheduleOptions.ENV, savedSchedule);
			if (gpu != null)
				gpu.close();
		}
	}

	private static ForwardPassHandler loadSingleShard(String modelPath, LlamaConfig cfg, MatVec backend)
			throws IOException {
		ShardContext context = new ShardContext("long-prompt", 0, cfg.numLayers(), true, true, cfg.vocabSize(),
				cfg.hiddenDim(), cfg.numHeads());
		return ForwardPassHandlerLoader.load(Path.of(modelPath), context, backend);
	}

	private static LocalInferencePipeline singleShardPipeline(LlamaConfig cfg, ForwardPassHandler handler) {
		return LocalInferencePipeline.from(ShardMap.evenSplit("model", cfg.numLayers(), 1), handler, cfg.vocabSize(),
				cfg.hiddenDim(), cfg.numHeads());
	}

	private static InferenceRequest longPromptRequest(String prompt) {
		return InferenceRequest.of("model", List.of(ChatMessage.user(prompt)),
				SamplingParams.deterministic().withMaxTokens(GENERATED_TOKENS), RequestPriority.NORMAL);
	}

	/**
	 * A user message of at least {@code minTokens} tokens on its own (the chat
	 * template adds more), from distinct numbered sentences so the prompt does not
	 * collapse into a repetition, ending in a question answered by its first sentence.
	 */
	private static String longPrompt(GgufTokenizer tokenizer, int minTokens) {
		String[] places = { "harbour", "orchard", "library", "bridge", "market", "lighthouse", "mill", "chapel" };
		String[] colours = { "red", "green", "blue", "white", "yellow", "grey", "black", "orange" };
		StringBuilder sb = new StringBuilder("Read the notes below, then answer the question at the end.\n");
		for (int i = 1; tokenizer.encode(sb.toString()).length < minTokens; i++) {
			sb.append("Note ").append(i).append(": the ").append(places[i % places.length]).append(" in district ")
					.append(i * 7 % 31).append(" was painted ").append(colours[(i * 3) % colours.length])
					.append(" in the year ").append(1800 + i * 13).append(".\n");
		}
		sb.append("Question: according to note 1, what colour was the ").append(places[1])
				.append(" painted? Answer in one sentence.");
		return sb.toString();
	}

	// ── Helpers ───────────────────────────────────────────────────────────────

	@FunctionalInterface
	private interface Check {
		/** Returns a short detail on success; throws on failure. */
		String run() throws Exception;
	}

	private static final class CheckFailed extends RuntimeException {
		CheckFailed(String message) {
			super(message);
		}
	}

	private static void check(boolean condition, String failure) {
		if (!condition)
			throw new CheckFailed(failure);
	}

	private static void record(List<LiveCheck> results, PrintStream progress, int number, String name, Check check) {
		LiveCheck outcome;
		try {
			outcome = new LiveCheck(number, name, true, check.run());
		} catch (CheckFailed e) {
			outcome = new LiveCheck(number, name, false, e.getMessage());
		} catch (Exception | AssertionError | OutOfMemoryError e) {
			outcome = new LiveCheck(number, name, false, e.getClass().getSimpleName() + ": " + e.getMessage());
		}
		add(results, progress, outcome);
	}

	private static void add(List<LiveCheck> results, PrintStream progress, LiveCheck outcome) {
		results.add(outcome);
		if (progress != null)
			progress.printf(Locale.ROOT, "%s  %d. %s: %s%n", outcome.passed() ? "PASS" : "FAIL", outcome.number(),
					outcome.name(), outcome.detail());
	}

	private static void stopQuietly(ClusterHarness harness) {
		try {
			harness.stop();
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}
	}

	private static KVCacheManager newKvCache(int cpuEntries) {
		return new KVCacheManager(new GpuKVCache(512L * 1024 * 1024), new CpuKVCache(cpuEntries));
	}

	private static InferenceRequest request(String userMessage, int maxTokens) {
		return InferenceRequest.of("model", List.of(ChatMessage.user(userMessage)),
				SamplingParams.defaults().withMaxTokens(maxTokens).withTemperature(0.7f), RequestPriority.NORMAL);
	}

	private static String cleanText(String raw) {
		for (String marker : TEMPLATE_MARKERS) {
			int idx = raw.indexOf(marker);
			if (idx >= 0)
				raw = raw.substring(0, idx);
		}
		return raw.strip();
	}

	private static int firstDifference(List<Integer> a, List<Integer> b) {
		int n = Math.min(a.size(), b.size());
		for (int i = 0; i < n; i++)
			if (!a.get(i).equals(b.get(i)))
				return i;
		return a.size() == b.size() ? -1 : n;
	}
}
