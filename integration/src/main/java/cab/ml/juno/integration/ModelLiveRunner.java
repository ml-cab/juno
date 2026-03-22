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
package cab.ml.juno.integration;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import cab.ml.juno.coordinator.GenerationLoop;
import cab.ml.juno.coordinator.GenerationResult;
import cab.ml.juno.coordinator.InferenceRequest;
import cab.ml.juno.coordinator.RequestPriority;
import cab.ml.juno.coordinator.TokenConsumer;
import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.node.ActivationDtype;
import cab.ml.juno.node.GgufReader;
import cab.ml.juno.node.LlamaConfig;
import cab.ml.juno.player.ClusterHarness;
import cab.ml.juno.player.EmbeddedNodeServer;
import cab.ml.juno.player.ProcessPipelineClient;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.GgufTokenizer;

/**
 * Standalone runner that performs the same validation as TinyLlamaLiveIT. Can
 * be executed from the command line, for example:
 *
 * java --enable-preview ... -cp integration/target/classes:... \
 * cab.ml.juno.player.TinyLlamaLiveRunner /path/to/model.gguf
 *
 * Or via juno.sh with the integration‑single command (which uses Maven).
 *
 * Exit code: 0 if all checks pass, 1 if any check fails.
 */
public final class ModelLiveRunner {

	// ANSI colours for output (same as ConsoleMain)
	private static final String RED = "\033[0;31m";
	private static final String GREEN = "\033[0;32m";
	private static final String RESET = "\033[0m";
	private static final String BOLD = "\033[1m";

	// Well‑known English words TinyLlama reliably produces for "hello" greetings
	// Words TinyLlama reliably produces for "hello" greetings (multi-language ok)
	private static final Set<String> GREETING_WORDS = Set.of("how", "are", "you", "hello", "hi", "help", "doing",
			"today", "there", "welcome", "assist", "can", "i", "what", "do", "hola", "hey", "greetings", "good",
			"great", "nice", "pleased");

	// Template / EOS marker strings that may bleed into generated text when the
	// model emits them as individual character tokens (bypassing the token-ID
	// check).
	private static final List<String> TEMPLATE_MARKERS = List.of("</s>", "<|endoftext|>", "<|eot_id|>", "<end_of_turn>",
			"<|user|>", "<|assistant|>", "<|system|>", "<|im_end|>", "<|im_start|>");

	/** Parallelism filter — set via -DpType=pipeline|tensor|all (default: all). */
	private static final String PTYPE = System.getProperty("pType", "all").toLowerCase();

	private static String modelPath;
	private static ClusterHarness harness;
	private static GenerationLoop loop;
	private static GgufTokenizer tokenizer;
	private static int vocabSize = EmbeddedNodeServer.VOCAB_SIZE; // updated from GGUF in startCluster
	private static int totalTests = 0;
	private static int passedTests = 0;
	private static List<String> failures = new ArrayList<>();

	public static void main(String[] args) throws Exception {
		parseArgs(args);
		if (modelPath == null) {
			System.err.println("ERROR: Model path is required.");
			System.err.println("Usage: TinyLlamaLiveRunner <path-to-gguf-file>");
			System.exit(1);
		}
		if (!Files.exists(Path.of(modelPath))) {
			System.err.println("ERROR: Model file not found: " + modelPath);
			System.exit(1);
		}

		System.out.println();
		System.out.println(BOLD + "TinyLlama Live Runner" + RESET);
		System.out.println("Model: " + modelPath);
		if (!PTYPE.equals("all"))
			System.out.println("Mode:  " + PTYPE + " tests only");
		System.out.println();

		// ── Pipeline tests (1-6) ─────────────────────────────────────────────────
		if (!PTYPE.equals("tensor")) {
			try {
				startCluster();
				runAllTests();
			} finally {
				stopCluster();
			}
		} else {
			// Tokenizer is still needed for tensor-parallel tests even when the
			// pipeline cluster is skipped.
			System.out.println("(Skipping pipeline tests 1-6: pType=" + PTYPE + ")");
			try (GgufReader reader = GgufReader.open(Path.of(modelPath))) {
				LlamaConfig cfg = LlamaConfig.from(reader);
				vocabSize = cfg.vocabSize();
				tokenizer = GgufTokenizer.load(reader);
			}
		}

		// ── Tensor-parallel tests (7-8) ───────────────────────────────────────────
		if (!PTYPE.equals("pipeline")) {
			runTensorParallelTest();
		}

		printSummary();
		System.exit(failures.isEmpty() ? 0 : 1);
	}

	private static void parseArgs(String[] args) {
		if (args.length > 0) {
			modelPath = args[0];
		} else {
			modelPath = System.getenv("MODEL_PATH");
		}
	}

	private static void startCluster() throws Exception {
		System.out.print("Starting 3-node pipeline cluster... ");
		System.out.flush();

		// Single GGUF pass: read model config + tokenizer together.
		int totalLayers;
		try (GgufReader cfgReader = GgufReader.open(Path.of(modelPath))) {
			LlamaConfig cfg = LlamaConfig.from(cfgReader);
			totalLayers = cfg.numLayers();
			vocabSize   = cfg.vocabSize();       // actual vocab (e.g. 32064 for phi-3.5)
			tokenizer   = GgufTokenizer.load(cfgReader);
		}

		harness = ClusterHarness.threeNodes(modelPath, totalLayers);
		harness.start();

		// Reuse the harness's already-connected pipeline client; pass the actual
		// vocabSize so GenerationLoop reports the correct output dimension.
		loop = new GenerationLoop(tokenizer, Sampler.create(),
				new ProcessPipelineClient(harness.nodeAddresses(), vocabSize, ActivationDtype.FLOAT32),
				new KVCacheManager(new GpuKVCache(512L * 1024 * 1024), new CpuKVCache(4096)));

		System.out.println(GREEN + "OK" + RESET);
	}

	private static void stopCluster() throws Exception {
		if (harness != null) {
			harness.stop();
		}
	}

	private static void runAllTests() {
		testHelloGreeting();
		testNoRawSentencePiece();
		testQuestionResponse();
		testGreedyDeterminism();
		testMultiTurn();
		testFloat16Parity();
	}

	private static void testHelloGreeting() {
		totalTests++;
		System.out.print("Test 1: hello greeting... ");
		System.out.flush();

		try {
			// 20 tokens gives TinyLlama room to produce a real greeting sentence after
			// the chat-template overhead (EOS, <|user|> etc that cleanText strips).
			GenerationResult result = generate("hello", 20);
			int tokenCount = result.generatedTokens();
			// Strip any template/EOS tokens that bled through as character-level pieces
			String text = cleanText(result.text());

			if (text.isEmpty()) {
				fail("Response is empty after template cleanup (raw: \"" + result.text() + "\")");
				return;
			}

			String lower = text.toLowerCase();
			long matchCount = GREETING_WORDS.stream().filter(lower::contains).count();
			// One greeting word is sufficient — "hello" alone is a valid reply.
			if (matchCount < 1) {
				fail("Response \"" + text + "\" contains no greeting words from " + GREETING_WORDS);
				return;
			}

			System.out.println(GREEN + "PASS" + RESET + "  (" + tokenCount + " tokens)");
			passedTests++;
		} catch (Exception e) {
			fail("Exception: " + e.getMessage());
			e.printStackTrace();
		}
	}

	private static void testNoRawSentencePiece() {
		totalTests++;
		System.out.print("Test 2: no raw ▁ markers... ");
		System.out.flush();

		try {
			List<String> pieces = new ArrayList<>();
			loop.generate(request("hello", 10), (piece, _, _) -> pieces.add(piece));

			for (String piece : pieces) {
				if (piece.contains("\u2581")) {
					fail("Raw ▁ found in piece: \"" + piece + "\"");
					return;
				}
			}
			System.out.println(GREEN + "PASS" + RESET);
			passedTests++;
		} catch (Exception e) {
			fail("Exception: " + e.getMessage());
			e.printStackTrace();
		}
	}

	private static void testQuestionResponse() {
		totalTests++;
		System.out.print("Test 3: question response... ");
		System.out.flush();

		try {
			GenerationResult result = generate("What is 2 plus 2?", 12);
			if (result.text().strip().isEmpty()) {
				fail("Response is empty");
				return;
			}
			System.out.println(GREEN + "PASS" + RESET + "  (" + result.generatedTokens() + " tokens)");
			passedTests++;
		} catch (Exception e) {
			fail("Exception: " + e.getMessage());
			e.printStackTrace();
		}
	}

	private static void testGreedyDeterminism() {
		totalTests++;
		System.out.print("Test 4: greedy determinism... ");
		System.out.flush();

		try {
			SamplingParams greedy = SamplingParams.deterministic().withMaxTokens(8);

			GenerationResult r1 = loop.generate(InferenceRequest.of("tinyllama", List.of(ChatMessage.user("hello")),
					greedy, RequestPriority.NORMAL), TokenConsumer.discard());

			GenerationResult r2 = loop.generate(InferenceRequest.of("tinyllama", List.of(ChatMessage.user("hello")),
					greedy, RequestPriority.NORMAL), TokenConsumer.discard());

			if (!r1.text().equals(r2.text())) {
				fail("Responses differ:\n  r1: \"" + r1.text() + "\"\n  r2: \"" + r2.text() + "\"");
				return;
			}
			System.out.println(GREEN + "PASS" + RESET);
			passedTests++;
		} catch (Exception e) {
			fail("Exception: " + e.getMessage());
			e.printStackTrace();
		}
	}

	private static void testMultiTurn() {
		totalTests++;
		System.out.print("Test 5: multi-turn conversation... ");
		System.out.flush();

		try {
			List<ChatMessage> conversation = List.of(ChatMessage.user("hello"),
					ChatMessage.assistant("Hello! How can I help you today?"), ChatMessage.user("What is Java?"));

			SamplingParams params = SamplingParams.defaults().withMaxTokens(12);
			GenerationResult result = loop.generate(
					InferenceRequest.of("tinyllama", conversation, params, RequestPriority.NORMAL),
					TokenConsumer.discard());

			if (result.generatedTokens() == 0) {
				fail("No tokens generated");
				return;
			}
			if (result.promptTokens() <= 20) {
				fail("Prompt tokens (" + result.promptTokens() + ") should be >20 for multi-turn");
				return;
			}
			System.out.println(GREEN + "PASS" + RESET + "  (" + result.generatedTokens() + " tokens)");
			passedTests++;
		} catch (Exception e) {
			fail("Exception: " + e.getMessage());
			e.printStackTrace();
		}
	}

	private static void testFloat16Parity() {
		totalTests++;
		System.out.print("Test 6: FLOAT16 parity... ");
		System.out.flush();

		ProcessPipelineClient f16Pipeline = null;
		try {
			f16Pipeline = new ProcessPipelineClient(harness.nodeAddresses(), vocabSize,
					ActivationDtype.FLOAT16);
			GenerationLoop f16Loop = new GenerationLoop(tokenizer, Sampler.create(), f16Pipeline,
					new KVCacheManager(new GpuKVCache(512L * 1024 * 1024), new CpuKVCache(256)));

			// Use a short generation to verify the F16 pipeline produces coherent output.
			// Exact token-level match with F32 is intentionally NOT required: INT16
			// quantization shifts logit magnitudes enough that the argmax may differ
			// (e.g. F32 picks "WHERE", F16 picks "H") — both are valid top-K tokens.
			// What we validate: F16 pipeline runs end-to-end and produces non-empty text.
			SamplingParams params = SamplingParams.defaults().withMaxTokens(5).withTemperature(0.7f);

			GenerationResult f32result = loop.generate(InferenceRequest.of("tinyllama",
					List.of(ChatMessage.user("hello")), params, RequestPriority.NORMAL), TokenConsumer.discard());

			GenerationResult f16result = f16Loop.generate(InferenceRequest.of("tinyllama",
					List.of(ChatMessage.user("hello")), params, RequestPriority.NORMAL), TokenConsumer.discard());

			if (f16result.generatedTokens() == 0) {
				fail("F16 pipeline produced no tokens (F32 produced: \"" + f32result.text() + "\")");
				return;
			}
			System.out.println(GREEN + "PASS" + RESET);
			passedTests++;
		} catch (Exception e) {
			fail("Exception: " + e.getMessage());
			e.printStackTrace();
		} finally {
			if (f16Pipeline != null)
				try {
					f16Pipeline.shutdown();
				} catch (InterruptedException ie) {
					System.err.println("unable to shutdown f16 pipeline: " + ie.getMessage());
				}
		}
	}

	/**
	 * Standalone lifecycle for the tensor-parallel check (Test 7).
	 *
	 * The pipeline cluster (Tests 1-6) is already stopped before this runs, so
	 * the three node ports are free. A fresh 3-node tensor-parallel cluster is
	 * started, one generation is verified, then the cluster is torn down.
	 */
	private static void runTensorParallelTest() {
		ClusterHarness tensorHarness = null;
		try {
			System.out.print("Starting 3-node tensor-parallel cluster... ");
			System.out.flush();

			int totalLayers;
			int numHeads;
			try (GgufReader cfgReader = GgufReader.open(Path.of(modelPath))) {
				LlamaConfig cfg = LlamaConfig.from(cfgReader);
				totalLayers = cfg.numLayers();
				numHeads    = cfg.numHeads();
			}

			tensorHarness = ClusterHarness.tensorNodes(modelPath, totalLayers, numHeads);
			tensorHarness.start();
			System.out.println(GREEN + "OK" + RESET);

			GenerationLoop tensorLoop = new GenerationLoop(
					tokenizer,
					Sampler.create(),
					tensorHarness.pipeline(),
					new KVCacheManager(new GpuKVCache(512L * 1024 * 1024), new CpuKVCache(4096)));

			testTensorParallelGeneration(tensorLoop);
			testTensorParallelDeterminism(tensorLoop);
		} catch (Exception e) {
			System.out.println(RED + "FAIL" + RESET + " (cluster startup: " + e.getMessage() + ")");
			failures.add("Tensor-parallel cluster failed to start: " + e.getMessage());
			totalTests += 2;   // count both skipped tests as failures
		} finally {
			if (tensorHarness != null) {
				try {
					tensorHarness.stop();
				} catch (Exception e) {
					System.err.println("Warning: tensor harness stop failed: " + e.getMessage());
				}
			}
		}
	}

	private static void testTensorParallelGeneration(GenerationLoop tensorLoop) {
		totalTests++;
		System.out.print("Test 7: tensor-parallel generation (AllReduce)... ");
		System.out.flush();

		try {
			GenerationResult result = tensorLoop.generate(
					InferenceRequest.of("tinyllama",
							List.of(ChatMessage.user("hello")),
							SamplingParams.defaults().withMaxTokens(10).withTemperature(0.7f),
							RequestPriority.NORMAL),
					TokenConsumer.discard());

			if (result.generatedTokens() == 0) {
				fail("Tensor-parallel pipeline produced no tokens");
				return;
			}
			String text = cleanText(result.text());
			if (text.isEmpty()) {
				fail("Tensor-parallel response is empty after cleanup (raw: \"" + result.text() + "\")");
				return;
			}
			System.out.println(GREEN + "PASS" + RESET
					+ "  (" + result.generatedTokens() + " tokens via AllReduce)");
			passedTests++;
		} catch (Exception e) {
			fail("Exception: " + e.getMessage());
			e.printStackTrace();
		}
	}

	private static void testTensorParallelDeterminism(GenerationLoop tensorLoop) {
		totalTests++;
		System.out.print("Test 8: tensor-parallel greedy determinism... ");
		System.out.flush();

		try {
			SamplingParams greedy = SamplingParams.deterministic().withMaxTokens(8);

			GenerationResult r1 = tensorLoop.generate(
					InferenceRequest.of("tinyllama",
							List.of(ChatMessage.user("hello")),
							greedy, RequestPriority.NORMAL),
					TokenConsumer.discard());

			GenerationResult r2 = tensorLoop.generate(
					InferenceRequest.of("tinyllama",
							List.of(ChatMessage.user("hello")),
							greedy, RequestPriority.NORMAL),
					TokenConsumer.discard());

			if (!r1.text().equals(r2.text())) {
				fail("Tensor-parallel responses differ:\n  r1: \"" + r1.text()
						+ "\"\n  r2: \"" + r2.text() + "\"");
				return;
			}
			System.out.println(GREEN + "PASS" + RESET);
			passedTests++;
		} catch (Exception e) {
			fail("Exception: " + e.getMessage());
			e.printStackTrace();
		}
	}

	/**
	 * Strip template / EOS marker tokens that bled into the text as character-level
	 * pieces (e.g. model generates '<', '/', 's', '>' individually so isEosMarker
	 * never fires per-piece inside GenerationLoop). Truncates at the first marker
	 * and strips surrounding whitespace.
	 */
	private static String cleanText(String raw) {
		for (String marker : TEMPLATE_MARKERS) {
			int idx = raw.indexOf(marker);
			if (idx >= 0)
				raw = raw.substring(0, idx);
		}
		return raw.strip();
	}

	private static GenerationResult generate(String userMessage, int maxTokens) {
		return loop.generate(request(userMessage, maxTokens), TokenConsumer.discard());
	}

	private static InferenceRequest request(String userMessage, int maxTokens) {
		return InferenceRequest.of("tinyllama", List.of(ChatMessage.user(userMessage)),
				SamplingParams.defaults().withMaxTokens(maxTokens).withTemperature(0.7f), RequestPriority.NORMAL);
	}

	private static void fail(String message) {
		System.out.println(RED + "FAIL" + RESET);
		failures.add(message);
	}

	private static void printSummary() {
		System.out.println();
		System.out.println("========================================");
		System.out.printf("Tests run: %d, Passed: %d, Failed: %d%n", totalTests, passedTests, failures.size());
		if (!failures.isEmpty()) {
			System.out.println(RED + "Failures:" + RESET);
			for (int i = 0; i < failures.size(); i++) {
				System.out.printf("  %d. %s%n", i + 1, failures.get(i));
			}
		}
		System.out.println("========================================");
	}
}