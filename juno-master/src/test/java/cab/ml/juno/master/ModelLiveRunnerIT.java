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

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.stream.Stream;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import cab.ml.juno.coordinator.GenerationLoop;
import cab.ml.juno.coordinator.GenerationResult;
import cab.ml.juno.coordinator.InferenceRequest;
import cab.ml.juno.coordinator.RequestPriority;
import cab.ml.juno.coordinator.TokenConsumer;
import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.node.ActivationDtype;
import cab.ml.juno.node.EmbeddedNodeServer;
import cab.ml.juno.node.GgufReader;
import cab.ml.juno.node.LlamaConfig;
import cab.ml.juno.player.ClusterHarness;
import cab.ml.juno.coordinator.ProcessPipelineClient;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.GgufTokenizer;

/**
 * Live model integration tests — run once per each model path listed in the
 * {@code MODELS} system property (comma-separated absolute GGUF paths).
 *
 * <p>Disabled by default. Activate with the {@code integration} Maven profile:
 *
 * <pre>
 *   mvn verify -pl integration -Pintegration \
 *       -DMODELS=/data/tinyllama.Q4_K_M.gguf,/data/phi-3.5.Q4_K_M.gguf
 * </pre>
 *
 * <p>Each model gets its own cluster lifecycle (start / run 8 tests / stop) so
 * failures in one model do not abort tests for the rest.
 *
 * <p>Test scenarios per model (mirrors {@code ModelLiveRunner} main-class logic):
 * <ol>
 *   <li>Hello greeting — response contains at least one greeting word</li>
 *   <li>No raw SentencePiece ▁ markers in emitted pieces</li>
 *   <li>Question response — non-empty reply to a simple question</li>
 *   <li>Greedy determinism — two runs with identical greedy params produce equal text</li>
 *   <li>Multi-turn — prompt token count reflects conversation history</li>
 *   <li>FLOAT16 parity — F16 pipeline produces non-empty output</li>
 *   <li>Tensor-parallel generation — AllReduce path produces non-empty text</li>
 *   <li>Tensor-parallel greedy determinism — two tensor-parallel runs match</li>
 * </ol>
 */
@DisplayName("Model Live Runner")
class ModelLiveRunnerIT {

    // ── Constants ─────────────────────────────────────────────────────────────

    private static final Set<String> GREETING_WORDS = Set.of(
            "how", "are", "you", "hello", "hi", "help", "doing", "today", "there",
            "welcome", "assist", "can", "i", "what", "do", "hola", "hey", "greetings",
            "good", "great", "nice", "pleased");

    private static final List<String> TEMPLATE_MARKERS = List.of(
            "</s>", "<|endoftext|>", "<|eot_id|>", "<end_of_turn>",
            "<|user|>", "<|assistant|>", "<|system|>", "<|im_end|>", "<|im_start|>");

    // ── Model source ──────────────────────────────────────────────────────────

    /**
     * Reads the {@code MODELS} system property (comma-separated GGUF paths).
     * Skips blank entries and paths that do not exist, logging a warning for each.
     * If no valid paths remain the stream is empty and JUnit reports no tests run.
     */
    static Stream<String> modelPaths() {
        String prop = System.getProperty("MODELS", "").strip();
        if (prop.isEmpty()) {
            System.err.println("[ModelLiveRunnerIT] MODELS property is not set — no tests will run.");
            return Stream.empty();
        }
        return Arrays.stream(prop.split(","))
                .map(String::strip)
                .filter(p -> {
                    if (p.isEmpty()) return false;
                    if (!Files.exists(Path.of(p))) {
                        System.err.println("[ModelLiveRunnerIT] Model not found, skipping: " + p);
                        return false;
                    }
                    return true;
                });
    }

    // ── Parametrized test ─────────────────────────────────────────────────────

    /**
     * Runs the full suite (tests 1–8) for {@code modelPath}.
     *
     * <p>All assertions are collected via {@link org.junit.jupiter.api.Assertions#assertAll}
     * so every sub-test is attempted even if earlier ones fail, giving a full
     * picture of model quality in a single run.
     */
    @ParameterizedTest(name = "{0}")
    @MethodSource("modelPaths")
    @DisplayName("Full suite")
    void testModel(String modelPath) throws Exception {

        // ── Read config + tokenizer ───────────────────────────────────────
        int totalLayers;
        int numHeads;
        int vocabSize = EmbeddedNodeServer.VOCAB_SIZE;
        GgufTokenizer tokenizer;

        try (GgufReader reader = GgufReader.open(Path.of(modelPath))) {
            LlamaConfig cfg = LlamaConfig.from(reader);
            totalLayers = cfg.numLayers();
            numHeads    = cfg.numHeads();
            vocabSize   = cfg.vocabSize();
            tokenizer   = GgufTokenizer.load(reader);
        }

        // ── Pipeline cluster (tests 1-6) ──────────────────────────────────
        ClusterHarness harness = ClusterHarness.threeNodes(modelPath, totalLayers);
        harness.start();

        final int finalVocabSize = vocabSize;
        ProcessPipelineClient f32Client =
                new ProcessPipelineClient(harness.nodeAddresses(), finalVocabSize, ActivationDtype.FLOAT32);
        GenerationLoop loop = new GenerationLoop(
                tokenizer, Sampler.create(), f32Client,
                new KVCacheManager(new GpuKVCache(512L * 1024 * 1024), new CpuKVCache(4096)));

        // Collect pipeline sub-test executables; run them even if earlier ones fail
        try {
            assertAll("pipeline tests for " + Path.of(modelPath).getFileName(),
                    () -> assertHelloGreeting(loop),
                    () -> assertNoRawSentencePiece(loop),
                    () -> assertQuestionResponse(loop),
                    () -> assertGreedyDeterminism(loop),
                    () -> assertMultiTurn(loop),
                    () -> assertFloat16Parity(harness, tokenizer, finalVocabSize));
        } finally {
            try { f32Client.shutdown(); } catch (Exception ignored) {}
            harness.stop();
        }

        // ── Tensor-parallel cluster (tests 7-8) ───────────────────────────
        // Nodes were freed by harness.stop(); ports are available again.
        ClusterHarness tensorHarness = ClusterHarness.tensorNodes(modelPath, totalLayers, numHeads);
        tensorHarness.start();
        GenerationLoop tensorLoop = new GenerationLoop(
                tokenizer, Sampler.create(), tensorHarness.pipeline(),
                new KVCacheManager(new GpuKVCache(512L * 1024 * 1024), new CpuKVCache(4096)));

        try {
            assertAll("tensor-parallel tests for " + Path.of(modelPath).getFileName(),
                    () -> assertTensorParallelGeneration(tensorLoop),
                    () -> assertTensorParallelDeterminism(tensorLoop));
        } finally {
            tensorHarness.stop();
        }
    }

    // ── Sub-test assertions ───────────────────────────────────────────────────

    private void assertHelloGreeting(GenerationLoop loop) {
        GenerationResult result = generate(loop, "hello", 20);
        String text = cleanText(result.text());
        assertFalse(text.isEmpty(),
                "Test 1: response is empty after template cleanup (raw: \"" + result.text() + "\")");
        long matchCount = GREETING_WORDS.stream().filter(text.toLowerCase()::contains).count();
        assertTrue(matchCount >= 1,
                "Test 1: response \"" + text + "\" contains no greeting words from " + GREETING_WORDS);
    }

    private void assertNoRawSentencePiece(GenerationLoop loop) {
        List<String> pieces = new java.util.ArrayList<>();
        loop.generate(request(loop, "hello", 10), (piece, tokenId, step) -> pieces.add(piece));
        for (String piece : pieces) {
            assertFalse(piece.contains("\u2581"),
                    "Test 2: raw \u2581 found in piece: \"" + piece + "\"");
        }
    }

    private void assertQuestionResponse(GenerationLoop loop) {
        GenerationResult result = generate(loop, "What is 2 plus 2?", 12);
        assertFalse(result.text().strip().isEmpty(), "Test 3: response is empty");
    }

    private void assertGreedyDeterminism(GenerationLoop loop) {
        SamplingParams greedy = SamplingParams.deterministic().withMaxTokens(8);
        GenerationResult r1 = loop.generate(
                InferenceRequest.of("model", List.of(ChatMessage.user("hello")), greedy, RequestPriority.NORMAL),
                TokenConsumer.discard());
        GenerationResult r2 = loop.generate(
                InferenceRequest.of("model", List.of(ChatMessage.user("hello")), greedy, RequestPriority.NORMAL),
                TokenConsumer.discard());
        assertTrue(r1.text().equals(r2.text()),
                "Test 4: responses differ — r1: \"" + r1.text() + "\", r2: \"" + r2.text() + "\"");
    }

    private void assertMultiTurn(GenerationLoop loop) {
        List<ChatMessage> conversation = List.of(
                ChatMessage.user("hello"),
                ChatMessage.assistant("Hello! How can I help you today?"),
                ChatMessage.user("What is Java?"));
        SamplingParams params = SamplingParams.defaults().withMaxTokens(12);
        GenerationResult result = loop.generate(
                InferenceRequest.of("model", conversation, params, RequestPriority.NORMAL),
                TokenConsumer.discard());
        assertNotEquals(0, result.generatedTokens(), "Test 5: no tokens generated");
        assertTrue(result.promptTokens() > 20,
                "Test 5: prompt tokens (" + result.promptTokens() + ") should be >20 for multi-turn");
    }

    private void assertFloat16Parity(ClusterHarness harness, GgufTokenizer tokenizer, int vocabSize) {
        ProcessPipelineClient f16Client =
                new ProcessPipelineClient(harness.nodeAddresses(), vocabSize, ActivationDtype.FLOAT16);
        try {
            GenerationLoop f16Loop = new GenerationLoop(
                    tokenizer, Sampler.create(), f16Client,
                    new KVCacheManager(new GpuKVCache(512L * 1024 * 1024), new CpuKVCache(256)));
            SamplingParams params = SamplingParams.defaults().withMaxTokens(5).withTemperature(0.7f);
            GenerationResult f16result = f16Loop.generate(
                    InferenceRequest.of("model", List.of(ChatMessage.user("hello")), params, RequestPriority.NORMAL),
                    TokenConsumer.discard());
            assertNotEquals(0, f16result.generatedTokens(),
                    "Test 6: F16 pipeline produced no tokens");
        } finally {
            try { f16Client.shutdown(); } catch (InterruptedException ignored) {}
        }
    }

    private void assertTensorParallelGeneration(GenerationLoop tensorLoop) {
        GenerationResult result = tensorLoop.generate(
                InferenceRequest.of("model", List.of(ChatMessage.user("hello")),
                        SamplingParams.defaults().withMaxTokens(10).withTemperature(0.7f),
                        RequestPriority.NORMAL),
                TokenConsumer.discard());
        assertNotEquals(0, result.generatedTokens(),
                "Test 7: tensor-parallel pipeline produced no tokens");
        assertFalse(cleanText(result.text()).isEmpty(),
                "Test 7: tensor-parallel response is empty after cleanup (raw: \"" + result.text() + "\")");
    }

    private void assertTensorParallelDeterminism(GenerationLoop tensorLoop) {
        SamplingParams greedy = SamplingParams.deterministic().withMaxTokens(8);
        GenerationResult r1 = tensorLoop.generate(
                InferenceRequest.of("model", List.of(ChatMessage.user("hello")), greedy, RequestPriority.NORMAL),
                TokenConsumer.discard());
        GenerationResult r2 = tensorLoop.generate(
                InferenceRequest.of("model", List.of(ChatMessage.user("hello")), greedy, RequestPriority.NORMAL),
                TokenConsumer.discard());
        assertTrue(r1.text().equals(r2.text()),
                "Test 8: tensor-parallel responses differ — r1: \"" + r1.text() + "\", r2: \"" + r2.text() + "\"");
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private GenerationResult generate(GenerationLoop loop, String userMessage, int maxTokens) {
        return loop.generate(request(loop, userMessage, maxTokens), TokenConsumer.discard());
    }

    private InferenceRequest request(GenerationLoop loop, String userMessage, int maxTokens) {
        return InferenceRequest.of("model", List.of(ChatMessage.user(userMessage)),
                SamplingParams.defaults().withMaxTokens(maxTokens).withTemperature(0.7f),
                RequestPriority.NORMAL);
    }

    private static String cleanText(String raw) {
        for (String marker : TEMPLATE_MARKERS) {
            int idx = raw.indexOf(marker);
            if (idx >= 0) raw = raw.substring(0, idx);
        }
        return raw.strip();
    }
}