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
package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/**
 * Teacher-forced perplexity of one fixed English text under both RoPE pair
 * layouts, on real model files, CPU backend.
 *
 * <p>A wrong pair layout does not fail: it assigns learned frequencies to the
 * wrong dimension pairs, and a short factual prompt can survive it. Perplexity
 * over several hundred positions cannot. The text is
 * {@code rope-pairing-perplexity.txt} (the opening of the United States
 * Declaration of Independence, public domain). {@code node} cannot depend on the
 * tokenizer module, so the token ids are checked in beside it, produced once by
 * {@code GgufTokenizer.encode} on the raw text (no chat template):
 * {@code .llama-spm.ids} from the TinyLlama file (leading BOS included, as the
 * tokenizer emits it), {@code .qwen2-bpe.ids} from the Qwen2.5 file; the Qwen3
 * file shares that vocabulary and encodes the text to the same ids.
 *
 * <p>TinyLlama is the control: its converter permuted Q/K rows for adjacent
 * pairs, so adjacent must win there, or the method does not discriminate.
 *
 * <p>Two kinds of test. The regression tests run by default whenever the model
 * file is present: the production loader ({@link ForwardPassHandlerLoader}), so
 * whatever layout production picks, over the first {@link #REGRESSION_TOKENS}
 * tokens, against a per-model ceiling. The ceilings sit between the measured
 * right-layout and wrong-layout perplexities at that length, far from both. The
 * diagnostic A/B (both layouts, whole text, about 36 minutes on the reference
 * host) runs only with {@code -Djuno.test.ropeAb=true}; it enables a test, not a
 * production behaviour.
 */
class RopePairingPerplexityLiveTest {

	private static final Path MODELS = Path.of(System.getProperty("user.dir")).endsWith("node")
			? Path.of(System.getProperty("user.dir")).getParent().resolve("models")
			: Path.of("models");

	private static final Path TINYLLAMA = MODELS.resolve("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
	private static final Path QWEN25 = MODELS.resolve("qwen2.5-3b-instruct-q4_k_m.gguf");
	private static final Path QWEN3 = MODELS.resolve("Qwen3-1.7B-Q4_K_M.gguf");

	// Ceilings over the first REGRESSION_TOKENS tokens. Measured on the reference
	// host, right layout / wrong layout: TinyLlama 2.88 / 467, Qwen2.5-3B
	// 1.29 / 23.0, Qwen3-1.7B 2.50 / 209. Each ceiling is two to three times the
	// right-layout reading and far below the wrong-layout one.
	private static final double TINYLLAMA_CEILING = 6.0;
	private static final double QWEN25_CEILING = 4.0;
	private static final double QWEN3_CEILING = 8.0;

	/** Prefix length for the default regression run: long enough to separate the layouts, short enough to run. */
	static final int REGRESSION_TOKENS = 128;

	@Test
	void production_layout_tinyllama() throws Exception {
		assertProductionBelow(TINYLLAMA, "llama-spm", TINYLLAMA_CEILING);
	}

	@Test
	void production_layout_qwen25_3b() throws Exception {
		assertProductionBelow(QWEN25, "qwen2-bpe", QWEN25_CEILING);
	}

	@Test
	void production_layout_qwen3_1_7b() throws Exception {
		assertProductionBelow(QWEN3, "qwen2-bpe", QWEN3_CEILING);
	}

	@Test
	@EnabledIfSystemProperty(named = "juno.test.ropeAb", matches = "true")
	void ab_control_tinyllama_adjacent_wins() throws Exception {
		double[] ppl = measure(TINYLLAMA, "llama-spm", false);
		assertThat(ppl[0]).as("adjacent must beat split-half on the control").isLessThan(ppl[1]);
	}

	@Test
	@EnabledIfSystemProperty(named = "juno.test.ropeAb", matches = "true")
	void ab_qwen25_3b_split_half_wins() throws Exception {
		double[] ppl = measure(QWEN25, "qwen2-bpe", false);
		assertThat(ppl[1]).as("split-half must beat adjacent on qwen2").isLessThan(ppl[0]);
	}

	@Test
	@EnabledIfSystemProperty(named = "juno.test.ropeAb", matches = "true")
	void ab_qwen3_1_7b_split_half_wins() throws Exception {
		double[] ppl = measure(QWEN3, "qwen2-bpe", true);
		assertThat(ppl[1]).as("split-half must beat adjacent on qwen3").isLessThan(ppl[0]);
	}

	private static void assertProductionBelow(Path model, String idsName, double ceiling) throws Exception {
		assumeTrue(Files.exists(model), "model not present: " + model);
		int[] ids = Arrays.copyOf(readIds(idsName), REGRESSION_TOKENS);
		ShardContext ctx = context(model);
		ForwardPassHandler handler = ForwardPassHandlerLoader.load(model, ctx, CpuMatVec.INSTANCE);
		double ppl = perplexity(handler, ctx, ids, "production");
		System.out.printf(java.util.Locale.ROOT, "ROPE-PPL %s production tokens=%d ppl=%.4f ceiling=%.1f%n",
				model.getFileName(), ids.length, ppl, ceiling);
		assertThat(ppl).as("perplexity of %s over %d tokens", model.getFileName(), ids.length)
				.isLessThan(ceiling);
	}

	/** Returns {adjacent perplexity, split-half perplexity}. */
	private static double[] measure(Path model, String idsName, boolean qwen3) throws Exception {
		assumeTrue(Files.exists(model), "model not present: " + model);
		int[] ids = readIds(idsName);
		double adjacent = perplexity(model, ids, qwen3, RopePairing.ADJACENT);
		double split = perplexity(model, ids, qwen3, RopePairing.SPLIT_HALF);
		System.out.printf(java.util.Locale.ROOT, "ROPE-PPL %s tokens=%d adjacent=%.4f split_half=%.4f ratio(adjacent/split)=%.3f%n",
				model.getFileName(), ids.length, adjacent, split, adjacent / split);
		return new double[] { adjacent, split };
	}

	private static ShardContext context(Path model) throws IOException {
		try (GgufReader r = GgufReader.open(model)) {
			LlamaConfig c = LlamaConfig.from(r);
			return new ShardContext("n0", 0, c.numLayers(), true, true, c.vocabSize(), c.hiddenDim(), c.numHeads());
		}
	}

	private static double perplexity(Path model, int[] ids, boolean qwen3, RopePairing pairing)
			throws IOException {
		ShardContext ctx = context(model);
		ForwardPassHandler handler = qwen3
				? Qwen3TransformerHandler.load(model, ctx, CpuMatVec.INSTANCE, pairing)
				: LlamaTransformerHandler.load(model, ctx, CpuMatVec.INSTANCE, pairing);
		double ppl = perplexity(handler, ctx, ids, "ppl-" + pairing);
		System.out.printf(java.util.Locale.ROOT, "ROPE-PPL %s pairing=%s ppl=%.4f%n", model.getFileName(), pairing,
				ppl);
		return ppl;
	}

	/** Teacher-forced: token t at position t, scored on token t+1. */
	private static double perplexity(ForwardPassHandler handler, ShardContext ctx, int[] ids, String kv) {
		int[] one = new int[1];
		double nll = 0;
		long t0 = System.nanoTime();
		for (int t = 0; t < ids.length - 1; t++) {
			one[0] = ids[t];
			float[] logits = handler.forward(ForwardRequest.withTokens(kv, one, t), ctx).logits();
			nll -= logSoftmaxAt(logits, ids[t + 1]);
		}
		handler.evict(kv);
		double ppl = Math.exp(nll / (ids.length - 1));
		System.out.printf(java.util.Locale.ROOT, "ROPE-PPL %s ppl=%.4f (%.1f s)%n", kv, ppl,
				(System.nanoTime() - t0) / 1e9);
		return ppl;
	}

	private static double logSoftmaxAt(float[] logits, int target) {
		double max = Double.NEGATIVE_INFINITY;
		for (float l : logits)
			max = Math.max(max, l);
		double sum = 0;
		for (float l : logits)
			sum += Math.exp(l - max);
		return logits[target] - max - Math.log(sum);
	}

	private static int[] readIds(String name) throws IOException {
		String res = "rope-pairing-perplexity." + name + ".ids";
		try (InputStream in = RopePairingPerplexityLiveTest.class.getResourceAsStream(res)) {
			assertThat(in).as(res).isNotNull();
			String s = new String(in.readAllBytes(), StandardCharsets.US_ASCII).strip();
			return Arrays.stream(s.split("\\s+")).mapToInt(Integer::parseInt).toArray();
		}
	}
}
