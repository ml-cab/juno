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
import java.util.Locale;

import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * Model quality of the integer-dot CPU kernels: mean teacher-forced negative
 * log-likelihood over a fixed 2,048-token text, integer kernels against the
 * float-activation kernels, on every sweep model, CPU backend.
 *
 * <p>The text is {@code model-quality-nll.txt}: the public-domain opening of the
 * United States Declaration of Independence (the RoPE perplexity text) followed
 * by a frozen excerpt of this project's how-to reference. {@code node} cannot
 * depend on the tokenizer module, so each model's token ids are checked in
 * beside it ({@code model-quality-nll.<name>.ids}), produced once by
 * {@code GgufTokenizer.encode} on the raw text and cut to 2,049 ids.
 *
 * <p>Teacher forcing runs through {@link ForwardPassHandler#forwardVerify},
 * which keeps every position's logits: the batched prefill matmul on the
 * LLaMA-family handler, and one decode-width forward per position on the
 * handlers that take the interface default. Both are production paths. The
 * same handler instance is scored twice, once per kernel setting.
 *
 * <p>Threshold: integer NLL at most 1.01x the float NLL on every model. Also
 * reported: the share of the first 512 positions at which the two settings'
 * logits have the same argmax. Runs only with
 * {@code -Djuno.test.modelQuality=true} (over an hour on the reference host).
 */
class ModelQualityNllLiveTest {

	private static final Path MODELS = Path.of(System.getProperty("user.dir")).endsWith("node")
			? Path.of(System.getProperty("user.dir")).getParent().resolve("models")
			: Path.of("models");

	static final int TOKENS = 2048;
	static final int ARGMAX_POSITIONS = 512;
	static final int WINDOW = 128;
	static final double MAX_RATIO = 1.01;

	@ParameterizedTest(name = "{0}")
	@CsvSource({ "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf, tinyllama", "qwen2.5-3b-instruct-q4_k_m.gguf, qwen25",
			"Phi-3.5-mini-instruct-Q4_K_M.gguf, phi35", "mistral-7b-instruct-v0.1-q4_k_m.gguf, mistral" })
	@EnabledIfSystemProperty(named = "juno.test.modelQuality", matches = "true")
	void integerKernelNllWithinOnePercentOfFloat(String file, String idsName) throws Exception {
		Path model = MODELS.resolve(file);
		assumeTrue(Files.exists(model), "model not present: " + model);
		int[] ids = readIds(idsName);
		assertThat(ids.length).as("ids for %s", idsName).isGreaterThanOrEqualTo(TOKENS + 1);
		ids = Arrays.copyOf(ids, TOKENS + 1);

		ShardContext ctx;
		try (GgufReader r = GgufReader.open(model)) {
			LlamaConfig c = LlamaConfig.from(r);
			ctx = new ShardContext("n0", 0, c.numLayers(), true, true, c.vocabSize(), c.hiddenDim(), c.numHeads());
		}
		ForwardPassHandler handler = ForwardPassHandlerLoader.load(model, ctx, CpuMatVec.INSTANCE);

		boolean was = CpuKernelSelection.setIntegerEnabled(true);
		Score integer;
		Score floatAct;
		try {
			integer = score(handler, ctx, ids, "nll-int");
			CpuKernelSelection.setIntegerEnabled(false);
			floatAct = score(handler, ctx, ids, "nll-float");
		} finally {
			CpuKernelSelection.setIntegerEnabled(was);
		}
		int agree = 0;
		for (int t = 0; t < ARGMAX_POSITIONS; t++)
			if (integer.argmax[t] == floatAct.argmax[t])
				agree++;
		double ratio = integer.meanNll / floatAct.meanNll;
		System.out.printf(Locale.ROOT,
				"QUALITY-NLL %s tokens=%d nll_int=%.5f nll_float=%.5f ratio=%.5f ppl_int=%.4f ppl_float=%.4f "
						+ "argmax_agree_first%d=%d (%.1f%%) int_s=%.1f float_s=%.1f%n",
				file, TOKENS, integer.meanNll, floatAct.meanNll, ratio, Math.exp(integer.meanNll),
				Math.exp(floatAct.meanNll), ARGMAX_POSITIONS, agree, 100.0 * agree / ARGMAX_POSITIONS,
				integer.seconds, floatAct.seconds);
		assertThat(ratio).as("integer NLL over float NLL on %s", file).isLessThanOrEqualTo(MAX_RATIO);
	}

	record Score(double meanNll, int[] argmax, double seconds) {
	}

	/** Teacher-forced: token t at position t, scored on token t+1, in windows of {@link #WINDOW}. */
	private static Score score(ForwardPassHandler handler, ShardContext ctx, int[] ids, String kv) {
		long t0 = System.nanoTime();
		int predictions = ids.length - 1;
		int[] argmax = new int[predictions];
		double nll = 0;
		for (int start = 0; start < predictions; start += WINDOW) {
			int w = Math.min(WINDOW, predictions - start);
			int[] window = Arrays.copyOfRange(ids, start, start + w);
			float[] logits = handler.forwardVerify(BatchForwardRequest.withTokens(kv, window, start), ctx).logits();
			int vocab = logits.length / w;
			for (int i = 0; i < w; i++) {
				int off = i * vocab;
				double max = Double.NEGATIVE_INFINITY;
				int best = 0;
				for (int v = 0; v < vocab; v++)
					if (logits[off + v] > max) {
						max = logits[off + v];
						best = v;
					}
				double sum = 0;
				for (int v = 0; v < vocab; v++)
					sum += Math.exp(logits[off + v] - max);
				nll -= logits[off + ids[start + i + 1]] - max - Math.log(sum);
				argmax[start + i] = best;
			}
		}
		handler.evict(kv);
		return new Score(nll / predictions, argmax, (System.nanoTime() - t0) / 1e9);
	}

	private static int[] readIds(String name) throws IOException {
		String res = "model-quality-nll." + name + ".ids";
		try (InputStream in = ModelQualityNllLiveTest.class.getResourceAsStream(res)) {
			assertThat(in).as(res).isNotNull();
			String s = new String(in.readAllBytes(), StandardCharsets.US_ASCII).strip();
			return Arrays.stream(s.split("\\s+")).mapToInt(Integer::parseInt).toArray();
		}
	}
}
