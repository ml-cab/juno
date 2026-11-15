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

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Locale;

import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import cab.ml.juno.kvcache.CacheTypeOptions;

/**
 * Model quality of a quantized host KV against the {@code f16} host KV, on every sweep model, CPU backend (so attention
 * reads the host rows; GPU attention reads its own FP16 copy).
 *
 * <p>Two readings per model and KV type, each from a handler loaded with that {@code --cache-type-k/-v}:
 * <ul>
 * <li><b>NLL</b>: mean teacher-forced negative log-likelihood over the fixed 2,048-token text of
 * {@link ModelQualityNllLiveTest}, in windows of 128 through {@link ForwardPassHandler#forwardVerify}.</li>
 * <li><b>Greedy agreement</b>: the {@code f16} handler continues the text's first {@value #PROMPT} tokens greedily for
 * {@value #GENERATED} tokens. That sequence is then teacher-forced through both handlers in the same windows, and the
 * share of the {@value #GENERATED} generated positions at which the two handlers' argmax agree is the agreement. Both
 * sides are read the same way, so the only difference between them is the KV type. Also reported: the first generated
 * position where they disagree (where a free-running quantized decode would first leave the {@code f16} reply), and
 * the argmax agreement over the text's first 512 positions (the integer-kernel quality test's reading).</li>
 * </ul>
 *
 * <p>Thresholds for {@value #GATED} (8-bit keys, 4-bit values, the 4-bit setting held to the bar; 4-bit keys lose
 * quality on some models and are a warned setting): NLL at most {@value #MAX_NLL_RATIO}x the {@code f16} NLL, and
 * greedy agreement at least {@value #MIN_GREEDY_AGREEMENT}. Other types named in {@code -Djuno.test.kvQuality.types}
 * (default {@value #GATED}; a type for both K and V, or {@code k/v}) are read and printed, not gated.
 * {@code -Djuno.test.kvQuality.models} (ids names, such as {@code tinyllama,mistral}) runs a subset. Runs only with
 * {@code -Djuno.test.kvQuality=true}.
 */
class KvQuantQualityLiveTest {

	static final Path MODELS = Path.of(System.getProperty("user.dir")).endsWith("node")
			? Path.of(System.getProperty("user.dir")).getParent().resolve("models")
			: Path.of("models");

	static final int TOKENS = 2048;
	static final int PROMPT = 1024;
	static final int GENERATED = 512;
	static final int TEXT_ARGMAX_POSITIONS = 512;
	static final int WINDOW = 128;
	static final double MAX_NLL_RATIO = 1.02;
	static final double MIN_GREEDY_AGREEMENT = 0.97;

	/** 8-bit keys with 4-bit values, the 4-bit setting held to the quality bar. */
	static final String GATED = "q8_0/q4_0";

	@ParameterizedTest(name = "{0}")
	@CsvSource({ "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf, tinyllama", "qwen2.5-3b-instruct-q4_k_m.gguf, qwen25",
			"Phi-3.5-mini-instruct-Q4_K_M.gguf, phi35", "mistral-7b-instruct-v0.1-q4_k_m.gguf, mistral" })
	@EnabledIfSystemProperty(named = "juno.test.kvQuality", matches = "true")
	void quantizedKvAgainstF16(String file, String idsName) throws Exception {
		Path model = MODELS.resolve(file);
		assumeTrue(Files.exists(model), "model not present: " + model);
		String only = System.getProperty("juno.test.kvQuality.models", "");
		assumeTrue(only.isBlank() || java.util.Arrays.asList(only.split(",")).contains(idsName),
				"not in juno.test.kvQuality.models: " + idsName);
		int[] text = TeacherForcedNll.readIds("model-quality-nll." + idsName + ".ids", TOKENS + 1);
		ShardContext ctx;
		try (GgufReader r = GgufReader.open(model)) {
			LlamaConfig c = LlamaConfig.from(r);
			ctx = new ShardContext("n0", 0, c.numLayers(), true, true, c.vocabSize(), c.hiddenDim(), c.numHeads());
		}

		long t0 = System.nanoTime();
		ForwardPassHandler f16 = load(model, ctx, "f16", "f16");
		Pass f16Text = teacherForce(f16, ctx, text, "kvq-text");
		int[] generated = greedy(f16, ctx, Arrays.copyOf(text, PROMPT), GENERATED, "kvq-gen");
		int[] sequence = concat(Arrays.copyOf(text, PROMPT), generated);
		Pass f16Seq = teacherForce(f16, ctx, sequence, "kvq-seq");
		f16 = null;
		double f16Seconds = (System.nanoTime() - t0) / 1e9;
		int selfAgree = agreement(f16Seq.argmax, generated);
		System.out.printf(Locale.ROOT, "KV-QUALITY %s type=f16 nll=%.5f ppl=%.4f self_agree_generated=%d/%d s=%.1f%n",
				file, f16Text.meanNll, Math.exp(f16Text.meanNll), selfAgree, GENERATED, f16Seconds);

		for (String type : System.getProperty("juno.test.kvQuality.types", GATED).split(",")) {
			type = type.strip();
			String kType = type.contains("/") ? type.substring(0, type.indexOf('/')) : type;
			String vType = type.contains("/") ? type.substring(type.indexOf('/') + 1) : type;
			System.gc();
			long t1 = System.nanoTime();
			ForwardPassHandler q = load(model, ctx, kType, vType);
			Pass qText = teacherForce(q, ctx, text, "kvq-text");
			Pass qSeq = teacherForce(q, ctx, sequence, "kvq-seq");
			q = null;
			double seconds = (System.nanoTime() - t1) / 1e9;

			double ratio = qText.meanNll / f16Text.meanNll;
			int agree = agreement(qSeq.argmax, f16Seq.argmax);
			int firstDisagree = firstDisagreement(qSeq.argmax, f16Seq.argmax);
			int textAgree = 0;
			for (int t = 0; t < TEXT_ARGMAX_POSITIONS; t++)
				if (qText.argmaxAll[t] == f16Text.argmaxAll[t])
					textAgree++;
			System.out.printf(Locale.ROOT,
					"KV-QUALITY %s type=%s nll=%.5f nll_f16=%.5f ratio=%.5f ppl=%.4f greedy_agree=%d/%d (%.2f%%) "
							+ "first_disagree=%d text_argmax_agree_first%d=%d (%.1f%%) s=%.1f%n",
					file, type, qText.meanNll, f16Text.meanNll, ratio, Math.exp(qText.meanNll), agree, GENERATED,
					100.0 * agree / GENERATED, firstDisagree, TEXT_ARGMAX_POSITIONS, textAgree,
					100.0 * textAgree / TEXT_ARGMAX_POSITIONS, seconds);
			if (type.equals(GATED)) {
				assertThat(ratio).as("%s NLL over f16 NLL on %s", type, file).isLessThanOrEqualTo(MAX_NLL_RATIO);
				assertThat((double) agree / GENERATED).as("%s greedy agreement on %s", type, file)
						.isGreaterThanOrEqualTo(MIN_GREEDY_AGREEMENT);
			}
		}
	}

	/** A handler whose host KV is {@code kType} and {@code vType}; the types are read when the handler is built. */
	private static ForwardPassHandler load(Path model, ShardContext ctx, String kType, String vType) throws Exception {
		String prevK = System.getProperty(CacheTypeOptions.ENV_K);
		String prevV = System.getProperty(CacheTypeOptions.ENV_V);
		System.setProperty(CacheTypeOptions.ENV_K, kType);
		System.setProperty(CacheTypeOptions.ENV_V, vType);
		try {
			return ForwardPassHandlerLoader.load(model, ctx, CpuMatVec.INSTANCE);
		} finally {
			restore(CacheTypeOptions.ENV_K, prevK);
			restore(CacheTypeOptions.ENV_V, prevV);
		}
	}

	/**
	 * Teacher-forced over {@code ids}: position {@code t} reads {@code ids[0..t]} and is scored on {@code ids[t + 1]}.
	 * {@code argmax} holds the positions that predict the generated part of a {@link #PROMPT}-token-prompt sequence
	 * (empty for a shorter one), {@code argmaxAll} every position.
	 */
	static Pass teacherForce(ForwardPassHandler handler, ShardContext ctx, int[] ids, String session) {
		int predictions = ids.length - 1;
		int[] all = new int[predictions];
		double nll = 0;
		try {
			for (int start = 0; start < predictions; start += WINDOW) {
				int w = Math.min(WINDOW, predictions - start);
				float[] logits = handler
						.forwardVerify(BatchForwardRequest.withTokens(session, Arrays.copyOfRange(ids, start, start + w),
								start), ctx)
						.logits();
				int vocab = logits.length / w;
				for (int i = 0; i < w; i++) {
					int off = i * vocab;
					int best = argmax(logits, off, vocab);
					double max = logits[off + best];
					double sum = 0;
					for (int v = 0; v < vocab; v++)
						sum += Math.exp(logits[off + v] - max);
					nll -= logits[off + ids[start + i + 1]] - max - Math.log(sum);
					all[start + i] = best;
				}
			}
		} finally {
			handler.evict(session);
		}
		int[] gen = ids.length == PROMPT + GENERATED ? Arrays.copyOfRange(all, PROMPT - 1, PROMPT - 1 + GENERATED)
				: new int[0];
		return new Pass(nll / predictions, gen, all);
	}

	/** {@code count} tokens continuing {@code prompt}, each the argmax of the previous position's logits. */
	static int[] greedy(ForwardPassHandler handler, ShardContext ctx, int[] prompt, int count, String session) {
		int[] out = new int[count];
		try {
			float[] last = null;
			for (int start = 0; start < prompt.length; start += WINDOW) {
				int w = Math.min(WINDOW, prompt.length - start);
				float[] logits = handler.forwardVerify(
						BatchForwardRequest.withTokens(session, Arrays.copyOfRange(prompt, start, start + w), start), ctx)
						.logits();
				last = Arrays.copyOfRange(logits, logits.length / w * (w - 1), logits.length);
			}
			out[0] = argmax(last, 0, last.length);
			for (int i = 1; i < count; i++) {
				float[] logits = handler.forwardVerify(
						BatchForwardRequest.withTokens(session, new int[] { out[i - 1] }, prompt.length + i - 1), ctx)
						.logits();
				out[i] = argmax(logits, 0, logits.length);
			}
		} finally {
			handler.evict(session);
		}
		return out;
	}

	private static int argmax(float[] a, int off, int n) {
		int best = 0;
		float max = Float.NEGATIVE_INFINITY;
		for (int v = 0; v < n; v++)
			if (a[off + v] > max) {
				max = a[off + v];
				best = v;
			}
		return best;
	}

	static int agreement(int[] a, int[] b) {
		int n = 0;
		for (int i = 0; i < a.length; i++)
			if (a[i] == b[i])
				n++;
		return n;
	}

	/** The first index where {@code a} and {@code b} differ, or their length when they never do. */
	static int firstDisagreement(int[] a, int[] b) {
		for (int i = 0; i < a.length; i++)
			if (a[i] != b[i])
				return i;
		return a.length;
	}

	static int[] concat(int[] a, int[] b) {
		int[] out = Arrays.copyOf(a, a.length + b.length);
		System.arraycopy(b, 0, out, a.length, b.length);
		return out;
	}

	static void restore(String key, String prev) {
		if (prev == null)
			System.clearProperty(key);
		else
			System.setProperty(key, prev);
	}

	/** Mean NLL over the pass; argmax at the generated positions; argmax at every position. */
	record Pass(double meanNll, int[] argmax, int[] argmaxAll) {
	}
}
