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
 * Model quality of the {@code f16} host KV (IEEE halves) against the {@code f32} host KV (float32 rows, what
 * {@code f16} stored before), on every sweep model, CPU backend so attention reads the host rows. The readings are
 * {@link KvQuantQualityLiveTest}'s: teacher-forced mean NLL over the 2,048-token quality text in windows of 128, and
 * greedy agreement over the {@value KvQuantQualityLiveTest#GENERATED} positions the {@code f32} handler generates
 * after the text's first {@value KvQuantQualityLiveTest#PROMPT} tokens, both handlers teacher-forced through that
 * sequence. Bars: NLL at most {@value #MAX_NLL_RATIO}x the {@code f32} NLL and agreement at least
 * {@value #MIN_GREEDY_AGREEMENT}. {@code -Djuno.test.hostKvHalfQuality.models} (ids names, such as
 * {@code tinyllama,mistral}) runs a subset. Runs only with {@code -Djuno.test.hostKvHalfQuality=true}.
 */
class HostKvHalfQualityLiveTest {

	static final double MAX_NLL_RATIO = 1.005;
	static final double MIN_GREEDY_AGREEMENT = 0.99;

	@ParameterizedTest(name = "{0}")
	@CsvSource({ "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf, tinyllama", "qwen2.5-3b-instruct-q4_k_m.gguf, qwen25",
			"Phi-3.5-mini-instruct-Q4_K_M.gguf, phi35", "mistral-7b-instruct-v0.1-q4_k_m.gguf, mistral" })
	@EnabledIfSystemProperty(named = "juno.test.hostKvHalfQuality", matches = "true")
	void halvesAgainstFloats(String file, String idsName) throws Exception {
		Path model = KvQuantQualityLiveTest.MODELS.resolve(file);
		assumeTrue(Files.exists(model), "model not present: " + model);
		String only = System.getProperty("juno.test.hostKvHalfQuality.models", "");
		assumeTrue(only.isBlank() || Arrays.asList(only.split(",")).contains(idsName),
				"not in juno.test.hostKvHalfQuality.models: " + idsName);
		int[] text = TeacherForcedNll.readIds("model-quality-nll." + idsName + ".ids", KvQuantQualityLiveTest.TOKENS + 1);
		ShardContext ctx;
		try (GgufReader r = GgufReader.open(model)) {
			LlamaConfig c = LlamaConfig.from(r);
			ctx = new ShardContext("n0", 0, c.numLayers(), true, true, c.vocabSize(), c.hiddenDim(), c.numHeads());
		}
		int prompt = KvQuantQualityLiveTest.PROMPT;
		int generatedCount = KvQuantQualityLiveTest.GENERATED;

		long t0 = System.nanoTime();
		ForwardPassHandler f32 = load(model, ctx, "f32");
		KvQuantQualityLiveTest.Pass f32Text = KvQuantQualityLiveTest.teacherForce(f32, ctx, text, "half-text");
		int[] generated = KvQuantQualityLiveTest.greedy(f32, ctx, Arrays.copyOf(text, prompt), generatedCount, "half-gen");
		int[] sequence = KvQuantQualityLiveTest.concat(Arrays.copyOf(text, prompt), generated);
		KvQuantQualityLiveTest.Pass f32Seq = KvQuantQualityLiveTest.teacherForce(f32, ctx, sequence, "half-seq");
		f32 = null;
		double f32Seconds = (System.nanoTime() - t0) / 1e9;

		System.gc();
		long t1 = System.nanoTime();
		ForwardPassHandler f16 = load(model, ctx, "f16");
		KvQuantQualityLiveTest.Pass f16Text = KvQuantQualityLiveTest.teacherForce(f16, ctx, text, "half-text");
		KvQuantQualityLiveTest.Pass f16Seq = KvQuantQualityLiveTest.teacherForce(f16, ctx, sequence, "half-seq");
		f16 = null;
		double f16Seconds = (System.nanoTime() - t1) / 1e9;

		double ratio = f16Text.meanNll() / f32Text.meanNll();
		int agree = KvQuantQualityLiveTest.agreement(f16Seq.argmax(), f32Seq.argmax());
		int firstDisagree = KvQuantQualityLiveTest.firstDisagreement(f16Seq.argmax(), f32Seq.argmax());
		int textAgree = 0;
		int textPositions = KvQuantQualityLiveTest.TEXT_ARGMAX_POSITIONS;
		for (int t = 0; t < textPositions; t++)
			if (f16Text.argmaxAll()[t] == f32Text.argmaxAll()[t])
				textAgree++;
		System.out.printf(Locale.ROOT,
				"HOST-KV-HALF %s nll_f16=%.5f nll_f32=%.5f ratio=%.5f greedy_agree=%d/%d (%.2f%%) first_disagree=%d "
						+ "text_argmax_agree_first%d=%d (%.1f%%) s_f32=%.1f s_f16=%.1f%n",
				file, f16Text.meanNll(), f32Text.meanNll(), ratio, agree, generatedCount, 100.0 * agree / generatedCount,
				firstDisagree, textPositions, textAgree, 100.0 * textAgree / textPositions, f32Seconds, f16Seconds);
		assertThat(ratio).as("f16 NLL over f32 NLL on %s", file).isLessThanOrEqualTo(MAX_NLL_RATIO);
		assertThat((double) agree / generatedCount).as("f16 greedy agreement with f32 on %s", file)
				.isGreaterThanOrEqualTo(MIN_GREEDY_AGREEMENT);
	}

	/** A CPU handler whose host KV is {@code type} for K and V; the type is read when the handler is built. */
	private static ForwardPassHandler load(Path model, ShardContext ctx, String type) throws Exception {
		String prevK = System.getProperty(CacheTypeOptions.ENV_K);
		String prevV = System.getProperty(CacheTypeOptions.ENV_V);
		System.setProperty(CacheTypeOptions.ENV_K, type);
		System.setProperty(CacheTypeOptions.ENV_V, type);
		try {
			return ForwardPassHandlerLoader.load(model, ctx, CpuMatVec.INSTANCE);
		} finally {
			KvQuantQualityLiveTest.restore(CacheTypeOptions.ENV_K, prevK);
			KvQuantQualityLiveTest.restore(CacheTypeOptions.ENV_V, prevV);
		}
	}
}
