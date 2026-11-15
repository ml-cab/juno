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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import cab.ml.juno.kvcache.CacheTypeOptions;
import cab.ml.juno.kvcache.KvElementType;

/**
 * GPU decode throughput at a long context, for one device KV type: the reading behind the quantized device KV's decode
 * bar (tg at a 2,048-token context with {@code q8_0} device KV at least 0.95x the FP16 copy). One JVM runs one model at
 * one type, so a caller alternates types between JVMs (A B A B A B) and compares medians.
 *
 * <p>Each repetition starts a session, prefills the model's quality text to {@code context} tokens in windows of 128,
 * then times {@code decode} greedy single-token steps through {@link ForwardPassHandler#forward}, the serving decode
 * path, which attends over the whole context in the device KV. One warm-up repetition is discarded. Prints one
 * {@code DEVICE-KV-TG} line per repetition and one with the median.
 *
 * <p>{@code -Djuno.test.deviceKvTg=true} enables it; {@code .models} (one id), {@code .type} ({@code f16}, a type, or
 * {@code k/v}), {@code .context} (default 2048), {@code .decode} (default 128), {@code .reps} (default 3).
 */
@Tag("gpu")
class DeviceKvDecodeTgLiveTest {

	@ParameterizedTest(name = "{0}")
	@CsvSource({ "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf, tinyllama", "qwen2.5-3b-instruct-q4_k_m.gguf, qwen25",
			"Phi-3.5-mini-instruct-Q4_K_M.gguf, phi35", "mistral-7b-instruct-v0.1-q4_k_m.gguf, mistral" })
	@EnabledIfSystemProperty(named = "juno.test.deviceKvTg", matches = "true")
	void decodeAtLongContext(String file, String idsName) throws Exception {
		Path model = KvQuantQualityLiveTest.MODELS.resolve(file);
		assumeTrue(Files.exists(model), "model not present: " + model);
		assumeTrue(idsName.equals(System.getProperty("juno.test.deviceKvTg.models", "")),
				"one model per JVM: not juno.test.deviceKvTg.models");
		assumeTrue(CudaAvailability.isAvailable(), "no CUDA device");
		String type = System.getProperty("juno.test.deviceKvTg.type", "f16").strip();
		String kType = type.contains("/") ? type.substring(0, type.indexOf('/')) : type;
		String vType = type.contains("/") ? type.substring(type.indexOf('/') + 1) : type;
		int context = Integer.getInteger("juno.test.deviceKvTg.context", 2048);
		int decode = Integer.getInteger("juno.test.deviceKvTg.decode", 128);
		int reps = Integer.getInteger("juno.test.deviceKvTg.reps", 3);

		int[] text = TeacherForcedNll.readIds("model-quality-nll." + idsName + ".ids", context + 1);
		int[] prompt = Arrays.copyOf(text, context);
		ShardContext ctx;
		int kvDim;
		int headDim;
		try (GgufReader r = GgufReader.open(model)) {
			LlamaConfig c = LlamaConfig.from(r);
			ctx = new ShardContext("n0", 0, c.numLayers(), true, true, c.vocabSize(), c.hiddenDim(), c.numHeads());
			kvDim = c.kvDim();
			headDim = c.headDim();
		}
		DeviceKvFormats formats = DeviceKvFormats.select(
				CacheTypeOptions.of(KvElementType.parse(kType), KvElementType.parse(vType)), headDim,
				() -> GqaAttentionKvKernel.tryLoad() != null);
		ForwardPassHandler handler = load(model, ctx, kType, vType);
		assertThat(handler.gpuAttentionActive()).as("GPU attention on %s", file).isTrue();
		// Session capacity: whole sessions of this context whose device KV fits in what the weights left free.
		long free = GpuContext.shared(0).freeVramBytes();
		System.out.printf(Locale.ROOT,
				"DEVICE-KV-CAPACITY %s type=%s device=%s free_after_load=%d session_bytes=%d sessions=%d "
						+ "fp16_session_bytes=%d fp16_sessions=%d%n",
				file, type, formats.summary(), free, formats.sessionBytes(ctx.endLayer() - ctx.startLayer(), kvDim, context),
				formats.sessionsThatFit(free, ctx.endLayer() - ctx.startLayer(), kvDim, context),
				DeviceKvFormats.F16.sessionBytes(ctx.endLayer() - ctx.startLayer(), kvDim, context),
				DeviceKvFormats.F16.sessionsThatFit(free, ctx.endLayer() - ctx.startLayer(), kvDim, context));

		double[] tg = new double[reps];
		for (int rep = -1; rep < reps; rep++) {
			String session = "dkvtg-" + rep;
			try {
				float[] last = null;
				for (int start = 0; start < context; start += KvQuantQualityLiveTest.WINDOW) {
					int w = Math.min(KvQuantQualityLiveTest.WINDOW, context - start);
					float[] logits = handler.forwardVerify(
							BatchForwardRequest.withTokens(session, Arrays.copyOfRange(prompt, start, start + w), start),
							ctx).logits();
					last = Arrays.copyOfRange(logits, logits.length / w * (w - 1), logits.length);
				}
				int token = argmax(last);
				long t0 = System.nanoTime();
				for (int i = 0; i < decode; i++) {
					float[] logits = handler.forward(ForwardRequest.withTokens(session, new int[] { token }, context + i),
							ctx).logits();
					token = argmax(logits);
				}
				double seconds = (System.nanoTime() - t0) / 1e9;
				if (rep >= 0) {
					tg[rep] = decode / seconds;
					System.out.printf(Locale.ROOT, "DEVICE-KV-TG %s type=%s device=%s context=%d decode=%d rep=%d tg=%.3f%n",
							file, type, formats.summary(), context, decode, rep, tg[rep]);
				}
			} finally {
				handler.evict(session);
			}
		}
		double[] sorted = tg.clone();
		Arrays.sort(sorted);
		System.out.printf(Locale.ROOT, "DEVICE-KV-TG %s type=%s device=%s context=%d decode=%d median_tg=%.3f reps=%s%n",
				file, type, formats.summary(), context, decode, sorted[reps / 2], Arrays.toString(tg));
	}

	private static ForwardPassHandler load(Path model, ShardContext ctx, String kType, String vType) throws Exception {
		String prevK = System.getProperty(CacheTypeOptions.ENV_K);
		String prevV = System.getProperty(CacheTypeOptions.ENV_V);
		System.setProperty(CacheTypeOptions.ENV_K, kType);
		System.setProperty(CacheTypeOptions.ENV_V, vType);
		try {
			return ForwardPassHandlerLoader.load(model, ctx, new CudaMatVec(GpuContext.shared(0)));
		} finally {
			KvQuantQualityLiveTest.restore(CacheTypeOptions.ENV_K, prevK);
			KvQuantQualityLiveTest.restore(CacheTypeOptions.ENV_V, prevV);
		}
	}

	private static int argmax(float[] a) {
		int best = 0;
		for (int i = 1; i < a.length; i++)
			if (a[i] > a[best])
				best = i;
		return best;
	}
}
