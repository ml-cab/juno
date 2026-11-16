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
 * GPU prefill time of a long prompt, for one device KV type: the reading behind the quantized device KV's prefill bar
 * (a 2,048-token prefill with {@code q8_0} device KV at most 1.10x the FP16 copy's time). One JVM runs one model at
 * one type, so a caller alternates types between JVMs (A B A B A B) and compares medians.
 *
 * <p>Each repetition starts a session and prefills the model's quality text to {@code context} tokens through
 * {@link ForwardPassHandler#forwardBatch}, the serving prefill path (last position's logits only), in windows of each
 * width in {@code windows}: the whole prompt in one window (what the local GPU surface's default chunk gives when the
 * window fits) and narrower chunks ({@code --prefill-batch}). The wall time of the whole prefill is the reading; the
 * call returns host logits, so the device work is complete when it does. One warm-up repetition is discarded. Prints
 * one {@code DEVICE-KV-PP} line per repetition and width, and one with the median per width.
 *
 * <p>{@code -Djuno.test.deviceKvPp=true} enables it; {@code .models} (one id), {@code .type} ({@code f16}, a type, or
 * {@code k/v}), {@code .context} (default 2048), {@code .windows} (comma-separated widths, default {@code 2048,128}),
 * {@code .reps} (default 5).
 */
@Tag("gpu")
class DeviceKvPrefillTimeLiveTest {

	@ParameterizedTest(name = "{0}")
	@CsvSource({ "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf, tinyllama", "qwen2.5-3b-instruct-q4_k_m.gguf, qwen25",
			"Phi-3.5-mini-instruct-Q4_K_M.gguf, phi35", "mistral-7b-instruct-v0.1-q4_k_m.gguf, mistral" })
	@EnabledIfSystemProperty(named = "juno.test.deviceKvPp", matches = "true")
	void prefillOfALongPrompt(String file, String idsName) throws Exception {
		Path model = KvQuantQualityLiveTest.MODELS.resolve(file);
		assumeTrue(Files.exists(model), "model not present: " + model);
		assumeTrue(idsName.equals(System.getProperty("juno.test.deviceKvPp.models", "")),
				"one model per JVM: not juno.test.deviceKvPp.models");
		assumeTrue(CudaAvailability.isAvailable(), "no CUDA device");
		String type = System.getProperty("juno.test.deviceKvPp.type", "f16").strip();
		String kType = type.contains("/") ? type.substring(0, type.indexOf('/')) : type;
		String vType = type.contains("/") ? type.substring(type.indexOf('/') + 1) : type;
		int context = Integer.getInteger("juno.test.deviceKvPp.context", 2048);
		int[] windows = Arrays.stream(System.getProperty("juno.test.deviceKvPp.windows", "2048,128").split(","))
				.map(String::strip).mapToInt(Integer::parseInt).map(w -> Math.min(w, context)).toArray();
		int reps = Integer.getInteger("juno.test.deviceKvPp.reps", 5);

		int[] prompt = Arrays.copyOf(TeacherForcedNll.readIds("model-quality-nll." + idsName + ".ids", context + 1),
				context);
		ShardContext ctx;
		int headDim;
		try (GgufReader r = GgufReader.open(model)) {
			LlamaConfig c = LlamaConfig.from(r);
			ctx = new ShardContext("n0", 0, c.numLayers(), true, true, c.vocabSize(), c.hiddenDim(), c.numHeads());
			headDim = c.headDim();
		}
		DeviceKvFormats formats = DeviceKvFormats.select(
				CacheTypeOptions.of(KvElementType.parse(kType), KvElementType.parse(vType)), headDim,
				() -> GqaAttentionKvKernel.tryLoad() != null);
		ForwardPassHandler handler = DeviceKvDecodeTgLiveTest.load(model, ctx, kType, vType);
		assertThat(handler.gpuAttentionActive()).as("GPU attention on %s", file).isTrue();
		assertThat(handler.prefillRegionActive()).as("device prefill region on %s", file).isTrue();

		double[][] ms = new double[windows.length][reps];
		for (int rep = -1; rep < reps; rep++) {
			for (int wi = 0; wi < windows.length; wi++) {
				String session = "dkvpp-" + rep + "-" + windows[wi];
				try {
					long t0 = System.nanoTime();
					float[] logits = null;
					for (int start = 0; start < context; start += windows[wi]) {
						int w = Math.min(windows[wi], context - start);
						logits = handler.forwardBatch(
								BatchForwardRequest.withTokens(session, Arrays.copyOfRange(prompt, start, start + w), start),
								ctx).lastLogits();
					}
					double elapsed = (System.nanoTime() - t0) / 1e6;
					assertThat(logits).as("final-position logits").isNotNull();
					if (rep >= 0) {
						ms[wi][rep] = elapsed;
						System.out.printf(Locale.ROOT,
								"DEVICE-KV-PP %s type=%s device=%s context=%d window=%d rep=%d ms=%.2f%n", file, type,
								formats.summary(), context, windows[wi], rep, elapsed);
					}
				} finally {
					handler.evict(session);
				}
			}
		}
		assertThat(handler.prefillRegionActive()).as("device prefill region still open on %s", file).isTrue();
		for (int wi = 0; wi < windows.length; wi++) {
			double[] sorted = ms[wi].clone();
			Arrays.sort(sorted);
			System.out.printf(Locale.ROOT,
					"DEVICE-KV-PP %s type=%s device=%s context=%d window=%d median_ms=%.2f reps=%s%n", file, type,
					formats.summary(), context, windows[wi], sorted[reps / 2], Arrays.toString(ms[wi]));
		}
	}
}
