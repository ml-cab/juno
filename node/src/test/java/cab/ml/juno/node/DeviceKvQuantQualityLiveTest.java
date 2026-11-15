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
import java.util.Properties;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import cab.ml.juno.kvcache.CacheTypeOptions;
import cab.ml.juno.kvcache.KvElementType;

/**
 * Model quality of a quantized device KV against the FP16 device KV, on the GPU, on every sweep model: the reading
 * {@link KvQuantQualityLiveTest} takes on the CPU, with attention reading the device copy that
 * {@code --cache-type-k/-v} now selects.
 *
 * <p>Two handlers of a 7B model do not fit the card together and a handler has no way to release its device memory,
 * so one JVM runs one model at one cache type ({@code -Djuno.test.deviceKvQuality.models}, one id;
 * {@code -Djuno.test.deviceKvQuality.type}, {@code f16}, a type, or {@code k/v}). The {@code f16} run saves its
 * reference (teacher-forced NLL over the 2,048-token text, the greedy continuation of the text's first 1,024 tokens,
 * the argmax along it) under {@code -Djuno.test.deviceKvQuality.dir}; a quantized run reads it and reports:
 * <ul>
 * <li><b>Greedy agreement</b>, the gate: the share of the 512 generated positions where the quantized handler's argmax,
 * teacher-forced along the reference continuation in the same windows of 128, equals the FP16 handler's. These
 * windows run the prefill path, so attention reads the device KV at prefill width.</li>
 * <li>NLL against the FP16 run, and the free-running quantized reply's agreement with the reference reply (decode
 * width): how many of its 512 tokens match before the first difference. Reported, not gated.</li>
 * <li>The device KV the handler allocated for a 128-token window, which must be the selected formats' size: a
 * handler that silently kept FP16 fails here.</li>
 * </ul>
 * Bars: at least {@value #MIN_Q8} at {@code q8_0}; at least {@value #MIN_Q4} where a side is {@code q4_0} (scored on
 * {@code q8_0} keys with {@code q4_0} values). Runs only with {@code -Djuno.test.deviceKvQuality=true} on a CUDA host.
 */
@Tag("gpu")
class DeviceKvQuantQualityLiveTest {

	static final double MIN_Q8 = 0.99;
	static final double MIN_Q4 = 0.97;

	@ParameterizedTest(name = "{0}")
	@CsvSource({ "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf, tinyllama", "qwen2.5-3b-instruct-q4_k_m.gguf, qwen25",
			"Phi-3.5-mini-instruct-Q4_K_M.gguf, phi35", "mistral-7b-instruct-v0.1-q4_k_m.gguf, mistral" })
	@EnabledIfSystemProperty(named = "juno.test.deviceKvQuality", matches = "true")
	void quantizedDeviceKvAgainstFp16(String file, String idsName) throws Exception {
		Path model = KvQuantQualityLiveTest.MODELS.resolve(file);
		assumeTrue(Files.exists(model), "model not present: " + model);
		assumeTrue(idsName.equals(System.getProperty("juno.test.deviceKvQuality.models", "")),
				"one model per JVM: not juno.test.deviceKvQuality.models");
		assumeTrue(CudaAvailability.isAvailable(), "no CUDA device");
		String type = System.getProperty("juno.test.deviceKvQuality.type", "f16").strip();
		String kType = type.contains("/") ? type.substring(0, type.indexOf('/')) : type;
		String vType = type.contains("/") ? type.substring(type.indexOf('/') + 1) : type;
		Path dir = Path.of(System.getProperty("juno.test.deviceKvQuality.dir", "target/device-kv-quality"));
		Files.createDirectories(dir);
		Path reference = dir.resolve(idsName + ".f16.properties");

		int[] text = TeacherForcedNll.readIds("model-quality-nll." + idsName + ".ids", KvQuantQualityLiveTest.TOKENS + 1);
		ShardContext ctx;
		int layers;
		int kvDim;
		int headDim;
		try (GgufReader r = GgufReader.open(model)) {
			LlamaConfig c = LlamaConfig.from(r);
			ctx = new ShardContext("n0", 0, c.numLayers(), true, true, c.vocabSize(), c.hiddenDim(), c.numHeads());
			layers = c.numLayers();
			kvDim = c.kvDim();
			headDim = c.headDim();
		}
		long t0 = System.nanoTime();
		ForwardPassHandler handler = load(model, ctx, kType, vType);
		DeviceKvFormats formats = DeviceKvFormats.select(
				CacheTypeOptions.of(KvElementType.parse(kType), KvElementType.parse(vType)), headDim,
				() -> GqaAttentionKvKernel.tryLoad() != null);
		long deviceBytes = deviceKvBytesForAWindow(handler, ctx, text);
		assertThat(deviceBytes).as("device KV of one 128-token window, %s", formats.summary())
				.isEqualTo(formats.sessionBytes(layers, kvDim, KvQuantQualityLiveTest.WINDOW));

		int prompt = KvQuantQualityLiveTest.PROMPT;
		int generated = KvQuantQualityLiveTest.GENERATED;
		KvQuantQualityLiveTest.Pass textPass = KvQuantQualityLiveTest.teacherForce(handler, ctx, text, "dkvq-text");
		if (type.equals("f16")) {
			int[] reply = KvQuantQualityLiveTest.greedy(handler, ctx, Arrays.copyOf(text, prompt), generated, "dkvq-gen");
			int[] sequence = KvQuantQualityLiveTest.concat(Arrays.copyOf(text, prompt), reply);
			KvQuantQualityLiveTest.Pass seq = KvQuantQualityLiveTest.teacherForce(handler, ctx, sequence, "dkvq-seq");
			Properties p = new Properties();
			p.setProperty("nll", Double.toString(textPass.meanNll()));
			p.setProperty("reply", join(reply));
			p.setProperty("argmax", join(seq.argmax()));
			try (var out = Files.newBufferedWriter(reference)) {
				p.store(out, "FP16 device KV reference, " + file);
			}
			System.out.printf(Locale.ROOT,
					"DEVICE-KV-QUALITY %s type=f16 device=%s device_bytes_window=%d nll=%.5f self_agree=%d/%d s=%.1f%n",
					file, formats.summary(), deviceBytes, textPass.meanNll(),
					KvQuantQualityLiveTest.agreement(seq.argmax(), reply), generated, (System.nanoTime() - t0) / 1e9);
			return;
		}

		assumeTrue(Files.exists(reference), "run the f16 reference first: " + reference);
		Properties p = new Properties();
		try (var in = Files.newBufferedReader(reference)) {
			p.load(in);
		}
		double nllF16 = Double.parseDouble(p.getProperty("nll"));
		int[] reply = split(p.getProperty("reply"));
		int[] argmaxF16 = split(p.getProperty("argmax"));
		int[] sequence = KvQuantQualityLiveTest.concat(Arrays.copyOf(text, prompt), reply);
		KvQuantQualityLiveTest.Pass seq = KvQuantQualityLiveTest.teacherForce(handler, ctx, sequence, "dkvq-seq");
		int[] ownReply = KvQuantQualityLiveTest.greedy(handler, ctx, Arrays.copyOf(text, prompt), generated, "dkvq-gen");
		int agree = KvQuantQualityLiveTest.agreement(seq.argmax(), argmaxF16);
		int firstDisagree = KvQuantQualityLiveTest.firstDisagreement(seq.argmax(), argmaxF16);
		int samePrefix = KvQuantQualityLiveTest.firstDisagreement(ownReply, reply);
		double ratio = textPass.meanNll() / nllF16;
		System.out.printf(Locale.ROOT,
				"DEVICE-KV-QUALITY %s type=%s device=%s device_bytes_window=%d nll=%.5f nll_f16=%.5f ratio=%.5f "
						+ "greedy_agree=%d/%d (%.2f%%) first_disagree=%d free_running_same_prefix=%d/%d s=%.1f%n",
				file, type, formats.summary(), deviceBytes, textPass.meanNll(), nllF16, ratio, agree, generated,
				100.0 * agree / generated, firstDisagree, samePrefix, generated,
				(System.nanoTime() - t0) / 1e9);
		double bar = formats.k() == DeviceKvFormat.Q4_0 || formats.v() == DeviceKvFormat.Q4_0 ? MIN_Q4 : MIN_Q8;
		assertThat((double) agree / generated).as("%s greedy agreement with the FP16 device KV on %s", type, file)
				.isGreaterThanOrEqualTo(bar);
	}

	/**
	 * Device KV bytes the handler holds for one session after its first {@code WINDOW}-token window: the mirrors grow
	 * from 64 to 128 positions, so this is layers x 128 x the formats' bytes per position.
	 */
	private static long deviceKvBytesForAWindow(ForwardPassHandler handler, ShardContext ctx, int[] text) {
		long before = DeviceKvCache.allocatedBytes();
		try {
			handler.forwardVerify(BatchForwardRequest.withTokens("dkvq-size",
					Arrays.copyOf(text, KvQuantQualityLiveTest.WINDOW), 0), ctx);
			return DeviceKvCache.allocatedBytes() - before;
		} finally {
			handler.evict("dkvq-size");
		}
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

	private static String join(int[] a) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < a.length; i++)
			sb.append(i == 0 ? "" : ",").append(a[i]);
		return sb.toString();
	}

	private static int[] split(String s) {
		return Arrays.stream(s.split(",")).mapToInt(Integer::parseInt).toArray();
	}
}
