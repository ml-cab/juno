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
package cab.ml.juno.kit;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import cab.ml.juno.kvcache.CacheTypeOptions;
import cab.ml.juno.node.BatchForwardRequest;
import cab.ml.juno.node.CudaMatVec;
import cab.ml.juno.node.ForwardPassHandler;
import cab.ml.juno.node.ForwardPassHandlerLoader;
import cab.ml.juno.node.GgufReader;
import cab.ml.juno.node.GpuContext;
import cab.ml.juno.node.LlamaConfig;
import cab.ml.juno.node.ShardContext;
import cab.ml.juno.tokenizer.GgufTokenizer;

/**
 * One prefill timing run for {@link GpuKvPrefillProbe}: {@code KvPrefillRunMain <model> <tokens> <windows> <reps>},
 * the handler on CUDA device 0 with the KV types this JVM's {@code JUNO_CACHE_TYPE_K}/{@code JUNO_CACHE_TYPE_V}
 * properties select (on the GPU they also set the device copy GPU attention reads). Each repetition prefills the first
 * {@code tokens} tokens of the quality text in a new session through {@link ForwardPassHandler#forwardBatch}, the
 * serving prefill path, in windows of each width in {@code windows} (comma-separated), and takes the wall time of the
 * whole prefill; one warm-up repetition is discarded. Prints one line after {@link #PREFIX}: the times per width, the
 * handler's notice about the device KV copy, and whether GPU attention and the device prefill region were open.
 */
public final class KvPrefillRunMain {

	static final String PREFIX = "JUNO-KIT-KV-PREFILL ";

	private KvPrefillRunMain() {
	}

	public static void main(String[] args) throws Exception {
		Path model = Path.of(args[0]);
		int tokens = Integer.parseInt(args[1]);
		int[] windows = Arrays.stream(args[2].split(",")).map(String::strip).mapToInt(Integer::parseInt)
				.map(w -> Math.min(w, tokens)).toArray();
		int reps = Integer.parseInt(args[3]);
		ShardContext ctx;
		int[] ids;
		try (GgufReader r = GgufReader.open(model)) {
			LlamaConfig c = LlamaConfig.from(r);
			ctx = new ShardContext("n0", 0, c.numLayers(), true, true, c.vocabSize(), c.hiddenDim(), c.numHeads());
			ids = GgufTokenizer.load(r).encode(QualityProbe.text());
		}
		if (ids.length < tokens)
			throw new IllegalStateException("the text is " + ids.length + " tokens, " + tokens + " needed");
		int[] prompt = Arrays.copyOf(ids, tokens);
		NoticeCapture notices = NoticeCapture.install();
		ForwardPassHandler handler = ForwardPassHandlerLoader.load(model, ctx, new CudaMatVec(GpuContext.shared(0)));

		Map<String, List<Double>> ms = new LinkedHashMap<>();
		for (int w : windows)
			ms.put(String.valueOf(w), new ArrayList<>());
		for (int rep = -1; rep < reps; rep++) {
			for (int width : windows) {
				String session = "kv-prefill-" + rep + "-" + width;
				try {
					long t0 = System.nanoTime();
					for (int start = 0; start < tokens; start += width) {
						int w = Math.min(width, tokens - start);
						handler.forwardBatch(BatchForwardRequest.withTokens(session,
								Arrays.copyOfRange(prompt, start, start + w), start), ctx).lastLogits();
					}
					double elapsed = (System.nanoTime() - t0) / 1e6;
					if (rep >= 0)
						ms.get(String.valueOf(width)).add(elapsed);
				} finally {
					handler.evict(session);
				}
			}
		}
		Map<String, Object> out = new LinkedHashMap<>();
		out.put("ms", ms);
		out.put("gpuAttention", handler.gpuAttentionActive());
		out.put("prefillRegion", handler.prefillRegionActive());
		out.put("kv", CacheTypeOptions.fromEnv().policySummary());
		out.put("deviceKv", GpuKvQualityProbe.deviceKvNotice(notices.messages()));
		System.out.println(PREFIX + Json.MAPPER.writer()
				.without(com.fasterxml.jackson.databind.SerializationFeature.INDENT_OUTPUT).writeValueAsString(out));
		System.exit(0);
	}
}
