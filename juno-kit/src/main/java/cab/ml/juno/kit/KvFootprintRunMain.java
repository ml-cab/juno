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

import java.lang.management.ManagementFactory;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

import cab.ml.juno.kvcache.CacheTypeOptions;
import cab.ml.juno.node.BatchForwardRequest;
import cab.ml.juno.node.CpuMatVec;
import cab.ml.juno.node.CudaMatVec;
import cab.ml.juno.node.ForwardPassHandler;
import cab.ml.juno.node.ForwardPassHandlerLoader;
import cab.ml.juno.node.ForwardRequest;
import cab.ml.juno.node.GgufReader;
import cab.ml.juno.node.GpuContext;
import cab.ml.juno.node.LlamaConfig;
import cab.ml.juno.node.ShardContext;
import cab.ml.juno.tokenizer.GgufTokenizer;

/**
 * One host KV footprint run for {@link KvFootprintProbe}: {@code KvFootprintRunMain <model> <tokens> <generated> <gpu>},
 * the handler on the CPU backend or on CUDA device 0, with the KV types this JVM's {@code JUNO_CACHE_TYPE_K/V}
 * properties select. One request as the server runs it: the handler told the most positions it can write
 * ({@code tokens + generated}), the first {@code tokens} tokens of the quality text prefilled in one window, then
 * {@code generated} tokens decoded greedily one at a time. Prints one line after {@link #PREFIX}: the request's host KV
 * bytes, the live heap after load and with the request held (each after full collections), and what was open.
 */
public final class KvFootprintRunMain {

	static final String PREFIX = "JUNO-KIT-KV-FOOTPRINT ";

	private KvFootprintRunMain() {
	}

	public static void main(String[] args) throws Exception {
		Path model = Path.of(args[0]);
		int tokens = Integer.parseInt(args[1]);
		int generated = Integer.parseInt(args[2]);
		boolean gpu = Boolean.parseBoolean(args[3]);
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
		ForwardPassHandler handler = ForwardPassHandlerLoader.load(model, ctx,
				gpu ? new CudaMatVec(GpuContext.shared(0)) : CpuMatVec.INSTANCE);
		long afterLoad = liveHeap();

		String session = "kv-footprint";
		handler.reserveKv(session, tokens + generated);
		float[] logits = handler.forwardBatch(BatchForwardRequest.withTokens(session, prompt, 0), ctx).lastLogits();
		for (int i = 0; i < generated; i++)
			logits = handler.forward(ForwardRequest.withTokens(session, new int[] { argmax(logits) }, tokens + i), ctx)
					.logits();
		long withRequest = liveHeap();

		Map<String, Object> out = new LinkedHashMap<>();
		out.put("positions", tokens + generated);
		out.put("hostKvBytes", handler.hostKvBytes(session));
		out.put("hostKvBytesPerPosition", handler.hostKvBytesPerToken());
		out.put("liveHeapAfterLoadBytes", afterLoad);
		out.put("liveHeapWithRequestBytes", withRequest);
		out.put("maxHeapBytes", Runtime.getRuntime().maxMemory());
		out.put("kv", CacheTypeOptions.fromEnv().policySummary());
		out.put("gpuAttention", handler.gpuAttentionActive());
		handler.evict(session);
		System.out.println(PREFIX + Json.MAPPER.writer()
				.without(com.fasterxml.jackson.databind.SerializationFeature.INDENT_OUTPUT).writeValueAsString(out));
		System.exit(0);
	}

	/** Heap in use after two full collections: the live set. */
	static long liveHeap() {
		for (int i = 0; i < 2; i++)
			System.gc();
		return ManagementFactory.getMemoryMXBean().getHeapMemoryUsage().getUsed();
	}

	private static int argmax(float[] a) {
		int best = 0;
		for (int i = 1; i < a.length; i++)
			if (a[i] > a[best])
				best = i;
		return best;
	}
}
