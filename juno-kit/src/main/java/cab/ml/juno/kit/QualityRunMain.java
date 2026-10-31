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
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

import cab.ml.juno.node.BatchForwardRequest;
import cab.ml.juno.node.CpuMatVec;
import cab.ml.juno.node.ForwardPassHandler;
import cab.ml.juno.node.ForwardPassHandlerLoader;
import cab.ml.juno.node.GgufReader;
import cab.ml.juno.node.LlamaConfig;
import cab.ml.juno.node.ShardContext;
import cab.ml.juno.tokenizer.GgufTokenizer;

/**
 * One kernel setting's NLL score for {@link QualityProbe}: {@code QualityRunMain <model> <tokens>}, on
 * whichever CPU kernels this JVM's {@code juno.cpu.integerKernels} selects. Prints one line after
 * {@link #PREFIX}.
 */
public final class QualityRunMain {

	static final String PREFIX = "JUNO-KIT-NLL ";
	static final int WINDOW = 128;

	private QualityRunMain() {
	}

	public static void main(String[] args) throws Exception {
		Path model = Path.of(args[0]);
		int tokens = Integer.parseInt(args[1]);
		ShardContext ctx;
		int[] ids;
		try (GgufReader r = GgufReader.open(model)) {
			LlamaConfig c = LlamaConfig.from(r);
			ctx = new ShardContext("n0", 0, c.numLayers(), true, true, c.vocabSize(), c.hiddenDim(), c.numHeads());
			ids = GgufTokenizer.load(r).encode(QualityProbe.text());
		}
		if (ids.length < tokens + 1)
			throw new IllegalStateException("the text is " + ids.length + " tokens, " + (tokens + 1) + " needed");
		ids = Arrays.copyOf(ids, tokens + 1);
		ForwardPassHandler handler = ForwardPassHandlerLoader.load(model, ctx, CpuMatVec.INSTANCE);
		long t0 = System.nanoTime();
		int predictions = ids.length - 1;
		int[] argmax = new int[Math.min(QualityProbe.ARGMAX_POSITIONS, predictions)];
		double nll = 0;
		for (int start = 0; start < predictions; start += WINDOW) {
			int w = Math.min(WINDOW, predictions - start);
			int[] window = Arrays.copyOfRange(ids, start, start + w);
			float[] logits = handler.forwardVerify(BatchForwardRequest.withTokens("nll", window, start), ctx).logits();
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
				if (start + i < argmax.length)
					argmax[start + i] = best;
			}
		}
		Map<String, Object> out = new LinkedHashMap<>();
		out.put("meanNll", nll / predictions);
		out.put("argmax", argmax);
		out.put("seconds", (System.nanoTime() - t0) / 1e9);
		out.put("kernels", "false".equalsIgnoreCase(System.getProperty("juno.cpu.integerKernels")) ? "float-activation"
				: "integer where the Vector API is present");
		System.out.println(PREFIX + Json.MAPPER.writer()
				.without(com.fasterxml.jackson.databind.SerializationFeature.INDENT_OUTPUT).writeValueAsString(out));
		System.exit(0);
	}
}
