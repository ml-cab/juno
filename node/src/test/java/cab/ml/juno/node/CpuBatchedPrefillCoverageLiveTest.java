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

import java.nio.file.Path;
import java.util.Set;
import java.util.TreeSet;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * A CPU prefill window runs the batched integer matmul for every weight format a
 * transformer layer of the model carries, and agrees with token-by-token decode.
 *
 * <p>Before this, Phi-3's CPU window path ran one decode-width matVec per
 * prompt token, and Q6_K weights had no batched kernel on any handler; both read
 * every weight once per prompt token. The test counts {@link CpuIntegerGemm}
 * calls per format during one window and requires every block-weight format of
 * the model to appear, then compares the window's last-position logits with the
 * same prompt fed one token at a time: same greedy token, and logits within 2%
 * of their largest magnitude (both paths quantize each activation row the same
 * way; only float summation order separates them, and the 8-bit rounding can
 * turn that into a step).
 */
class CpuBatchedPrefillCoverageLiveTest {

	private static final int PROMPT = 40;

	@ParameterizedTest(name = "{0}")
	@CsvSource({ "Phi-3.5-mini-instruct-Q4_K_M.gguf", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" })
	void windowTakesTheBatchedKernelForEveryFormatAndMatchesDecode(String file) throws Exception {
		Path model = ContextShiftLiveTest.model(file);
		assumeTrue(model.toFile().exists(), file + " not present");
		Set<Integer> layerFormats = new TreeSet<>();
		ShardContext ctx;
		try (GgufReader r = GgufReader.open(model)) {
			for (String name : r.tensorNames())
				if (name.startsWith("blk.") && name.endsWith(".weight") && CpuKernelSelection.integerFormat(r.tensorType(name)))
					layerFormats.add(r.tensorType(name));
			LlamaConfig c = LlamaConfig.from(r);
			ctx = new ShardContext("n0", 0, c.numLayers(), true, true, c.vocabSize(), c.hiddenDim(), c.numHeads());
		}
		assertThat(layerFormats).as("layer weight formats of %s", file).isNotEmpty();
		ForwardPassHandler h = ForwardPassHandlerLoader.load(model, ctx, CpuMatVec.INSTANCE);
		int[] prompt = new int[PROMPT];
		for (int i = 0; i < PROMPT; i++)
			prompt[i] = ContextShiftLiveCheck.token(i);
		try {
			CpuIntegerGemm.resetCalls();
			float[] window = h.forwardBatch(BatchForwardRequest.withTokens("window", prompt, 0), ctx).lastLogits();
			for (int type : layerFormats)
				assertThat(CpuIntegerGemm.calls(type)).as("batched matmul calls for type %d in %s", type, file)
						.isGreaterThan(0);

			float[] step = null;
			for (int i = 0; i < PROMPT; i++)
				step = h.forward(ForwardRequest.withTokens("steps", new int[] { prompt[i] }, i), ctx).logits();
			int a = argmax(window);
			int b = argmax(step);
			float scale = 0f;
			float diff = 0f;
			for (int i = 0; i < window.length; i++) {
				scale = Math.max(scale, Math.abs(step[i]));
				diff = Math.max(diff, Math.abs(window[i] - step[i]));
			}
			System.out.printf("%s: formats %s, window vs steps max |logit diff| %.4f of max %.2f, argmax %d / %d%n", file,
					layerFormats, diff, scale, a, b);
			assertThat(a).as("greedy token, window against steps").isEqualTo(b);
			assertThat(diff).as("max |logit diff|, window against steps").isLessThanOrEqualTo(0.02f * scale);
		} finally {
			h.evict("window");
			h.evict("steps");
			h.releaseGpuResources();
		}
	}

	private static int argmax(float[] v) {
		int best = 0;
		for (int i = 1; i < v.length; i++)
			if (v[i] > v[best])
				best = i;
		return best;
	}
}
