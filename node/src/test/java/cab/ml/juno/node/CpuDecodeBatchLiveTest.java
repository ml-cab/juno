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
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/**
 * Real weights, the default CPU integer kernels: a decode batch gives every request, bit for bit, the logits its
 * single-request decode gives, step after step, so the KV each path writes is the same too. Before the decode-batch
 * matmul, the batch ran the prefill gemm, whose rows differ from the single-row kernel's in the last bits; the
 * layers' 8-bit activation rounding carried that to the logits, and a greedy answer at a near tie changed with what
 * else was being served. Each case skips when its model file is absent; the Qwen3-MoE case is opt-in.
 */
@DisplayName("CPU decode batch against single-request decode - real models, bit for bit")
class CpuDecodeBatchLiveTest {

	/** Three prompts of different lengths, so the requests sit at different positions. Ids valid in every vocabulary. */
	private static final int[][] PROMPTS = { { 1, 450, 7483, 310 }, { 1, 3532, 338, 263, 717, 892, 920 },
			{ 1, 871, 13, 277, 99 } };
	private static final int STEPS = 3;

	private static ShardContext shard(Path path) throws Exception {
		try (GgufReader r = GgufReader.open(path)) {
			LlamaConfig cfg = LlamaConfig.from(r);
			return new ShardContext("n0", 0, cfg.numLayers(), true, true, cfg.vocabSize(), cfg.hiddenDim(),
					cfg.numHeads());
		}
	}

	private static Path present(String file) {
		Path path = ContextShiftLiveTest.model(file);
		assumeTrue(path.toFile().exists(), file + " not present");
		return path;
	}

	/**
	 * Requests {@code s0..s2} decode one at a time and {@code b0..b2} as one batch, from identical prompt histories
	 * written by the single-request path; each step feeds both sets the serial argmax.
	 */
	private static void assertBatchEqualsSingle(ForwardPassHandler h, ShardContext ctx) {
		int n = PROMPTS.length;
		List<String> batchIds = new ArrayList<>();
		for (int i = 0; i < n; i++) {
			batchIds.add("b" + i);
			for (String id : List.of("s" + i, "b" + i))
				for (int p = 0; p < PROMPTS[i].length - 1; p++)
					h.forward(ForwardRequest.withTokens(id, new int[] { PROMPTS[i][p] }, p), ctx);
		}
		int[] next = new int[n];
		int[] pos = new int[n];
		for (int i = 0; i < n; i++) {
			next[i] = PROMPTS[i][PROMPTS[i].length - 1];
			pos[i] = PROMPTS[i].length - 1;
		}
		for (int step = 0; step < STEPS; step++) {
			float[][] serial = new float[n][];
			for (int i = 0; i < n; i++)
				serial[i] = h.forward(ForwardRequest.withTokens("s" + i, new int[] { next[i] }, pos[i]), ctx).logits();
			float[][] batched = h
					.forwardMultiDecode(MultiDecodeForwardRequest.withTokens(batchIds, next.clone(), pos.clone()), ctx)
					.logits();
			for (int i = 0; i < n; i++) {
				int differing = 0;
				double worst = 0;
				for (int v = 0; v < serial[i].length; v++)
					if (Float.floatToRawIntBits(serial[i][v]) != Float.floatToRawIntBits(batched[i][v])) {
						differing++;
						worst = Math.max(worst, Math.abs(serial[i][v] - batched[i][v]));
					}
				assertThat(differing).as("step %d, request %d: logits differing from single-request decode (max |diff| %s)",
						step, i, worst).isZero();
			}
			for (int i = 0; i < n; i++) {
				next[i] = argmax(serial[i]);
				pos[i]++;
			}
		}
		for (int i = 0; i < n; i++) {
			h.evict("s" + i);
			h.evict("b" + i);
		}
	}

	private static int argmax(float[] a) {
		int best = 0;
		for (int i = 1; i < a.length; i++)
			if (a[i] > a[best])
				best = i;
		return best;
	}

	@Test
	@DisplayName("LLaMA handler: TinyLlama 1.1B")
	void tinyllama() throws Exception {
		Path path = present("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
		ShardContext ctx = shard(path);
		assertBatchEqualsSingle(LlamaTransformerHandler.load(path, ctx, CpuMatVec.INSTANCE), ctx);
	}

	@Test
	@DisplayName("LLaMA handler, Qwen2 family: Qwen2.5-3B")
	void qwen25() throws Exception {
		Path path = present("qwen2.5-3b-instruct-q4_k_m.gguf");
		ShardContext ctx = shard(path);
		assertBatchEqualsSingle(LlamaTransformerHandler.load(path, ctx, CpuMatVec.INSTANCE), ctx);
	}

	@Test
	@DisplayName("Phi-3 handler: Phi-3.5-mini")
	void phi35() throws Exception {
		Path path = present("Phi-3.5-mini-instruct-Q4_K_M.gguf");
		ShardContext ctx = shard(path);
		assertBatchEqualsSingle(Phi3TransformerHandler.load(path, ctx, CpuMatVec.INSTANCE), ctx);
	}

	@Test
	@DisplayName("Qwen3 handler: Qwen3-1.7B")
	void qwen3() throws Exception {
		Path path = present("Qwen3-1.7B-Q4_K_M.gguf");
		ShardContext ctx = shard(path);
		assertBatchEqualsSingle(Qwen3TransformerHandler.load(path, ctx, CpuMatVec.INSTANCE), ctx);
	}

	@Test
	@DisplayName("Phi-2 handler: moondream2's text backbone (Q5_K weights)")
	void phi2() throws Exception {
		Path path = present("moondream2-q5_k.llamafile");
		ShardContext ctx = shard(path);
		assertBatchEqualsSingle(Phi2TransformerHandler.load(path, ctx, CpuMatVec.INSTANCE), ctx);
	}

	/** Opt-in: 18.6 GB. Run with {@code -Djuno.test.largeModels=true} and a heap of about 40 GB. */
	@Test
	@EnabledIfSystemProperty(named = "juno.test.largeModels", matches = "true")
	@DisplayName("Qwen3-MoE handler: Qwen3-Coder-30B-A3B")
	void qwen3Moe() throws Exception {
		Path path = present("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf");
		ShardContext ctx = shard(path);
		assertBatchEqualsSingle(Qwen3MoeTransformerHandler.load(path, ctx, CpuMatVec.INSTANCE), ctx);
	}
}
