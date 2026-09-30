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
package cab.ml.juno.master;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Path;
import java.util.Locale;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import cab.ml.juno.node.BatchForwardRequest;
import cab.ml.juno.node.CudaAvailability;
import cab.ml.juno.node.CudaMatVec;
import cab.ml.juno.node.ForwardPassHandler;
import cab.ml.juno.node.ForwardPassHandlerLoader;
import cab.ml.juno.node.ForwardRequest;
import cab.ml.juno.node.GgufReader;
import cab.ml.juno.node.GpuAttentionOptions;
import cab.ml.juno.node.GpuContext;
import cab.ml.juno.node.LlamaConfig;
import cab.ml.juno.node.ShardContext;
import cab.ml.juno.tokenizer.GgufTokenizer;

/**
 * How far greedy decoding with the GPU attention kernel ({@code --gpu-attention})
 * follows the scalar CPU attention ({@code off}) on the same CUDA backend, per
 * architecture, on real prompts. The kernel reads an FP16 copy of the KV cache,
 * so the two can part ways after enough steps compound the rounding; this test
 * measures where, rather than inheriting the Llama-family answer for another
 * architecture.
 *
 * <p>For each prompt it prefills in one window, decodes {@value #STEPS} tokens by
 * argmax both ways, and prints the first step at which the two differ ("none" when
 * they never do). The hard check is the first generated token, which must agree on
 * every prompt: it is decided by the prefill alone, so a disagreement there is a
 * kernel fault, not accumulated rounding. The divergence steps are the
 * characterization the docs report.
 *
 * <p>Run: {@code mvn verify -pl juno-master -Pgpu}. Requires CUDA and the model
 * files under {@code models/}; skipped otherwise.
 */
@Tag("gpu")
@DisplayName("GPU attention kernel against scalar attention - greedy divergence per architecture (requires CUDA + models)")
class GpuAttentionDivergenceIT {

	private static final int STEPS = 64;

	private static final String[] PROMPTS = {
			"The capital of France is",
			"Write a short poem about the sea.",
			"Explain in two sentences why the sky is blue.",
			"List three prime numbers greater than ten and explain how you know they are prime.",
			"def fibonacci(n):",
			"Once upon a time, in a small village at the edge of a forest, there lived", };

	private static GpuContext gpu;
	private String saved;

	@BeforeAll
	static void init() {
		assumeTrue(Boolean.getBoolean("juno.gpu.test"),
				"Skipping GpuAttentionDivergenceIT - pass -Djuno.gpu.test=true to enable");
		assumeTrue(CudaAvailability.isAvailable(), "Skipping - no CUDA device");
		gpu = GpuContext.init(0);
	}

	@AfterAll
	static void destroy() {
		if (gpu != null)
			gpu.close();
	}

	@AfterEach
	void restore() {
		if (saved == null)
			System.clearProperty(GpuAttentionOptions.ENV_PROPERTY);
		else
			System.setProperty(GpuAttentionOptions.ENV_PROPERTY, saved);
	}

	@ParameterizedTest(name = "{0}")
	@ValueSource(strings = { "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", "Phi-3.5-mini-instruct-Q4_K_M.gguf",
			"Qwen3-1.7B-Q4_K_M.gguf" })
	@DisplayName("first token agrees on every prompt; divergence step recorded")
	void greedy_divergence(String file) throws Exception {
		Path model = model(file);
		assumeTrue(model.toFile().exists(), "Skipping - model not found: " + model);
		saved = System.getProperty(GpuAttentionOptions.ENV_PROPERTY);

		ShardContext shard;
		GgufTokenizer tokenizer;
		try (GgufReader r = GgufReader.open(model)) {
			LlamaConfig cfg = LlamaConfig.from(r);
			shard = new ShardContext("gpu-attn-div", 0, cfg.numLayers(), true, true, cfg.vocabSize(), cfg.hiddenDim(),
					cfg.numHeads());
			tokenizer = GgufTokenizer.load(r);
		}
		int[][] prompts = new int[PROMPTS.length][];
		for (int p = 0; p < PROMPTS.length; p++)
			prompts[p] = tokenizer.encode(PROMPTS[p]);

		int[][] off = decodeAll(model, shard, prompts, "off");
		int[][] on = decodeAll(model, shard, prompts, "on");

		int identical = 0;
		StringBuilder steps = new StringBuilder();
		for (int p = 0; p < prompts.length; p++) {
			int first = firstDifference(off[p], on[p]);
			if (first < 0)
				identical++;
			steps.append(first < 0 ? "none" : String.valueOf(first)).append(p + 1 < prompts.length ? ", " : "");
			assertThat(on[p][0]).as(file + " prompt " + p + ": first generated token, kernel on vs off")
					.isEqualTo(off[p][0]);
		}
		System.out.printf(Locale.ROOT, "GPU-ATTN-DIV %s steps=%d prompts=%d identical=%d first-divergence=[%s]%n", file,
				STEPS, prompts.length, identical, steps);
	}

	private static int[][] decodeAll(Path model, ShardContext shard, int[][] prompts, String mode) throws Exception {
		System.setProperty(GpuAttentionOptions.ENV_PROPERTY, mode);
		ForwardPassHandler h = ForwardPassHandlerLoader.load(model, shard, new CudaMatVec(gpu));
		try {
			assertThat(h.gpuAttentionActive()).as("--gpu-attention " + mode).isEqualTo("on".equals(mode));
			int[][] out = new int[prompts.length][];
			for (int p = 0; p < prompts.length; p++)
				out[p] = greedy(h, shard, prompts[p], "div-" + mode + "-" + p);
			return out;
		} finally {
			h.releaseGpuResources();
		}
	}

	private static int[] greedy(ForwardPassHandler h, ShardContext shard, int[] prompt, String id) {
		int[] out = new int[STEPS];
		float[] logits = h.forwardBatch(BatchForwardRequest.withTokens(id, prompt, 0), shard).lastLogits();
		int pos = prompt.length;
		int[] one = new int[1];
		for (int s = 0; s < STEPS; s++) {
			out[s] = argmax(logits);
			one[0] = out[s];
			if (s + 1 < STEPS)
				logits = h.forward(ForwardRequest.withTokens(id, one, pos++), shard).logits();
		}
		h.evict(id);
		return out;
	}

	private static int firstDifference(int[] a, int[] b) {
		for (int i = 0; i < a.length; i++)
			if (a[i] != b[i])
				return i;
		return -1;
	}

	private static int argmax(float[] a) {
		int best = 0;
		for (int i = 1; i < a.length; i++)
			if (a[i] > a[best])
				best = i;
		return best;
	}

	private static Path model(String file) {
		Path here = Path.of(System.getProperty("user.dir"));
		Path root = here.endsWith("juno-master") ? here.getParent() : here;
		return root.resolve("models").resolve(file);
	}
}
