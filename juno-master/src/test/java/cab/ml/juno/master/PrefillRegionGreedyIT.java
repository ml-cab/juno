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
import cab.ml.juno.node.PrefillRegionOptions;
import cab.ml.juno.node.GpuContext;
import cab.ml.juno.node.LlamaConfig;
import cab.ml.juno.node.ShardContext;
import cab.ml.juno.tokenizer.GgufTokenizer;

/**
 * The prefill-window device region against the host window path over 64 greedy
 * tokens, on six real prompts per model: TinyLlama and Mistral 7B, the two models
 * the region's greedy-output criterion names.
 *
 * <p>Each prompt is prefixed with one fixed instruction sentence so that every
 * prefill window is wider than the region's eight-row floor and actually runs on
 * the region. The region reproduces the host path's arithmetic, so the two must
 * agree on every generated token of every prompt, not merely on the first.
 *
 * <p>The comparison runs both paths in one build: with the region off, a window
 * takes exactly the host path the build before the region ran (the only change on
 * that path writes the KV mirror in one copy per tensor instead of one per row, with
 * the same bytes).
 */
@Tag("gpu")
@DisplayName("Prefill-window device region against the host window path - 64 greedy tokens (requires CUDA + models)")
class PrefillRegionGreedyIT {

	private static final int STEPS = 64;

	private static final String PREAMBLE = "You are a careful assistant. Answer the request below clearly and"
			+ " completely. ";

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
				"Skipping PrefillRegionGreedyIT - pass -Djuno.gpu.test=true to enable");
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
			System.clearProperty(PrefillRegionOptions.ENV_PROPERTY);
		else
			System.setProperty(PrefillRegionOptions.ENV_PROPERTY, saved);
	}

	@ParameterizedTest(name = "{0}")
	@ValueSource(strings = { "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", "mistral-7b-instruct-v0.1-q4_k_m.gguf" })
	@DisplayName("every generated token agrees on every prompt, region on against region off")
	void greedy_identical(String file) throws Exception {
		Path model = model(file);
		assumeTrue(model.toFile().exists(), "Skipping - model not found: " + model);
		saved = System.getProperty(PrefillRegionOptions.ENV_PROPERTY);

		ShardContext shard;
		GgufTokenizer tokenizer;
		try (GgufReader r = GgufReader.open(model)) {
			LlamaConfig cfg = LlamaConfig.from(r);
			shard = new ShardContext("prefill-region-greedy", 0, cfg.numLayers(), true, true, cfg.vocabSize(),
					cfg.hiddenDim(), cfg.numHeads());
			tokenizer = GgufTokenizer.load(r);
		}
		int[][] prompts = new int[PROMPTS.length][];
		for (int p = 0; p < PROMPTS.length; p++) {
			prompts[p] = tokenizer.encode(PREAMBLE + PROMPTS[p]);
			assertThat(prompts[p].length).as("prompt " + p + " must be a region-wide window").isGreaterThan(8);
		}

		int[][] off = decodeAll(model, shard, prompts, "off");
		int[][] on = decodeAll(model, shard, prompts, "on");

		int identical = 0;
		StringBuilder steps = new StringBuilder();
		for (int p = 0; p < prompts.length; p++) {
			int first = firstDifference(off[p], on[p]);
			if (first < 0)
				identical++;
			steps.append(first < 0 ? "none" : String.valueOf(first)).append(p + 1 < prompts.length ? ", " : "");
		}
		System.out.printf(Locale.ROOT, "PREFILL-REGION-GREEDY %s steps=%d prompts=%d identical=%d first-divergence=[%s]%n",
				file, STEPS, prompts.length, identical, steps);
		assertThat(identical).as(file + ": prompts whose 64 greedy tokens are identical, region on vs off")
				.isEqualTo(prompts.length);
	}

	private static int[][] decodeAll(Path model, ShardContext shard, int[][] prompts, String mode) throws Exception {
		System.setProperty(PrefillRegionOptions.ENV_PROPERTY, mode);
		ForwardPassHandler h = ForwardPassHandlerLoader.load(model, shard, new CudaMatVec(gpu));
		try {
			assertThat(h.prefillRegionActive()).as("prefill region " + mode).isEqualTo("on".equals(mode));
			int[][] out = new int[prompts.length][];
			for (int p = 0; p < prompts.length; p++)
				out[p] = greedy(h, shard, prompts[p], "greedy-" + mode + "-" + p);
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
