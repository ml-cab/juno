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

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * The decode batch device region on real weights: a batched decode of independent
 * requests against each request decoded alone, greedy, and against the host-driven
 * batch it replaces. A reply must not depend on what is batched with it.
 */
@Tag("gpu")
@DisplayName("Decode batch device region - real models")
class BatchDecodeRegionHandlerTest {

	private static final int STEPS = 64;
	/** Largest logit difference from the host-driven batch: the norm runs on another path there (see LlamaTransformerHandlerGpuResidencyTest). */
	private static final double HOST_BATCH_TOL = 1.0;

	private static GpuContext gpu;
	private String saved;
	private final List<CudaMatVec> backends = new ArrayList<>();
	private final List<ForwardPassHandler> handlers = new ArrayList<>();

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping - no CUDA device");
		gpu = GpuContext.init(0);
	}

	@AfterAll
	static void destroy() {
		if (gpu != null)
			gpu.close();
	}

	@AfterEach
	void release() {
		handlers.forEach(ForwardPassHandler::releaseGpuResources);
		handlers.clear();
		backends.forEach(CudaMatVec::releaseScratch);
		backends.clear();
		if (saved == null)
			System.clearProperty(BatchDecodeRegionOptions.ENV_PROPERTY);
		else
			System.setProperty(BatchDecodeRegionOptions.ENV_PROPERTY, saved);
	}

	private ForwardPassHandler load(Path model, ShardContext shard) throws Exception {
		CudaMatVec backend = new CudaMatVec(gpu);
		backends.add(backend);
		ForwardPassHandler h = ForwardPassHandlerLoader.load(model, shard, backend);
		handlers.add(h);
		return h;
	}

	@ParameterizedTest(name = "{0}, batch of {1}")
	@CsvSource({ "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf,8", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf,2",
			"qwen2.5-3b-instruct-q4_k_m.gguf,8", "Phi-3.5-mini-instruct-Q4_K_M.gguf,8", "Phi-3.5-mini-instruct-Q4_K_M.gguf,2",
			"mistral-7b-instruct-v0.1-q4_k_m.gguf,8", "Qwen3-1.7B-Q4_K_M.gguf,8" })
	@DisplayName("each request's greedy reply, and every logit of every step, equals decoding it alone")
	void batchEqualsSingleBitForBit(String file, int n) throws Exception {
		Path model = model(file);
		assumeTrue(model.toFile().exists(), "Skipping - model not found: " + model);
		saved = System.getProperty(BatchDecodeRegionOptions.ENV_PROPERTY);
		System.clearProperty(BatchDecodeRegionOptions.ENV_PROPERTY);
		ShardContext shard = wholeModel(model);
		ForwardPassHandler h = load(model, shard);
		assertThat(h.batchDecodeRegionActive()).as("the batch region runs by default where the single region does").isTrue();

		Run single = decodeAlone(h, shard, n, "s");
		Run batched = decodeBatched(h, shard, n, "b");

		for (int b = 0; b < n; b++) {
			assertThat(batched.tokens[b]).as("%s batch of %d: request %d reply", file, n, b).containsExactly(single.tokens[b]);
			for (int s = 0; s < STEPS; s++)
				assertBits(batched.logits[b][s], single.logits[b][s], file + " batch of " + n + " request " + b + " step " + s);
		}
	}

	@ParameterizedTest(name = "{0}")
	@ValueSource(strings = { "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", "qwen2.5-3b-instruct-q4_k_m.gguf",
			"Phi-3.5-mini-instruct-Q4_K_M.gguf", "Qwen3-1.7B-Q4_K_M.gguf" })
	@DisplayName("one batched step, requests at different positions, stays within the host-driven batch's tolerance")
	void regionMatchesTheHostDrivenBatch(String file) throws Exception {
		Path model = model(file);
		assumeTrue(model.toFile().exists(), "Skipping - model not found: " + model);
		saved = System.getProperty(BatchDecodeRegionOptions.ENV_PROPERTY);
		ShardContext shard = wholeModel(model);
		for (int n : new int[] { 1, 2, 8 }) {
			System.setProperty(BatchDecodeRegionOptions.ENV_PROPERTY, "off");
			ForwardPassHandler off = load(model, shard);
			assertThat(off.batchDecodeRegionActive()).as("the switch turns it off").isFalse();
			float[][] offLogits = oneStep(off, shard, n);
			off.releaseGpuResources();
			handlers.remove(off);

			System.clearProperty(BatchDecodeRegionOptions.ENV_PROPERTY);
			ForwardPassHandler on = load(model, shard);
			assertThat(on.batchDecodeRegionActive()).isTrue();
			float[][] onLogits = oneStep(on, shard, n);
			on.releaseGpuResources();
			handlers.remove(on);

			for (int b = 0; b < n; b++)
				for (int v = 0; v < onLogits[b].length; v++)
					assertThat((double) Math.abs(onLogits[b][v] - offLogits[b][v])).as("%s n=%d request %d vocab %d", file, n, b, v)
							.isLessThanOrEqualTo(HOST_BATCH_TOL);
		}
	}

	private record Run(int[][] tokens, float[][][] logits) {
	}

	/** One batched step after a prefill of each request's own length: {@code n} requests at different positions. */
	private static float[][] oneStep(ForwardPassHandler h, ShardContext shard, int n) {
		List<String> ids = new ArrayList<>();
		int[] next = new int[n];
		int[] pos = new int[n];
		for (int b = 0; b < n; b++) {
			String id = "o" + b;
			ids.add(id);
			int[] prompt = prompt(b, shard.vocabSize());
			float[] last = h.forwardBatch(BatchForwardRequest.withTokens(id, prompt, 0), shard).lastLogits();
			next[b] = argmax(last);
			pos[b] = prompt.length;
		}
		float[][] logits = h.forwardMultiDecode(MultiDecodeForwardRequest.withTokens(ids, next, pos), shard).logits();
		ids.forEach(h::evict);
		return logits;
	}

	private static Run decodeAlone(ForwardPassHandler h, ShardContext shard, int n, String idPrefix) {
		int[][] tokens = new int[n][STEPS];
		float[][][] logits = new float[n][STEPS][];
		for (int b = 0; b < n; b++) {
			String id = idPrefix + b;
			int[] prompt = prompt(b, shard.vocabSize());
			float[] last = h.forwardBatch(BatchForwardRequest.withTokens(id, prompt, 0), shard).lastLogits();
			int tok = argmax(last);
			int pos = prompt.length;
			for (int s = 0; s < STEPS; s++) {
				tokens[b][s] = tok;
				float[] l = h.forward(ForwardRequest.withTokens(id, new int[] { tok }, pos), shard).logits();
				logits[b][s] = l;
				tok = argmax(l);
				pos++;
			}
			h.evict(id);
		}
		return new Run(tokens, logits);
	}

	private static Run decodeBatched(ForwardPassHandler h, ShardContext shard, int n, String idPrefix) {
		List<String> ids = new ArrayList<>();
		int[] next = new int[n];
		int[] pos = new int[n];
		for (int b = 0; b < n; b++) {
			String id = idPrefix + b;
			ids.add(id);
			int[] prompt = prompt(b, shard.vocabSize());
			float[] last = h.forwardBatch(BatchForwardRequest.withTokens(id, prompt, 0), shard).lastLogits();
			next[b] = argmax(last);
			pos[b] = prompt.length;
		}
		int[][] tokens = new int[n][STEPS];
		float[][][] logits = new float[n][STEPS][];
		for (int s = 0; s < STEPS; s++) {
			float[][] step = h.forwardMultiDecode(MultiDecodeForwardRequest.withTokens(ids, next.clone(), pos.clone()), shard)
					.logits();
			for (int b = 0; b < n; b++) {
				tokens[b][s] = next[b];
				logits[b][s] = step[b];
				next[b] = argmax(step[b]);
				pos[b]++;
			}
		}
		ids.forEach(h::evict);
		return new Run(tokens, logits);
	}

	/** Request {@code b}'s prompt: ordinary-vocabulary token ids, each request a different length. */
	private static int[] prompt(int b, int vocab) {
		int[] t = new int[3 + 2 * b];
		for (int i = 0; i < t.length; i++)
			t[i] = 1000 + (int) ((300L * (b + 1) + 7919L * i) % (vocab / 2));
		return t;
	}

	private static void assertBits(float[] actual, float[] expected, String what) {
		for (int i = 0; i < expected.length; i++)
			if (Float.floatToRawIntBits(actual[i]) != Float.floatToRawIntBits(expected[i]))
				throw new AssertionError(what + ": logit " + i + " is " + actual[i] + ", decoding alone gives " + expected[i]);
	}

	private static int argmax(float[] a) {
		int best = 0;
		for (int i = 1; i < a.length; i++)
			if (a[i] > a[best])
				best = i;
		return best;
	}

	private static ShardContext wholeModel(Path model) throws Exception {
		try (GgufReader r = GgufReader.open(model)) {
			LlamaConfig cfg = LlamaConfig.from(r);
			return new ShardContext("batch-decode-region", 0, cfg.numLayers(), true, true, cfg.vocabSize(), cfg.hiddenDim(),
					cfg.numHeads());
		}
	}

	private static Path model(String file) {
		Path here = Path.of(System.getProperty("user.dir"));
		Path root = here.endsWith("node") ? here.getParent() : here;
		return root.resolve("models").resolve(file);
	}
}
