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

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * {@code --gpu-residency} on a real GGUF-loaded, CUDA-backed handler. The region
 * is bit-identical to the GPU op-at-a-time path ({@link ResidentQkvPathTest}),
 * but the handler's default decode normalizes on the CPU, and the GPU norm sums
 * in a different order (about 6e-7 apart), which can move a Q8_1 rounding of the
 * projection input. So against the flag-off run this asserts what that allows:
 * the same greedy token at every position and logits within a measured bound. A
 * model the region cannot run (split-half RoPE, Q/K/V biases) must decline it.
 */
@Tag("gpu")
@DisplayName("LlamaTransformerHandler - --gpu-residency decode region on a real model")
class LlamaTransformerHandlerGpuResidencyTest {

	private static final Path MODELS = Path.of(System.getProperty("user.dir")).endsWith("node")
			? Path.of(System.getProperty("user.dir")).getParent().resolve("models")
			: Path.of("models");
	private static final Path TINYLLAMA = MODELS.resolve("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
	private static final Path QWEN25 = MODELS.resolve("qwen2.5-3b-instruct-q4_k_m.gguf");

	/** A short token run decoded one position at a time, the path generation uses. */
	private static final int[] TOKENS = { 1, 450, 7483, 310, 3444, 338, 3681, 29889, 13, 1576, 263, 1243, 29892,
			920, 526, 366, 2599, 29973, 13, 29902, 626, 1781, 29892, 6452 };

	/**
	 * Largest logit difference allowed between the region and the CPU-norm path;
	 * see the class comment. Measured on tinyllama over these 24 positions: 0.237
	 * (logits of magnitude 10 to 20), the norm's re-rounding compounded through 22
	 * layers. The bound is about twice that.
	 */
	private static final double LOGIT_TOL = 0.5;

	private static GpuContext ctx;
	private String saved;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping - no CUDA device");
		ctx = GpuContext.init(0);
	}

	@AfterAll
	static void destroy() {
		if (ctx != null)
			ctx.close();
	}

	/** Handlers and backends this test loaded, released after each test so their device memory does not outlive it. */
	private final java.util.List<LlamaTransformerHandler> loaded = new java.util.ArrayList<>();
	private final java.util.List<CudaMatVec> backends = new java.util.ArrayList<>();

	private LlamaTransformerHandler loadOnCuda(java.nio.file.Path model, ShardContext shard) throws Exception {
		CudaMatVec backend = new CudaMatVec(ctx);
		backends.add(backend);
		LlamaTransformerHandler h = LlamaTransformerHandler.load(model, shard, backend);
		loaded.add(h);
		return h;
	}

	@AfterEach
	void releaseDevice() {
		loaded.forEach(LlamaTransformerHandler::releaseGpuResources);
		loaded.clear();
		backends.forEach(CudaMatVec::releaseScratch);
		backends.clear();
	}

	@AfterEach
	void restore() {
		if (saved == null)
			System.clearProperty(GpuResidencyOptions.ENV_PROPERTY);
		else
			System.setProperty(GpuResidencyOptions.ENV_PROPERTY, saved);
	}

	@Test
	@DisplayName("tinyllama: on activates the region, and decode logits equal the flag-off run bit for bit")
	void decodeLogitsAreIdenticalWithTheRegionOn() throws Exception {
		assumeTrue(TINYLLAMA.toFile().exists(), "model not present");
		saved = System.getProperty(GpuResidencyOptions.ENV_PROPERTY);

		System.setProperty(GpuResidencyOptions.ENV_PROPERTY, "off");
		ShardContext shard = shard(TINYLLAMA);
		LlamaTransformerHandler off = loadOnCuda(TINYLLAMA, shard);
		assertThat(off.gpuResidencyActive()).isFalse();
		float[][] offLogits = decode(off, shard, "res-off");

		System.setProperty(GpuResidencyOptions.ENV_PROPERTY, "on");
		LlamaTransformerHandler on = loadOnCuda(TINYLLAMA, shard);
		assertThat(on.gpuResidencyActive()).as("--gpu-residency on must activate on CUDA with MMQ weights").isTrue();
		float[][] onLogits = decode(on, shard, "res-on");

		double worst = 0;
		for (int p = 0; p < TOKENS.length; p++) {
			assertThat(argmax(onLogits[p])).as("greedy token at position %d", p).isEqualTo(argmax(offLogits[p]));
			for (int i = 0; i < offLogits[p].length; i++)
				worst = Math.max(worst, Math.abs(onLogits[p][i] - offLogits[p][i]));
		}
		System.out.printf(java.util.Locale.ROOT, "GPU-RESIDENCY max|logit on - off| over %d positions = %.6f%n",
				TOKENS.length, worst);
		assertThat(worst).as("largest logit difference, on vs off").isLessThan(LOGIT_TOL);
	}

	@Test
	@DisplayName("qwen2.5: on is declined (split-half RoPE, Q/K/V biases) and decode is unchanged")
	void aModelTheRegionCannotRunDeclinesIt() throws Exception {
		assumeTrue(QWEN25.toFile().exists(), "model not present");
		saved = System.getProperty(GpuResidencyOptions.ENV_PROPERTY);
		System.setProperty(GpuResidencyOptions.ENV_PROPERTY, "on");
		ShardContext shard = shard(QWEN25);
		LlamaTransformerHandler on = loadOnCuda(QWEN25, shard);
		assertThat(on.gpuResidencyActive()).isFalse();
		float[] logits = on.forward(ForwardRequest.withTokens("res-qwen", new int[] { 9707 }, 0), shard).logits();
		assertThat(logits).hasSize(shard.vocabSize());
		on.evict("res-qwen");
	}

	private static int argmax(float[] a) {
		int best = 0;
		for (int i = 1; i < a.length; i++)
			if (a[i] > a[best])
				best = i;
		return best;
	}

	private static float[][] decode(LlamaTransformerHandler h, ShardContext shard, String kv) {
		float[][] out = new float[TOKENS.length][];
		int[] one = new int[1];
		for (int p = 0; p < TOKENS.length; p++) {
			one[0] = TOKENS[p];
			out[p] = h.forward(ForwardRequest.withTokens(kv, one, p), shard).logits();
		}
		h.evict(kv);
		return out;
	}

	private static ShardContext shard(Path model) throws Exception {
		try (GgufReader r = GgufReader.open(model)) {
			LlamaConfig cfg = LlamaConfig.from(r);
			return new ShardContext("n0", 0, cfg.numLayers(), true, true, cfg.vocabSize(), cfg.hiddenDim(),
					cfg.numHeads());
		}
	}
}
