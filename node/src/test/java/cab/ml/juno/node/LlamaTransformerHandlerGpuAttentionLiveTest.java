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
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * End-to-end proof that {@code --gpu-attention on} actually activates the
 * GPU-resident attention path on a real GGUF-loaded, CUDA-backed handler —
 * not just in isolation ({@link DeviceKvCacheLifecycleTest}) or via
 * {@link LlamaTransformerHandler#newTestInstance} (which never constructs a
 * GPU backend at all, so it can't exercise this wiring).
 *
 * <p>Motivation: a manual CLI smoke test (interactive console mode disables
 * {@code java.util.logging} for a clean UX) could not visually confirm
 * activation, and JFR attention/MatVec counts alone don't distinguish the
 * Stage-1 stub path from the plain CPU path (both call the same
 * {@link GqaMath#attend} oracle). This test checks the real signal directly:
 * {@link LlamaTransformerHandler#gpuAttentionActive()} plus a live device-byte
 * delta during a real forward pass, then confirms {@code evict()} returns
 * device memory to baseline.
 */
@Tag("gpu")
@DisplayName("LlamaTransformerHandler — --gpu-attention live activation on a real model")
class LlamaTransformerHandlerGpuAttentionLiveTest {

	private static final Path MODEL = Path.of(System.getProperty("user.dir")).endsWith("node")
			? Path.of(System.getProperty("user.dir")).getParent().resolve("models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
			: Path.of("models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");

	private static GpuContext ctx;
	private String savedProperty;

	private static boolean modelPresent() {
		return MODEL.toFile().exists();
	}

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping — no CUDA device");
		ctx = GpuContext.init(0);
	}

	@AfterAll
	static void destroy() {
		if (ctx != null)
			ctx.close();
	}

	@AfterEach
	void restoreProperty() {
		if (savedProperty == null)
			System.clearProperty(GpuAttentionOptions.ENV_PROPERTY);
		else
			System.setProperty(GpuAttentionOptions.ENV_PROPERTY, savedProperty);
	}

	@Test
	@EnabledIf("modelPresent")
	@DisplayName("gpuAttentionActive() is auto (on) by default on CUDA, false with --gpu-attention off, true with --gpu-attention on, and frees device memory on evict")
	void gpu_attention_activates_and_frees_on_evict() throws Exception {
		savedProperty = System.getProperty(GpuAttentionOptions.ENV_PROPERTY);

		// Default (auto): on CUDA with a supported architecture, the GPU path activates.
		System.clearProperty(GpuAttentionOptions.ENV_PROPERTY);
		ShardContext shardAuto = shardContext();
		LlamaTransformerHandler auto = LlamaTransformerHandler.load(MODEL, shardAuto, new CudaMatVec(ctx));
		assertThat(auto.gpuAttentionActive()).as("default (auto) must activate on CUDA for a supported architecture")
				.isTrue();

		// Explicit off: no device KV mirror should ever be allocated.
		System.setProperty(GpuAttentionOptions.ENV_PROPERTY, "off");
		ShardContext shardOff = shardContext();
		LlamaTransformerHandler off = LlamaTransformerHandler.load(MODEL, shardOff, new CudaMatVec(ctx));
		assertThat(off.gpuAttentionActive()).as("--gpu-attention off must disable the GPU path").isFalse();

		long baselineBytes = DeviceKvCache.allocatedBytes();
		int[] prompt = { 1, 2, 3, 4, 5, 6, 7, 8 };
		off.forwardBatch(BatchForwardRequest.withTokens("live-off", prompt, 0), shardOff);
		assertThat(DeviceKvCache.allocatedBytes())
				.as("no device KV mirror should be allocated when the flag is off")
				.isEqualTo(baselineBytes);
		off.evict("live-off");

		// On: gpuAttentionActive() must flip true, and a real forward pass must
		// allocate device KV bytes, then free them fully on evict().
		System.setProperty(GpuAttentionOptions.ENV_PROPERTY, "on");
		ShardContext shardOn = shardContext();
		LlamaTransformerHandler on = LlamaTransformerHandler.load(MODEL, shardOn, new CudaMatVec(ctx));
		assertThat(on.gpuAttentionActive()).as("--gpu-attention on must activate the GPU path on CUDA").isTrue();

		long beforeForward = DeviceKvCache.allocatedBytes();
		float[] onLogits = on.forwardBatch(BatchForwardRequest.withTokens("live-on", prompt, 0), shardOn)
				.lastLogits();
		assertThat(DeviceKvCache.allocatedBytes())
				.as("a real prefill must allocate device KV bytes when the flag is on")
				.isGreaterThan(beforeForward);

		on.evict("live-on");
		assertThat(DeviceKvCache.allocatedBytes())
				.as("evict() must free every device KV byte allocated for this request")
				.isEqualTo(beforeForward);

		// Stage-1 stub is a correctness-preserving round trip through FP16 device
		// storage — output should closely track the CPU-only run (loose tolerance
		// covers FP16 K/V rounding, not an algorithmic difference).
		System.setProperty(GpuAttentionOptions.ENV_PROPERTY, "off");
		ShardContext shardRef = shardContext();
		LlamaTransformerHandler ref = LlamaTransformerHandler.load(MODEL, shardRef, new CudaMatVec(ctx));
		float[] refLogits = ref.forwardBatch(BatchForwardRequest.withTokens("live-ref", prompt, 0), shardRef)
				.lastLogits();
		ref.evict("live-ref");

		int onArgmax = argmax(onLogits);
		int refArgmax = argmax(refLogits);
		assertThat(onArgmax).as("greedy top token must match between GPU-attention on/off").isEqualTo(refArgmax);
	}

	private static ShardContext shardContext() throws Exception {
		try (GgufReader r = GgufReader.open(MODEL)) {
			LlamaConfig cfg = LlamaConfig.from(r);
			return new ShardContext("n0", 0, cfg.numLayers(), true, true, cfg.vocabSize(),
					cfg.hiddenDim(), cfg.numHeads());
		}
	}

	private static int argmax(float[] logits) {
		int best = 0;
		float bestScore = Float.NEGATIVE_INFINITY;
		for (int i = 0; i < logits.length; i++) {
			if (logits[i] > bestScore) {
				bestScore = logits[i];
				best = i;
			}
		}
		return best;
	}
}
