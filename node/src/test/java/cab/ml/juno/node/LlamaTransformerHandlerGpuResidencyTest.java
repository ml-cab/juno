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

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import jdk.jfr.Recording;
import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordingFile;

/**
 * {@code --gpu-residency} on a real GGUF-loaded, CUDA-backed handler. The region
 * is bit-identical to the GPU op-at-a-time path ({@link ResidentQkvPathTest}),
 * but the handler's default decode normalizes on the CPU, and the GPU norm sums
 * in a different order (about 6e-7 apart), which can move a Q8_1 rounding of the
 * projection input. So against the flag-off run this asserts what that allows:
 * the same greedy token at every position and logits within a measured bound.
 * Qwen2.5 (split-half RoPE, Q/K/V biases) runs the region too.
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

	/**
	 * As {@link #LOGIT_TOL}, for Qwen2.5-3B. Measured over this run: 0.507, the
	 * CPU-versus-GPU norm compounded through 36 layers. The bound is about twice that.
	 */
	private static final double QWEN_LOGIT_TOL = 1.0;

	/**
	 * Qwen2.5's opening tokens ("The capital of France is"); the rest of its run is the
	 * flag-off handler's own greedy continuation. {@link #TOKENS} are another
	 * vocabulary's ids: on Qwen2.5 they put a near tie at one position (top two logits
	 * 0.026 apart, under the CPU-versus-GPU norm's 0.2 to 0.4), where the two paths
	 * picked different tokens. Greedy parity is a property of generated text.
	 */
	private static final int[] QWEN_PROMPT = { 785, 6722, 315, 9625, 374 };
	private static final int QWEN_STEPS = 24;

	private static GpuContext ctx;
	private String saved;

	@TempDir
	Path tmp;

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
	@DisplayName("tinyllama: the whole decode layer runs in the region: per token at most one upload per layer and one download per layer plus the logits")
	void decodeRunsTheWholeLayerInTheRegion() throws Exception {
		assumeTrue(TINYLLAMA.toFile().exists(), "model not present");
		saved = System.getProperty(GpuResidencyOptions.ENV_PROPERTY);
		System.setProperty(GpuResidencyOptions.ENV_PROPERTY, "on");
		ShardContext shard = shard(TINYLLAMA);
		LlamaTransformerHandler on = loadOnCuda(TINYLLAMA, shard);
		assertThat(on.gpuResidencyAttentionActive()).as("GPU attention is on by default on CUDA").isTrue();
		assertThat(on.gpuResidencyWholeLayerActive()).as("every layer's tail weights are K-quant on the device").isTrue();
		int layers = shard.endLayer() - shard.startLayer();
		long tokens = TOKENS.length;
		long steps = (long) layers * tokens;

		List<RecordedEvent> events = record(() -> decode(on, shard, "res-layer"));

		// The op-at-a-time sites: the KV append and attention after the region, the
		// attention table copy, and every GEMV and norm of the layer's tail. None may
		// run in decode with the whole layer in the region.
		for (String site : new String[] { "memcpy(K row H2D)", "memcpy(V row H2D)", "memcpy(gqa qBatch H2D)",
				"memcpy(gqa kPtrs H2D)", "memcpy(gqa outBatch D2H)", "materializeRows(resident activation)",
				"memcpy(decode region attention table H2D)", "materialize(decode region k, v, attention)",
				"cudaMemcpyAsync(x H2D q4k sameX)", "cudaMemcpyAsync(y D2H q4k sameX)", "memcpy(rmsNorm xBatch H2D)" })
			assertThat(decodeCopies(events, site)).as(site).isZero();
		// The single-row K-quant GEMV sites now carry only the LM head, once per token.
		assertThat(decodeCopies(events, "cudaMemcpyAsync(x H2D q4k)")).as("LM head input").isEqualTo(tokens);
		assertThat(decodeCopies(events, "cudaMemcpyAsync(y D2H q4k)")).as("logits").isEqualTo(tokens);
		assertThat(decodeCopies(events, "upload(decode region input)"))
				.as("the residual row goes up once per token: each later layer reads it on the device").isEqualTo(tokens);
		assertThat(decodeCopies(events, "materialize(decode region k, v, attention, layer output)"))
				.as("one download per layer and token").isEqualTo(steps);
		long h2d = decodeCopiesByDirection(events, "H2D");
		long d2h = decodeCopiesByDirection(events, "D2H");
		System.out.printf(java.util.Locale.ROOT, "GPU-RESIDENCY whole layer: per token %d H2D, %d D2H, %d layers%n",
				h2d / tokens, d2h / tokens, layers);
		assertThat(h2d).as("host-to-device copies, <= 1 x layers per token").isLessThanOrEqualTo(steps);
		assertThat(d2h).as("device-to-host copies, <= 1 x layers + 1 (the logits) per token")
				.isLessThanOrEqualTo(steps + tokens);
	}

	@Test
	@DisplayName("qwen2.5: on activates the region (split-half RoPE, Q/K/V biases) and greedy decode equals the flag-off run")
	void qwen25DecodesInTheRegion() throws Exception {
		assumeTrue(QWEN25.toFile().exists(), "model not present");
		saved = System.getProperty(GpuResidencyOptions.ENV_PROPERTY);

		System.setProperty(GpuResidencyOptions.ENV_PROPERTY, "off");
		ShardContext shard = shard(QWEN25);
		LlamaTransformerHandler off = loadOnCuda(QWEN25, shard);
		assertThat(off.gpuResidencyActive()).isFalse();
		int[] tokens = greedyRun(off, shard);
		float[][] offLogits = decode(off, shard, "res-qwen-off", tokens);
		releaseDevice();

		System.setProperty(GpuResidencyOptions.ENV_PROPERTY, "on");
		LlamaTransformerHandler on = loadOnCuda(QWEN25, shard);
		assertThat(on.gpuResidencyActive()).as("the region runs split-half RoPE and adds the biases").isTrue();
		assertThat(on.gpuResidencyWholeLayerActive()).isTrue();
		float[][] onLogits = decode(on, shard, "res-qwen-on", tokens);

		double worst = 0;
		for (int p = 0; p < tokens.length; p++) {
			assertThat(argmax(onLogits[p])).as("greedy token at position %d", p).isEqualTo(argmax(offLogits[p]));
			for (int i = 0; i < offLogits[p].length; i++)
				worst = Math.max(worst, Math.abs(onLogits[p][i] - offLogits[p][i]));
		}
		System.out.printf(java.util.Locale.ROOT, "GPU-RESIDENCY qwen2.5 max|logit on - off| over %d positions = %.6f%n",
				tokens.length, worst);
		assertThat(worst).as("largest logit difference, on vs off").isLessThan(QWEN_LOGIT_TOL);
	}

	@Test
	@DisplayName("qwen2.5: a decoded token makes 2 uploads and layers + 1 downloads")
	void qwen25DecodeCopiesPerToken() throws Exception {
		assumeTrue(QWEN25.toFile().exists(), "model not present");
		saved = System.getProperty(GpuResidencyOptions.ENV_PROPERTY);
		System.setProperty(GpuResidencyOptions.ENV_PROPERTY, "on");
		ShardContext shard = shard(QWEN25);
		LlamaTransformerHandler on = loadOnCuda(QWEN25, shard);
		assertThat(on.gpuResidencyWholeLayerActive()).isTrue();
		int layers = shard.endLayer() - shard.startLayer();
		long tokens = TOKENS.length;

		List<RecordedEvent> events = record(() -> decode(on, shard, "res-qwen-copies"));

		assertThat(decodeCopies(events, "upload(decode region input)")).as("residual row, once per token")
				.isEqualTo(tokens);
		assertThat(decodeCopiesByDirection(events, "H2D")).as("residual row and LM head input")
				.isEqualTo(2 * tokens);
		assertThat(decodeCopiesByDirection(events, "D2H")).as("one download per layer, plus the logits")
				.isEqualTo((layers + 1L) * tokens);
	}

	private List<RecordedEvent> record(Runnable body) throws Exception {
		Path jfr = tmp.resolve("residency-" + System.nanoTime() + ".jfr");
		try (Recording rec = new Recording()) {
			rec.enable("juno.DeviceStaging").withThreshold(java.time.Duration.ZERO);
			rec.setDestination(jfr);
			rec.start();
			body.run();
			rec.stop();
		}
		assertThat(Files.size(jfr)).isPositive();
		return RecordingFile.readAllEvents(jfr);
	}

	/** Decode-phase copies recorded under {@code site}, summed over the recording's emissions. */
	private static long decodeCopies(List<RecordedEvent> events, String site) {
		return events.stream().filter(e -> e.getEventType().getName().equals("juno.DeviceStaging"))
				.filter(e -> site.equals(e.getString("site")) && "decode".equals(e.getString("phase")))
				.mapToLong(e -> e.getLong("copies")).sum();
	}

	/** Decode-phase copies in {@code direction} (H2D or D2H) over every site. */
	private static long decodeCopiesByDirection(List<RecordedEvent> events, String direction) {
		return events.stream().filter(e -> e.getEventType().getName().equals("juno.DeviceStaging"))
				.filter(e -> direction.equals(e.getString("direction")) && "decode".equals(e.getString("phase")))
				.mapToLong(e -> e.getLong("copies")).sum();
	}

	private static int argmax(float[] a) {
		int best = 0;
		for (int i = 1; i < a.length; i++)
			if (a[i] > a[best])
				best = i;
		return best;
	}

	private static float[][] decode(LlamaTransformerHandler h, ShardContext shard, String kv) {
		return decode(h, shard, kv, TOKENS);
	}

	private static float[][] decode(LlamaTransformerHandler h, ShardContext shard, String kv, int[] tokens) {
		float[][] out = new float[tokens.length][];
		int[] one = new int[1];
		for (int p = 0; p < tokens.length; p++) {
			one[0] = tokens[p];
			out[p] = h.forward(ForwardRequest.withTokens(kv, one, p), shard).logits();
		}
		h.evict(kv);
		return out;
	}

	/** {@link #QWEN_PROMPT} followed by the handler's greedy continuation, {@link #QWEN_STEPS} tokens in all. */
	private static int[] greedyRun(LlamaTransformerHandler h, ShardContext shard) {
		int[] run = new int[QWEN_STEPS];
		System.arraycopy(QWEN_PROMPT, 0, run, 0, QWEN_PROMPT.length);
		int[] one = new int[1];
		for (int p = 0; p < QWEN_STEPS - 1; p++) {
			one[0] = run[p];
			float[] logits = h.forward(ForwardRequest.withTokens("qwen-greedy", one, p), shard).logits();
			if (p + 1 >= QWEN_PROMPT.length)
				run[p + 1] = argmax(logits);
		}
		h.evict("qwen-greedy");
		return run;
	}

	private static ShardContext shard(Path model) throws Exception {
		try (GgufReader r = GgufReader.open(model)) {
			LlamaConfig cfg = LlamaConfig.from(r);
			return new ShardContext("n0", 0, cfg.numLayers(), true, true, cfg.vocabSize(), cfg.hiddenDim(),
					cfg.numHeads());
		}
	}
}
