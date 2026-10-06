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
 * {@code --gpu-residency} on a real Qwen3-1.7B handler. The region is
 * bit-identical to the Qwen3 GPU op-at-a-time path ({@link ResidentQkvPathQwen3Test}),
 * but the handler's default decode normalizes on the CPU (the residual row, and each
 * head of q and k), and the GPU norm sums in a different order, which can move a Q8_1
 * rounding of a projection input. So against
 * the flag-off run this asserts what that allows: the same greedy token at every
 * position and logits within a measured bound. And it holds the copy counts the
 * region exists to bring down.
 */
@Tag("gpu")
@DisplayName("Qwen3TransformerHandler - --gpu-residency decode region on a real model")
class Qwen3TransformerHandlerGpuResidencyTest {

	private static final Path MODELS = Path.of(System.getProperty("user.dir")).endsWith("node")
			? Path.of(System.getProperty("user.dir")).getParent().resolve("models")
			: Path.of("models");
	private static final Path QWEN3 = MODELS.resolve("Qwen3-1.7B-Q4_K_M.gguf");

	/**
	 * The opening tokens; the rest of the run is the flag-off handler's own greedy
	 * continuation, so the text is what this model generates rather than ids from
	 * another vocabulary. A fixed run of another model's ids put a near-tie at one
	 * position (top two logits 0.008 apart) where the two paths' rounding picked
	 * different tokens; greedy parity is a property of generated text.
	 */
	private static final int[] PROMPT = { 785, 6722, 315, 9625, 374 };
	private static final int STEPS = 24;

	/**
	 * Largest logit difference allowed between the region and the CPU-norm path;
	 * see the class comment. Measured on Qwen3-1.7B over this run: 0.644, from three
	 * CPU-versus-GPU norms per layer (the residual row, each head of q and of k)
	 * compounded through 28 layers, on logits of magnitude 15 to 32 (about 1.5 to 2
	 * times TinyLlama's, where the LLaMA-family test measured 0.237). The bound is
	 * about twice the measurement, the convention that test uses.
	 */
	private static final double LOGIT_TOL = 1.3;

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

	private final java.util.List<Qwen3TransformerHandler> loaded = new java.util.ArrayList<>();
	private final java.util.List<CudaMatVec> backends = new java.util.ArrayList<>();

	private Qwen3TransformerHandler loadOnCuda(ShardContext shard) throws Exception {
		CudaMatVec backend = new CudaMatVec(ctx);
		backends.add(backend);
		Qwen3TransformerHandler h = Qwen3TransformerHandler.load(QWEN3, shard, backend);
		loaded.add(h);
		return h;
	}

	@AfterEach
	void releaseDevice() {
		loaded.forEach(Qwen3TransformerHandler::releaseGpuResources);
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
	@DisplayName("Qwen3-1.7B: on activates the region; greedy tokens equal the flag-off run, logits within the bound")
	void decodeMatchesTheFlagOffRun() throws Exception {
		assumeTrue(QWEN3.toFile().exists(), "model not present");
		saved = System.getProperty(GpuResidencyOptions.ENV_PROPERTY);
		ShardContext shard = shard();

		System.setProperty(GpuResidencyOptions.ENV_PROPERTY, "off");
		Qwen3TransformerHandler off = loadOnCuda(shard);
		assertThat(off.gpuResidencyActive()).isFalse();
		int[] tokens = greedyRun(off, shard);
		float[][] offLogits = decode(off, shard, "qwen3-res-off", tokens);
		off.releaseGpuResources();

		System.setProperty(GpuResidencyOptions.ENV_PROPERTY, "on");
		Qwen3TransformerHandler on = loadOnCuda(shard);
		assertThat(on.gpuResidencyActive()).as("--gpu-residency on must activate on CUDA with K-quant weights").isTrue();
		float[][] onLogits = decode(on, shard, "qwen3-res-on", tokens);

		double worst = 0;
		for (int p = 0; p < tokens.length; p++) {
			assertThat(argmax(onLogits[p])).as("greedy token at position %d", p).isEqualTo(argmax(offLogits[p]));
			for (int i = 0; i < offLogits[p].length; i++)
				worst = Math.max(worst, Math.abs(onLogits[p][i] - offLogits[p][i]));
		}
		System.out.printf(java.util.Locale.ROOT, "QWEN3 GPU-RESIDENCY max|logit on - off| over %d positions = %.6f%n",
				tokens.length, worst);
		assertThat(worst).as("largest logit difference, on vs off").isLessThan(LOGIT_TOL);
	}

	@Test
	@DisplayName("Qwen3-1.7B: the whole decode layer runs in the region: per token at most one upload per layer and one download per layer plus the logits")
	void decodeRunsTheWholeLayerInTheRegion() throws Exception {
		assumeTrue(QWEN3.toFile().exists(), "model not present");
		saved = System.getProperty(GpuResidencyOptions.ENV_PROPERTY);
		System.setProperty(GpuResidencyOptions.ENV_PROPERTY, "on");
		ShardContext shard = shard();
		Qwen3TransformerHandler on = loadOnCuda(shard);
		assertThat(on.gpuResidencyWholeLayerActive()).as("every layer's weights are K-quant on the device").isTrue();
		int layers = shard.endLayer() - shard.startLayer();
		int[] run = new int[STEPS];
		System.arraycopy(PROMPT, 0, run, 0, PROMPT.length);
		long tokens = run.length;
		long steps = (long) layers * tokens;

		List<RecordedEvent> events = record(() -> decode(on, shard, "qwen3-res-layer", run));

		// The op-at-a-time sites of a Qwen3 decode layer: the K-quant GEMVs, the KV append and attention.
		for (String site : new String[] { "memcpy(K row H2D)", "memcpy(V row H2D)", "memcpy(gqa qBatch H2D)",
				"memcpy(gqa kPtrs H2D)", "memcpy(gqa outBatch D2H)", "cudaMemcpyAsync(x H2D q4k)",
				"cudaMemcpyAsync(y D2H q4k)", "materializeRows(resident activation)" })
			assertThat(decodeCopies(events, site)).as(site).isZero();
		assertThat(decodeCopies(events, "upload(decode region input)"))
				.as("the residual row goes up once per token: each later layer reads it on the device").isEqualTo(tokens);
		assertThat(decodeCopies(events, "materialize(decode region k, v, attention, layer output)"))
				.as("one download per layer and token").isEqualTo(steps);
		long h2d = decodeCopiesByDirection(events, "H2D");
		long d2h = decodeCopiesByDirection(events, "D2H");
		System.out.printf(java.util.Locale.ROOT, "QWEN3 GPU-RESIDENCY whole layer: per token %d H2D, %d D2H, %d layers%n",
				h2d / tokens, d2h / tokens, layers);
		assertThat(h2d).as("host-to-device copies, <= 1 x layers per token").isLessThanOrEqualTo(steps);
		assertThat(d2h).as("device-to-host copies, <= 1 x layers + 1 (the logits) per token")
				.isLessThanOrEqualTo(steps + tokens);
	}

	private List<RecordedEvent> record(Runnable body) throws Exception {
		Path jfr = tmp.resolve("qwen3-residency-" + System.nanoTime() + ".jfr");
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

	private static long decodeCopies(List<RecordedEvent> events, String site) {
		return events.stream().filter(e -> e.getEventType().getName().equals("juno.DeviceStaging"))
				.filter(e -> site.equals(e.getString("site")) && "decode".equals(e.getString("phase")))
				.mapToLong(e -> e.getLong("copies")).sum();
	}

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

	/** Teacher-forced decode of {@code tokens}, one position at a time, the path generation uses. */
	private static float[][] decode(Qwen3TransformerHandler h, ShardContext shard, String kv, int[] tokens) {
		float[][] out = new float[tokens.length][];
		int[] one = new int[1];
		for (int p = 0; p < tokens.length; p++) {
			one[0] = tokens[p];
			out[p] = h.forward(ForwardRequest.withTokens(kv, one, p), shard).logits();
		}
		h.evict(kv);
		return out;
	}

	/** {@link #PROMPT} followed by the handler's greedy continuation, {@link #STEPS} tokens in all. */
	private static int[] greedyRun(Qwen3TransformerHandler h, ShardContext shard) {
		int[] run = new int[STEPS];
		System.arraycopy(PROMPT, 0, run, 0, PROMPT.length);
		int[] one = new int[1];
		for (int p = 0; p < STEPS - 1; p++) {
			one[0] = run[p];
			float[] logits = h.forward(ForwardRequest.withTokens("qwen3-greedy", one, p), shard).logits();
			if (p + 1 >= PROMPT.length)
				run[p + 1] = argmax(logits);
		}
		h.evict("qwen3-greedy");
		return run;
	}

	private static ShardContext shard() throws Exception {
		try (GgufReader r = GgufReader.open(QWEN3)) {
			LlamaConfig cfg = LlamaConfig.from(r);
			return new ShardContext("n0", 0, cfg.numLayers(), true, true, cfg.vocabSize(), cfg.hiddenDim(),
					cfg.numHeads());
		}
	}
}
