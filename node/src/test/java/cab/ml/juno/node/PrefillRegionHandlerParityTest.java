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

import jdk.jfr.Recording;
import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordingFile;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Locale;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The prefill-window device region on real models: the same CUDA backend with the
 * region on (the default) and off, on the LLaMA family with adjacent RoPE
 * (TinyLlama), the LLaMA family with split-half RoPE and Q/K/V biases
 * (Qwen2.5-3B), Phi-3.5-mini and Qwen3-1.7B.
 *
 * <p>Checked at every place a prefill window feeds: two fresh windows, a
 * continuation window that starts after a written prefix, a single decode and a
 * two-stream decode. The logits must be bit-identical: every operation in the
 * region reproduces the host window path's arithmetic (the GEMMs see the same FP16
 * bits, the norm sums in the host loop's order), so any difference is a defect,
 * not rounding - an earlier region with a tree-reduced device norm moved one
 * Qwen2.5 decode step's logits by 0.09 relative L2 through flipped FP16 roundings.
 * From a JFR recording of the region's first window: no host SwiGLU or residual
 * add, no matmul result copied back to the host, the KV mirror written with at
 * most one K and one V copy per layer, and the residual stream uploaded once and
 * downloaded once for the whole window rather than once per layer. Device memory returns to where it started
 * once the handler is released, and a repeated window allocates nothing new.
 */
@Tag("gpu")
@DisplayName("Prefill-window device region - region on against region off on real models")
class PrefillRegionHandlerParityTest {

	private static final int PROMPT_A = 48;
	private static final int PROMPT_B = 24;
	private static final int CONTINUATION = 16;
	/** Allocator granularity and driver bookkeeping; far below one layer of any of these models. */
	private static final long VRAM_SLACK = 16L << 20;

	/** The batched matmul's result copies back to the host; none may remain in a region window. */
	private static final Set<String> MATMUL_D2H_SITES = Set.of("cudaMemcpyAsync(y D2H batched-gemm)",
			"cudaMemcpyAsync(y D2H q4k-batched-gemm)", "cudaMemcpyAsync(y D2H batched)");

	private static GpuContext gpu;
	private String saved;

	@TempDir
	Path tmp;

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
	void restore() {
		if (saved == null)
			System.clearProperty(PrefillWindowRegion.ENV_PROPERTY);
		else
			System.setProperty(PrefillWindowRegion.ENV_PROPERTY, saved);
	}

	@ParameterizedTest(name = "{0}")
	@ValueSource(strings = { "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", "qwen2.5-3b-instruct-q4_k_m.gguf",
			"Phi-3.5-mini-instruct-Q4_K_M.gguf", "Qwen3-1.7B-Q4_K_M.gguf" })
	@DisplayName("region on is bit-identical to region off, leaves no host elementwise or matmul copy-back, frees its memory")
	void region_matches_hostPath(String file) throws Exception {
		Path model = model(file);
		assumeTrue(model.toFile().exists(), "Skipping - model not found: " + model);
		saved = System.getProperty(PrefillWindowRegion.ENV_PROPERTY);
		ShardContext shard = wholeModel(model);
		int layers = shard.endLayer() - shard.startLayer();
		long freeAtStart = gpu.freeVramBytes();

		System.setProperty(PrefillWindowRegion.ENV_PROPERTY, "off");
		CudaMatVec offBackend = new CudaMatVec(gpu);
		ForwardPassHandler off = ForwardPassHandlerLoader.load(model, shard, offBackend);
		float[][] ref;
		try {
			assertThat(off.prefillRegionActive()).as("off must not build the region").isFalse();
			ref = run(off, shard, null);
		} finally {
			off.releaseGpuResources();
			offBackend.releaseScratch();
		}

		System.clearProperty(PrefillWindowRegion.ENV_PROPERTY); // the default, on
		CudaMatVec onBackend = new CudaMatVec(gpu);
		ForwardPassHandler on = ForwardPassHandlerLoader.load(model, shard, onBackend);
		float[][] got;
		try {
			assertThat(on.prefillRegionActive()).as("the default must build the region on CUDA").isTrue();
			got = run(on, shard, new Recorded(layers));
		} finally {
			on.releaseGpuResources();
			onBackend.releaseScratch();
		}
		assertThat(gpu.freeVramBytes()).as("device memory back to where the test started (weights, KV mirror,"
				+ " region buffers and backend scratch all released)").isGreaterThanOrEqualTo(freeAtStart - VRAM_SLACK);

		String[] site = { "prefill window A", "prefill window B", "continuation window A", "single decode",
				"multi-decode A", "multi-decode B" };
		for (int i = 0; i < ref.length; i++) {
			double rel = relativeL2(got[i], ref[i]);
			System.out.printf(Locale.ROOT, "PREFILL-REGION %s %-22s top1 off=%d on=%d relL2=%.6f%n", file, site[i],
					argmax(ref[i]), argmax(got[i]), rel);
			assertThat(got[i]).as(site[i] + ": logits, region on vs off").containsExactly(ref[i]);
		}
	}

	/** What the region run records about its first window, checked inside {@link #run}. */
	private final class Recorded {
		final int layers;

		Recorded(int layers) {
			this.layers = layers;
		}

		void check(List<RecordedEvent> events, long freeAfterFirstWindow, long freeAfterRepeat) {
			assertThat(count(events, "juno.SwiGlu")).as("host SwiGLU events in a region window").isZero();
			assertThat(count(events, "juno.ResidualAdd")).as("host residual-add events in a region window").isZero();
			long matmulD2h = staging(events).stream().filter(e -> MATMUL_D2H_SITES.contains(e.getString("site")))
					.mapToLong(e -> e.getLong("copies")).sum();
			assertThat(matmulD2h).as("matmul results copied back to the host in a region window").isZero();
			long kvH2d = staging(events).stream().filter(e -> "H2D".equals(e.getString("direction")))
					.filter(e -> e.getString("site").startsWith("memcpy(K ") || e.getString("site").startsWith("memcpy(V "))
					.mapToLong(e -> e.getLong("copies")).sum();
			assertThat(kvH2d).as("KV mirror host-to-device copies for one window").isLessThanOrEqualTo(2L * layers);
			assertThat(copies(events, "H2D", "upload(prefill residual)"))
					.as("residual uploads for one window (kept on the device across layers)").isEqualTo(1);
			assertThat(copies(events, "D2H", "materialize(prefill residual)"))
					.as("residual downloads for one window (kept on the device across layers)").isEqualTo(1);
			assertThat(freeAfterRepeat).as("a repeated window allocates no new device memory")
					.isGreaterThanOrEqualTo(freeAfterFirstWindow - VRAM_SLACK);
		}
	}

	/**
	 * Prefills two requests, continues the first with a second window, decodes one
	 * alone and then both together at their different positions, and returns the six
	 * logit vectors. With {@code recorded}, records the first window and checks it.
	 */
	private float[][] run(ForwardPassHandler h, ShardContext shard, Recorded recorded) throws Exception {
		int[] a = tokens(PROMPT_A, 300, shard.vocabSize());
		int[] b = tokens(PROMPT_B, 900, shard.vocabSize());
		int[] c = tokens(CONTINUATION, 1700, shard.vocabSize());
		float[] pa;
		if (recorded != null) {
			Path jfr = tmp.resolve("region-" + System.nanoTime() + ".jfr");
			float[][] first = new float[1][];
			try (Recording rec = new Recording()) {
				rec.enable("juno.DeviceStaging").withThreshold(Duration.ZERO);
				rec.enable("juno.SwiGlu").withThreshold(Duration.ZERO);
				rec.enable("juno.ResidualAdd").withThreshold(Duration.ZERO);
				rec.setDestination(jfr);
				rec.start();
				first[0] = h.forwardBatch(BatchForwardRequest.withTokens("a", a, 0), shard).lastLogits();
				rec.stop();
			}
			pa = first[0];
			long freeAfterFirst = gpu.freeVramBytes();
			h.forwardBatch(BatchForwardRequest.withTokens("repeat", a, 0), shard);
			h.evict("repeat");
			recorded.check(RecordingFile.readAllEvents(jfr), freeAfterFirst, gpu.freeVramBytes());
		} else {
			pa = h.forwardBatch(BatchForwardRequest.withTokens("a", a, 0), shard).lastLogits();
		}
		float[] pb = h.forwardBatch(BatchForwardRequest.withTokens("b", b, 0), shard).lastLogits();
		float[] pc = h.forwardBatch(BatchForwardRequest.withTokens("a", c, PROMPT_A), shard).lastLogits();
		int posA = PROMPT_A + CONTINUATION;
		float[] da = h.forward(ForwardRequest.withTokens("a", new int[] { argmax(pc) }, posA), shard).logits();
		float[][] md = h.forwardMultiDecode(MultiDecodeForwardRequest.withTokens(List.of("a", "b"),
				new int[] { argmax(da), argmax(pb) }, new int[] { posA + 1, PROMPT_B }), shard).logits();
		h.evict("a");
		h.evict("b");
		return new float[][] { pa, pb, pc, da, md[0], md[1] };
	}

	private static List<RecordedEvent> staging(List<RecordedEvent> events) {
		return events.stream().filter(e -> e.getEventType().getName().equals("juno.DeviceStaging")).toList();
	}

	private static long copies(List<RecordedEvent> events, String direction, String site) {
		return staging(events).stream().filter(e -> direction.equals(e.getString("direction")))
				.filter(e -> site.equals(e.getString("site"))).mapToLong(e -> e.getLong("copies")).sum();
	}

	private static long count(List<RecordedEvent> events, String name) {
		return events.stream().filter(e -> e.getEventType().getName().equals(name)).count();
	}

	/** Deterministic, ordinary-vocabulary token ids (away from the special tokens at either end). */
	private static int[] tokens(int n, int seed, int vocab) {
		int[] t = new int[n];
		for (int i = 0; i < n; i++)
			t[i] = 1000 + (int) (((long) seed + 7919L * i) % (vocab / 2));
		return t;
	}

	private static ShardContext wholeModel(Path model) throws Exception {
		try (GgufReader r = GgufReader.open(model)) {
			LlamaConfig cfg = LlamaConfig.from(r);
			return new ShardContext("prefill-region", 0, cfg.numLayers(), true, true, cfg.vocabSize(),
					cfg.hiddenDim(), cfg.numHeads());
		}
	}

	private static Path model(String file) {
		Path here = Path.of(System.getProperty("user.dir"));
		Path root = here.endsWith("node") ? here.getParent() : here;
		return root.resolve("models").resolve(file);
	}

	private static double relativeL2(float[] actual, float[] expected) {
		double diff = 0, norm = 0;
		for (int i = 0; i < expected.length; i++) {
			double d = (double) actual[i] - expected[i];
			diff += d * d;
			norm += (double) expected[i] * expected[i];
		}
		return Math.sqrt(diff / norm);
	}

	private static int argmax(float[] a) {
		int best = 0;
		for (int i = 1; i < a.length; i++)
			if (a[i] > a[best])
				best = i;
		return best;
	}
}
