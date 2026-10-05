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
import java.util.List;
import java.util.Locale;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * The GPU attention kernel in the Phi-3 and Qwen3 handlers, on real models,
 * against the same handler with {@code --gpu-attention off} on the same CUDA
 * backend. Only attention differs between the two runs (FP16 KV mirror and the
 * kernel, against FP32 host KV and scalar attention), so this isolates the kernel
 * integration from the rest of the GPU path.
 *
 * <p>Each run covers all three attention call sites: a prefill window
 * ({@code forwardBatch}), a single-token decode ({@code forward}) and a two-stream
 * decode ({@code forwardMultiDecode}) where the streams sit at different
 * positions. It also checks that the handler reports the kernel truthfully and
 * that {@code evict} frees every device KV byte.
 *
 * <p>Two checks, each measuring one thing (restated 2026-10-05; until then every
 * site held the logits bound below, which compared the kernel's rounding order with
 * the scalar path's as much as it checked the kernel):
 * <ul>
 * <li><b>The kernel, at every launch.</b> A {@link GpuAttentionMirror.DispatchObserver}
 * sees each launch's queries, lengths, mirrors and output, and the output is held to
 * {@link GqaMath#attend} over the same FP16 rows read back from the mirror: relative
 * L2 at most {@value #KERNEL_REL_L2_MAX} per row, on every launch of every call site
 * kind (prefill window, single decode, multi-stream decode), each of which must be
 * seen. This isolates the kernel from the FP16 rounding of the cache, which is the
 * same on both sides. Measured 8.3e-7 to 3.5e-6; planted fault (below) 0.71 and 0.91.</li>
 * <li><b>The integration, at the prefill windows.</b> Logits relative L2 at most
 * {@value #LOGITS_REL_L2_MAX} and the same top-1 token, kernel on against off. That
 * is the bound {@code GpuForwardPassIT} holds the whole GPU path to against the CPU.
 * Measured 0.0001 to 0.0006 there; planted fault (one head's output zeroed after
 * every launch) 0.153 and 0.090 at the first window, top-1 unchanged.</li>
 * </ul>
 * The three decode sites' logits are printed, not bounded. There, on against off is
 * dominated by the FP16 cache's rounding amplified through the model, and it moved from
 * 0.0149 to between 0.0223 and 0.0368 on Qwen3-1.7B's second stream across attention
 * kernels whose own error is about 1e-6, so a bound there passes or fails by draw.
 *
 * <p>Both runs multiply prefill windows on the FP16 dequant route
 * ({@link CudaMatVec#dequantizeBatchedKQuant}), as the calibration did. On the
 * default tiled integer route each matmul rounds its input to 8 bits, and the
 * attention difference this test measures (3e-4 at the prefill windows) comes back
 * as whole rounding steps: 0.008 to 0.009 at the prefill windows and 0.038 at the
 * second multi-decode stream on Qwen3-1.7B, measured. That is the 8-bit rounding's
 * doing, which the packed matmul's own parity and quality tests bound, not the
 * attention kernel's.
 */
@Tag("gpu")
@DisplayName("Phi-3 and Qwen3 handlers - GPU attention against scalar attention on a real model")
class GpuAttentionHandlerParityTest {

	private static final double LOGITS_REL_L2_MAX = 0.025;
	/** The first two sites are prefill windows, where the logits bound applies. */
	private static final int PREFILL_SITES = 2;
	/**
	 * Kernel output against {@link GqaMath#attend} over the same FP16 rows, worst row.
	 * Measured on the reference host: 8.3e-7 to 3.5e-6 across all three call-site kinds
	 * on both models. Planted fault (one head's output zeroed after every launch): 0.91
	 * (Phi-3.5-mini) and 0.71 (Qwen3-1.7B) at the first prefill window.
	 */
	private static final double KERNEL_REL_L2_MAX = 1e-3;
	private static final int PROMPT_A = 48;
	private static final int PROMPT_B = 24;
	/** Allocator granularity and driver bookkeeping; far below one layer of either model. */
	private static final long VRAM_SLACK = 16L << 20;

	private static GpuContext gpu;
	private String saved;

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
			System.clearProperty(GpuAttentionOptions.ENV_PROPERTY);
		else
			System.setProperty(GpuAttentionOptions.ENV_PROPERTY, saved);
	}

	@ParameterizedTest(name = "{0}")
	@ValueSource(strings = { "Phi-3.5-mini-instruct-Q4_K_M.gguf", "Qwen3-1.7B-Q4_K_M.gguf" })
	@DisplayName("kernel matches the oracle at every launch, prefill logits match kernel off, reports itself, frees its mirror")
	void kernel_matches_scalar_attention(String file) throws Exception {
		Path model = model(file);
		assumeTrue(model.toFile().exists(), "Skipping - model not found: " + model);
		saved = System.getProperty(GpuAttentionOptions.ENV_PROPERTY);
		ShardContext shard = wholeModel(model);
		long freeAtStart = gpu.freeVramBytes();

		System.setProperty(GpuAttentionOptions.ENV_PROPERTY, "off");
		CudaMatVec offBackend = new CudaMatVec(gpu);
		offBackend.dequantizeBatchedKQuant(true);
		ForwardPassHandler off = ForwardPassHandlerLoader.load(model, shard, offBackend);
		float[][] ref;
		try {
			assertThat(off.gpuAttentionActive()).as("off must not activate the kernel").isFalse();
			long before = DeviceKvCache.allocatedBytes();
			ref = run(off, shard, null);
			assertThat(DeviceKvCache.allocatedBytes()).as("off allocates no device KV").isEqualTo(before);
		} finally {
			off.releaseGpuResources();
			offBackend.releaseScratch();
		}

		System.clearProperty(GpuAttentionOptions.ENV_PROPERTY); // the default, auto
		CudaMatVec onBackend = new CudaMatVec(gpu);
		onBackend.dequantizeBatchedKQuant(true);
		ForwardPassHandler on = ForwardPassHandlerLoader.load(model, shard, onBackend);
		float[][] got;
		KernelCheck check = new KernelCheck();
		GpuAttentionMirror.observer = check;
		try {
			assertThat(on.gpuAttentionActive()).as("the default must activate the kernel on CUDA").isTrue();
			long before = DeviceKvCache.allocatedBytes();
			got = run(on, shard, before);
			assertThat(DeviceKvCache.allocatedBytes()).as("evict frees every device KV byte").isEqualTo(before);
		} finally {
			GpuAttentionMirror.observer = null;
			on.releaseGpuResources();
			onBackend.releaseScratch();
		}
		assertThat(gpu.freeVramBytes()).as("device memory back to where the test started (weights, KV mirror,"
				+ " kernel scratch and backend scratch all released)").isGreaterThanOrEqualTo(freeAtStart - VRAM_SLACK);

		String[] site = { "prefill window A", "prefill window B", "single decode", "multi-decode A",
				"multi-decode B" };
		for (Kind kind : Kind.values()) {
			System.out.printf(Locale.ROOT, "GPU-ATTN-KERNEL %s %-12s launches=%d rows=%d max-relL2=%.3e%n", file, kind,
					check.launches[kind.ordinal()], check.rows[kind.ordinal()], check.maxRel[kind.ordinal()]);
			assertThat(check.launches[kind.ordinal()]).as(kind + ": launches seen").isPositive();
			assertThat(check.maxRel[kind.ordinal()]).as(kind + ": kernel against GqaMath over the mirror's rows")
					.isLessThanOrEqualTo(KERNEL_REL_L2_MAX);
		}
		for (int i = 0; i < ref.length; i++) {
			double rel = relativeL2(got[i], ref[i]);
			System.out.printf(Locale.ROOT, "GPU-ATTN %s %-16s top1 off=%d on=%d relL2=%.6f%s%n", file, site[i],
					argmax(ref[i]), argmax(got[i]), rel, i < PREFILL_SITES ? "" : " (reported)");
			if (i < PREFILL_SITES) {
				assertThat(argmax(got[i])).as(site[i] + ": top-1 token, kernel on vs off").isEqualTo(argmax(ref[i]));
				assertThat(rel).as(site[i] + ": logits relative L2, kernel on vs off")
						.isLessThanOrEqualTo(LOGITS_REL_L2_MAX);
			}
		}
	}

	private enum Kind {
		WINDOW, DECODE, STREAMS
	}

	/** Holds every launch's output to {@link GqaMath#attend} over the rows the kernel read. */
	private static final class KernelCheck implements GpuAttentionMirror.DispatchObserver {
		final int[] launches = new int[Kind.values().length];
		final long[] rows = new long[Kind.values().length];
		final double[] maxRel = new double[Kind.values().length];

		@Override
		public void dispatched(DeviceKvCache[] mirrors, float[][] q, int[] seqLens, float[][] out, int numHeads,
				int headDim, int gqaRatio, int kvDim) {
			boolean oneCache = true;
			for (DeviceKvCache m : mirrors)
				oneCache &= m == mirrors[0];
			Kind kind = !oneCache ? Kind.STREAMS : q.length == 1 ? Kind.DECODE : Kind.WINDOW;
			int k = kind.ordinal();
			launches[k]++;
			DeviceKvCache cached = null;
			float[] kRows = null;
			float[] vRows = null;
			for (int b = 0; b < q.length; b++) {
				if (mirrors[b] != cached || kRows.length < seqLens[b] * kvDim) {
					int len = oneCache ? max(seqLens) : seqLens[b];
					cached = mirrors[b];
					kRows = cached.downloadK(len);
					vRows = cached.downloadV(len);
				}
				float[] expected = new float[numHeads * headDim];
				GqaMath.attend(q[b], kRows, vRows, seqLens[b], expected, new float[seqLens[b]], numHeads, headDim,
						gqaRatio, kvDim);
				maxRel[k] = Math.max(maxRel[k], relativeL2(out[b], expected));
				rows[k]++;
			}
		}

		private static int max(int[] v) {
			int m = 0;
			for (int x : v)
				m = Math.max(m, x);
			return m;
		}
	}

	/**
	 * Prefills two requests, decodes one alone and then both together at their
	 * different positions, and returns the five logit vectors. When
	 * {@code mirrorBaseline} is set, also checks that the prefill allocated device
	 * KV for the kernel to read.
	 */
	private static float[][] run(ForwardPassHandler h, ShardContext shard, Long mirrorBaseline) {
		int[] a = tokens(PROMPT_A, 300, shard.vocabSize());
		int[] b = tokens(PROMPT_B, 900, shard.vocabSize());
		float[] pa = h.forwardBatch(BatchForwardRequest.withTokens("a", a, 0), shard).lastLogits();
		if (mirrorBaseline != null)
			assertThat(DeviceKvCache.allocatedBytes()).as("a prefill allocates the device KV mirror")
					.isGreaterThan(mirrorBaseline);
		float[] pb = h.forwardBatch(BatchForwardRequest.withTokens("b", b, 0), shard).lastLogits();
		float[] da = h.forward(ForwardRequest.withTokens("a", new int[] { argmax(pa) }, PROMPT_A), shard).logits();
		float[][] md = h.forwardMultiDecode(MultiDecodeForwardRequest.withTokens(List.of("a", "b"),
				new int[] { argmax(da), argmax(pb) }, new int[] { PROMPT_A + 1, PROMPT_B }), shard).logits();
		h.evict("a");
		h.evict("b");
		return new float[][] { pa, pb, da, md[0], md[1] };
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
			return new ShardContext("gpu-attn", 0, cfg.numLayers(), true, true, cfg.vocabSize(), cfg.hiddenDim(),
					cfg.numHeads());
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
