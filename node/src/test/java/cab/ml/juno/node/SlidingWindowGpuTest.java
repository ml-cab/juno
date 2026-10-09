package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Sliding-window attention on CUDA, real models, under a forced patterned window
 * (period 2: even layers windowed). The host path is held exactly by
 * {@link SlidingWindowLiveTest}; here each device path is held to it.
 * <ul>
 * <li>Decode and multi-stream decode: one handler, a request attending on the
 * device against the same request with its device KV retired (attending on the
 * host), after the same device prefill. A device path that ignored the window
 * would differ by about the window's whole effect (windowed against unwindowed);
 * one that applies it differs by FP16 rounding, under a tenth of it.</li>
 * <li>Prefill: the GPU handler's against CPU handlers at widths W - 1, W and
 * W + 1. Width W must be the closest by at least 1.5x, which fails for an ignored
 * window and for an off-by-one alike. Over a whole prompt the rounding grows with
 * depth (up to about 15% of the effect on 28 to 36 layers), so a share of the
 * effect cannot tell it from one key too many; one key too many is 2x to 3x
 * further.</li>
 * </ul>
 * The bound the context-shift GPU tests use ({@code 2 * control + 0.02}) does not
 * carry over: an 8-key window averages each attention row over fewer FP16 keys,
 * and moves the logits far from their unwindowed values, so rounding grows with
 * it. Controls without a window are printed for the record.
 */
@Tag("gpu")
@DisplayName("Sliding-window attention - real models, CUDA")
class SlidingWindowGpuTest {

	private static final int N = 48;
	private static final int SHORT = 39;
	private static final int WIDTH = 8;
	/**
	 * A device path that ignored the window would differ from the host by about the
	 * window's whole effect; one that applies it differs by FP16 rounding, a small
	 * share of it.
	 */
	private static final double EFFECT_SHARE = 0.1;
	/** The CPU under one key fewer or more must sit at least this much further from the GPU prefill. */
	private static final double OFF_BY_ONE_MARGIN = 1.5;

	private static GpuContext gpu;

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

	interface Load {
		ForwardPassHandler load(Path path, ShardContext shard, MatVec backend) throws Exception;
	}

	/** Logits of one handler: prefill, decode on device and on host, multi-decode on device and on host. */
	private record Run(float[] prefill, float[] decode, float[] decodeHost, float[][] multi, float[][] multiHost,
			boolean gpuAttention, boolean prefillRegion) {
	}

	private static int[] tokens(int n, int salt) {
		int[] t = new int[n];
		for (int i = 0; i < n; i++)
			t[i] = ContextShiftLiveCheck.token(i * 3 + salt);
		return t;
	}

	private static Run run(ForwardPassHandler h, ShardContext shard) {
		int[] a = tokens(N, 0);
		int[] b = tokens(SHORT, 5);
		int next = ContextShiftLiveCheck.token(N);
		try {
			float[] prefill = h.forwardBatch(BatchForwardRequest.withTokens("a", a, 0), shard).lastLogits();
			float[] decode = h.forward(ForwardRequest.withTokens("a", new int[] { next }, N), shard).logits();
			h.forwardBatch(BatchForwardRequest.withTokens("h", a, 0), shard);
			ContextShiftLiveCheck.retireDeviceKv(h, "h");
			float[] decodeHost = h.forward(ForwardRequest.withTokens("h", new int[] { next }, N), shard).logits();
			float[][] multi = multi(h, shard, "m1", "m2", a, b, false);
			float[][] multiHost = multi(h, shard, "n1", "n2", a, b, true);
			return new Run(prefill, decode, decodeHost, multi, multiHost, h.gpuAttentionActive(), h.prefillRegionActive());
		} finally {
			for (String id : new String[] { "a", "h", "m1", "m2", "n1", "n2" })
				h.evict(id);
		}
	}

	private static float[][] multi(ForwardPassHandler h, ShardContext shard, String x, String y, int[] a, int[] b,
			boolean host) {
		h.forwardBatch(BatchForwardRequest.withTokens(x, a, 0), shard);
		h.forwardBatch(BatchForwardRequest.withTokens(y, b, 0), shard);
		if (host) {
			ContextShiftLiveCheck.retireDeviceKv(h, x);
			ContextShiftLiveCheck.retireDeviceKv(h, y);
		}
		return h.forwardMultiDecode(MultiDecodeForwardRequest.withTokens(List.of(x, y),
				new int[] { ContextShiftLiveCheck.token(N), ContextShiftLiveCheck.token(SHORT) }, new int[] { N, SHORT }),
				shard).logits();
	}

	private static Run load(Load loader, Path path, ShardContext shard, SlidingWindow window, boolean onGpu)
			throws Exception {
		MatVec backend = onGpu ? new CudaMatVec(gpu) : CpuMatVec.INSTANCE;
		ForwardPassHandler h = SlidingWindowLiveCheck.loadUnder(window, () -> loader.load(path, shard, backend));
		try {
			return run(h, shard);
		} finally {
			h.releaseGpuResources();
			if (backend instanceof CudaMatVec c)
				c.releaseScratch();
		}
	}

	private static double diff(float[][] a, float[][] b) {
		double m = 0;
		for (int i = 0; i < a.length; i++)
			m = Math.max(m, SlidingWindowLiveCheck.maxAbsDiff(a[i], b[i]));
		return m;
	}

	private static void check(String file, Load loader) throws Exception {
		Path path = SlidingWindowLiveCheck.model(file);
		assumeTrue(path.toFile().exists(), file + " not present");
		LlamaConfig cfg;
		try (GgufReader r = GgufReader.open(path)) {
			cfg = LlamaConfig.from(r);
		}
		ShardContext shard = new ShardContext("n0", 0, cfg.numLayers(), true, true, cfg.vocabSize(), cfg.hiddenDim(),
				cfg.numHeads());
		SlidingWindow window = SlidingWindowLiveCheck.periodTwo(WIDTH, cfg.numLayers());

		Run gpuNone = load(loader, path, shard, SlidingWindow.NONE, true);
		assumeTrue(gpuNone.gpuAttention(), "GPU attention not active for " + file);
		Run gpuWin = load(loader, path, shard, window, true);
		Run cpuNone = load(loader, path, shard, SlidingWindow.NONE, false);
		Run cpuWin = load(loader, path, shard, window, false);

		Run cpuNarrow = load(loader, path, shard, SlidingWindowLiveCheck.periodTwo(WIDTH - 1, cfg.numLayers()), false);
		Run cpuWide = load(loader, path, shard, SlidingWindowLiveCheck.periodTwo(WIDTH + 1, cfg.numLayers()), false);
		System.out.printf("%s window on GPU: gpu attention %s, prefill region %s%n", file, gpuWin.gpuAttention(),
				gpuWin.prefillRegion());

		double effect = Math.min(SlidingWindowLiveCheck.maxAbsDiff(gpuWin.decode(), gpuNone.decode()),
				SlidingWindowLiveCheck.maxAbsDiff(gpuWin.prefill(), gpuNone.prefill()));
		System.out.printf("%s window on GPU: window vs none %.4f; max |logit| windowed %.2f, no window %.2f%n", file,
				effect, maxAbs(gpuWin.decodeHost()), maxAbs(gpuNone.decodeHost()));
		assertThat(effect).as("the window changes the output").isGreaterThan(1.0);

		double[] control = { SlidingWindowLiveCheck.maxAbsDiff(gpuNone.decode(), gpuNone.decodeHost()),
				diff(gpuNone.multi(), gpuNone.multiHost()) };
		double[] windowed = { SlidingWindowLiveCheck.maxAbsDiff(gpuWin.decode(), gpuWin.decodeHost()),
				diff(gpuWin.multi(), gpuWin.multiHost()) };
		String[] names = { "decode, device vs host", "multi-stream decode, device vs host" };
		for (int i = 0; i < names.length; i++) {
			System.out.printf("%s window on GPU, %s: %.5f (%.4f of the window's effect; no window: %.5f)%n", file,
					names[i], windowed[i], windowed[i] / effect, control[i]);
			assertThat(windowed[i]).as("%s under the window, effect %.4f, control %.5f", names[i], effect, control[i])
					.isLessThanOrEqualTo(EFFECT_SHARE * effect);
		}

		// Prefill on the GPU against the CPU, whose window is exact: the CPU under the same width must be the
		// closest match, clearly ahead of one key fewer and one key more.
		double same = SlidingWindowLiveCheck.maxAbsDiff(gpuWin.prefill(), cpuWin.prefill());
		double narrow = SlidingWindowLiveCheck.maxAbsDiff(gpuWin.prefill(), cpuNarrow.prefill());
		double wide = SlidingWindowLiveCheck.maxAbsDiff(gpuWin.prefill(), cpuWide.prefill());
		System.out.printf("%s window on GPU, prefill GPU w%d vs CPU w%d %.4f, w%d %.4f, w%d %.4f (no window: %.5f)%n",
				file, WIDTH, WIDTH - 1, narrow, WIDTH, same, WIDTH + 1, wide,
				SlidingWindowLiveCheck.maxAbsDiff(gpuNone.prefill(), cpuNone.prefill()));
		assertThat(Math.min(narrow, wide)).as("prefill matches the CPU at width %d, not one key off", WIDTH)
				.isGreaterThanOrEqualTo(OFF_BY_ONE_MARGIN * same);
	}

	private static double maxAbs(float[] a) {
		double m = 0;
		for (float v : a)
			m = Math.max(m, Math.abs(v));
		return m;
	}

	@Test
	@DisplayName("TinyLlama: device prefill and decode regions, GPU attention")
	void tinyllama() throws Exception {
		check("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", LlamaTransformerHandler::load);
	}

	@Test
	@DisplayName("TinyLlama with both device regions off: GPU attention between host layers")
	void tinyllamaRegionsOff() throws Exception {
		String prevResidency = System.getProperty(GpuResidencyOptions.ENV_PROPERTY);
		String prevRegion = System.getProperty(PrefillWindowRegion.ENV_PROPERTY);
		System.setProperty(GpuResidencyOptions.ENV_PROPERTY, "off");
		System.setProperty(PrefillWindowRegion.ENV_PROPERTY, "off");
		try {
			check("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", LlamaTransformerHandler::load);
		} finally {
			restore(GpuResidencyOptions.ENV_PROPERTY, prevResidency);
			restore(PrefillWindowRegion.ENV_PROPERTY, prevRegion);
		}
	}

	@Test
	@DisplayName("Qwen2.5-3B (LLaMA handler, split-half pairs: no decode region)")
	void qwen25() throws Exception {
		check("qwen2.5-3b-instruct-q4_k_m.gguf", LlamaTransformerHandler::load);
	}

	@Test
	@DisplayName("Phi-3.5-mini (Phi-3 handler: mirror, decode and prefill regions)")
	void phi35() throws Exception {
		check("Phi-3.5-mini-instruct-Q4_K_M.gguf", Phi3TransformerHandler::load);
	}

	@Test
	@DisplayName("Qwen3-1.7B (Qwen3 handler: mirror, decode and prefill regions)")
	void qwen3() throws Exception {
		check("Qwen3-1.7B-Q4_K_M.gguf", Qwen3TransformerHandler::load);
	}

	private static void restore(String key, String prev) {
		if (prev == null)
			System.clearProperty(key);
		else
			System.setProperty(key, prev);
	}
}
