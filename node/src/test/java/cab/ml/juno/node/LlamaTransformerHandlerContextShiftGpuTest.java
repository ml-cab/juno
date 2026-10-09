package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Path;
import java.util.Arrays;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import cab.ml.juno.kvcache.SessionKvTensor;

/**
 * Context shift on a CUDA-backed TinyLlama: the device KV mirror is rebuilt from
 * the shifted host KV (watermark at the new length, not retired), and the next
 * token's logits, attended on the device, match a request holding the oracle KV
 * ({@link ContextShiftOracle}) attended on the host, within the FP16 mirror's
 * tolerance.
 */
@Tag("gpu")
@DisplayName("LlamaTransformerHandler — context shift with the device KV mirror")
class LlamaTransformerHandlerContextShiftGpuTest {

	private static final Path MODEL = Path.of(System.getProperty("user.dir")).endsWith("node")
			? Path.of(System.getProperty("user.dir")).getParent().resolve("models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
			: Path.of("models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
	private static final int N = 96;
	private static final int KEEP = 8;
	private static final int DISCARD = 40;

	private static GpuContext gpu;

	static boolean modelPresent() {
		return MODEL.toFile().exists();
	}

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

	private static int token(int i) {
		return 300 + (i * 37) % 2000;
	}

	@Test
	@EnabledIf("modelPresent")
	@DisplayName("mirror rewritten to the shifted length; next logits match the oracle KV attended on the host")
	void shiftRewritesMirrorAndMatchesOracle() throws Exception {
		ShardContext shard;
		LlamaConfig cfg;
		try (GgufReader r = GgufReader.open(MODEL)) {
			cfg = LlamaConfig.from(r);
			shard = new ShardContext("n0", 0, cfg.numLayers(), true, true, cfg.vocabSize(), cfg.hiddenDim(),
					cfg.numHeads());
		}
		CudaMatVec backend = new CudaMatVec(gpu);
		LlamaTransformerHandler h = LlamaTransformerHandler.load(MODEL, shard, backend);
		try {
			assumeTrue(h.gpuAttentionActive(), "GPU attention not active on this host");
			int[] all = new int[N + 1];
			for (int i = 0; i <= N; i++)
				all[i] = token(i);

			h.forwardBatch(BatchForwardRequest.withTokens("s", Arrays.copyOf(all, N), 0), shard);
			h.shiftKv("s", N, KEEP, DISCARD);
			int newLen = N - DISCARD;
			int[] marks = h.deviceKvWatermarks("s");
			assertThat(marks).as("mirror watermarks after the shift").isNotNull();
			for (int m : marks)
				assertThat(m).as("each in-use mirror holds exactly the shifted history").isIn(0, newLen);
			assertThat(Arrays.stream(marks).max().getAsInt()).isEqualTo(newLen);
			float[] shifted = h.forward(ForwardRequest.withTokens("s", new int[] { all[N] }, newLen), shard).logits();
			assertThat(Arrays.stream(h.deviceKvWatermarks("s")).max().getAsInt())
					.as("the decode step after the shift appended to the mirror").isEqualTo(newLen + 1);

			int[] prefix = Arrays.copyOf(all, N);
			h.forwardBatch(BatchForwardRequest.withTokens("o", prefix, 0), shard);
			h.retireDeviceKv("o");
			SessionKvTensor[][] o = h.hostKv("o");
			int kvDim = cfg.kvDim();
			int kvHeads = cfg.numKvHeads();
			int headDim = cfg.headDim();
			float theta = cfg.ropeTheta();
			ContextShiftOracle.inject(o[0], ContextShiftOracle.expected(ContextShiftOracle.snapshot(o[0], N), kvDim, N,
					KEEP, DISCARD, true,
					(row, pos) -> LoraTrainingMath.ropeBackward(row, pos, kvHeads, headDim, theta, RopePairing.ADJACENT),
					(row, pos) -> LlamaTransformerHandler.rope(row, pos, kvHeads, headDim, theta, RopePairing.ADJACENT)),
					kvDim, newLen);
			ContextShiftOracle.inject(o[1], ContextShiftOracle.expected(ContextShiftOracle.snapshot(o[1], N), kvDim, N,
					KEEP, DISCARD, false, null, null), kvDim, newLen);
			float[] fresh = h.forward(ForwardRequest.withTokens("o", new int[] { all[N] }, newLen), shard).logits();

			// Control without a shift: the same request decoded with device attention and with
			// host attention differs by the FP16 mirror's rounding alone. The shift may add
			// no more than a small margin on top of it.
			h.forwardBatch(BatchForwardRequest.withTokens("gd", prefix, 0), shard);
			h.forwardBatch(BatchForwardRequest.withTokens("gh", prefix, 0), shard);
			h.retireDeviceKv("gh");
			float[] device = h.forward(ForwardRequest.withTokens("gd", new int[] { all[N] }, N), shard).logits();
			float[] host = h.forward(ForwardRequest.withTokens("gh", new int[] { all[N] }, N), shard).logits();
			double control = maxAbsDiff(device, host);
			double maxDiff = maxAbsDiff(shifted, fresh);
			System.out.printf("context shift on GPU: max |logit diff| = %.5f (device vs host attention without a"
					+ " shift: %.5f)%n", maxDiff, control);
			assertThat(argmax(shifted)).as("greedy next token").isEqualTo(argmax(fresh));
			assertThat(maxDiff).as("max |logit diff| against the oracle, control %.5f", control)
					.isLessThanOrEqualTo(2 * control + 0.02);
			h.evict("gd");
			h.evict("gh");
			h.evict("s");
			h.evict("o");
		} finally {
			h.releaseGpuResources();
			backend.releaseScratch();
		}
	}

	private static double maxAbsDiff(float[] a, float[] b) {
		double m = 0;
		for (int i = 0; i < a.length; i++)
			m = Math.max(m, Math.abs(a[i] - b[i]));
		return m;
	}

	private static int argmax(float[] a) {
		int best = 0;
		for (int i = 1; i < a.length; i++)
			if (a[i] > a[best])
				best = i;
		return best;
	}
}
