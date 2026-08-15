package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import cab.ml.juno.lora.LoraAdapter;
import cab.ml.juno.lora.LoraAdapterSet;

/**
 * CPU↔GPU LoRA train parity and speed gates (Tier 9).
 * Run: {@code mvn test -Dgroups=gpu -pl node -Dtest=LoraTrainableHandlerGpuBackwardTest}.
 */
@Tag("gpu")
@DisplayName("LoraTrainableHandler — GPU backward parity (requires CUDA + TinyLlama)")
class LoraTrainableHandlerGpuBackwardTest {

	private static final float LOSS_TOL = 2e-3f;
	private static final float GRAD_TOL = 5e-3f;
	private static final int RANK = 8;
	private static final float ALPHA = 16f;
	private static final int SEQ = 64;
	private static final long SEED = 42L;

	private static GpuContext ctx;
	private static CudaMatVec cuda;
	private static Path modelPath;
	private static String prevMicrobatch;

	static boolean tinyLlamaPresent() {
		return Files.isRegularFile(modelFile());
	}

	private static Path modelFile() {
		Path cwd = Path.of(System.getProperty("user.dir"));
		Path root = cwd.endsWith("node") ? cwd.getParent() : cwd;
		return root.resolve("models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
	}

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping — no CUDA device");
		assumeTrue(tinyLlamaPresent(), "Skipping — TinyLlama fixture absent");
		prevMicrobatch = System.getProperty("juno.lora.microbatch");
		System.setProperty("juno.lora.microbatch", "8");
		ctx = GpuContext.init(0);
		cuda = new CudaMatVec(ctx);
		modelPath = modelFile();
	}

	@AfterAll
	static void destroy() {
		if (prevMicrobatch == null)
			System.clearProperty("juno.lora.microbatch");
		else
			System.setProperty("juno.lora.microbatch", prevMicrobatch);
		if (ctx != null)
			ctx.close();
	}

	@Test
	@EnabledIf("tinyLlamaPresent")
	@DisplayName("CPU vs GPU: loss and A/B grads agree (qv, seq 64)")
	void cpu_gpu_grad_parity_qv() throws Exception {
		parityRun(LoraProjection.qv(), SEQ);
	}

	@Test
	@EnabledIf("tinyLlamaPresent")
	@DisplayName("Milestone-1 / microbatch speed: GPU backward and e2e vs CPU")
	void speed_gates_qv() throws Exception {
		LlamaConfig cfg;
		try (GgufReader r = GgufReader.open(modelPath)) {
			cfg = LlamaConfig.from(r);
		}
		ShardContext shard = new ShardContext("n0", 0, cfg.numLayers(), true, true, cfg.vocabSize(),
				cfg.hiddenDim(), cfg.numHeads());
		int[] tokens = randomTokens(SEQ + 1, Math.min(cfg.vocabSize(), 1000), SEED);

		// Warm-up both paths
		timeOnce(cfg, shard, tokens, CpuMatVec.INSTANCE);
		timeOnce(cfg, shard, tokens, cuda);

		long cpuNs = 0, gpuNs = 0;
		long cpuBackNs = 0, gpuBackNs = 0;
		int reps = 3;
		for (int i = 0; i < reps; i++) {
			var cpu = timeOnce(cfg, shard, tokens, CpuMatVec.INSTANCE);
			var gpu = timeOnce(cfg, shard, tokens, cuda);
			cpuNs += cpu[0];
			gpuNs += gpu[0];
			cpuBackNs += cpu[1];
			gpuBackNs += gpu[1];
		}
		double e2eSpeedup = (double) cpuNs / (double) gpuNs;
		double backSpeedup = (double) cpuBackNs / (double) gpuBackNs;
		System.out.printf(
				"LoRA speed gate (TinyLlama qv rank=8 seq=%d microbatch=8): e2e %.2fx  backward %.2fx  "
						+ "cpuE2eMs=%.1f gpuE2eMs=%.1f cpuBackMs=%.1f gpuBackMs=%.1f%n",
				SEQ, e2eSpeedup, backSpeedup,
				cpuNs / 1e6 / reps, gpuNs / 1e6 / reps,
				cpuBackNs / 1e6 / reps, gpuBackNs / 1e6 / reps);

		assertThat(backSpeedup)
				.as("GPU frozen backward should be >= 2x CPU (got %.2fx)", backSpeedup)
				.isGreaterThanOrEqualTo(2.0);
		assertThat(e2eSpeedup)
				.as("GPU e2e train step should be >= 1.5x CPU (got %.2fx)", e2eSpeedup)
				.isGreaterThanOrEqualTo(1.5);
	}

	private void parityRun(java.util.List<LoraProjection> targets, int seq) throws Exception {
		LlamaConfig cfg;
		try (GgufReader r = GgufReader.open(modelPath)) {
			cfg = LlamaConfig.from(r);
		}
		ShardContext shard = new ShardContext("n0", 0, cfg.numLayers(), true, true, cfg.vocabSize(),
				cfg.hiddenDim(), cfg.numHeads());
		int[] tokens = randomTokens(seq + 1, Math.min(cfg.vocabSize(), 1000), SEED);

		LoraAdapterSet cpuAdapters = LoraInitializer.create(cfg, targets, RANK, ALPHA, new Random(SEED));
		LoraAdapterSet gpuAdapters = LoraInitializer.create(cfg, targets, RANK, ALPHA, new Random(SEED));

		LoraTrainableHandler cpuH = LoraTrainableHandler.load(modelPath, shard, cpuAdapters, CpuMatVec.INSTANCE);
		LoraTrainableHandler gpuH = LoraTrainableHandler.load(modelPath, shard, gpuAdapters, cuda);
		try {
			cpuAdapters.zeroAllGrads();
			gpuAdapters.zeroAllGrads();
			var cpuGr = cpuH.computeGradients(tokens);
			var gpuGr = gpuH.computeGradients(tokens);

			assertThat(gpuGr.lossSum()).isCloseTo(cpuGr.lossSum(), within(LOSS_TOL * Math.max(1f, Math.abs(cpuGr.lossSum()))));
			assertThat(gpuGr.predictionCount()).isEqualTo(cpuGr.predictionCount());

			for (var e : cpuAdapters.asMap().entrySet()) {
				LoraAdapter ca = e.getValue();
				LoraAdapter ga = gpuAdapters.asMap().get(e.getKey());
				assertThat(ga).as(e.getKey()).isNotNull();
				assertClose(ca.gradA(), ga.gradA(), e.getKey() + ".gradA");
				assertClose(ca.gradB(), ga.gradB(), e.getKey() + ".gradB");
			}
		} finally {
			cpuH.releaseGpuResources();
			gpuH.releaseGpuResources();
		}
	}

	/** @return {e2eNanos, backwardMillisAsNanos} from LoraGradientResult timings */
	private static long[] timeOnce(LlamaConfig cfg, ShardContext shard, int[] tokens, MatVec backend)
			throws Exception {
		LoraAdapterSet adapters = LoraInitializer.create(cfg, LoraProjection.qv(), RANK, ALPHA, new Random(SEED));
		LoraTrainableHandler h = LoraTrainableHandler.load(modelPath, shard, adapters, backend);
		try {
			adapters.zeroAllGrads();
			long t0 = System.nanoTime();
			var gr = h.computeGradients(tokens);
			long e2e = System.nanoTime() - t0;
			long backNs = gr.backwardMs() * 1_000_000L;
			return new long[] { e2e, Math.max(1L, backNs) };
		} finally {
			h.releaseGpuResources();
		}
	}

	private static void assertClose(float[] a, float[] b, String label) {
		assertThat(b).as(label).hasSameSizeAs(a);
		float maxAbs = 0f;
		for (float v : a)
			maxAbs = Math.max(maxAbs, Math.abs(v));
		float tol = Math.max(GRAD_TOL, GRAD_TOL * maxAbs);
		for (int i = 0; i < a.length; i++)
			assertThat(b[i]).as("%s[%d]", label, i).isCloseTo(a[i], within(tol));
	}

	private static int[] randomTokens(int n, int vocab, long seed) {
		Random rng = new Random(seed);
		int[] t = new int[n];
		for (int i = 0; i < n; i++)
			t[i] = 1 + rng.nextInt(Math.max(1, vocab - 1));
		return t;
	}
}
