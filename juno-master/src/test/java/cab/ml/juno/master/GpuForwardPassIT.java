package cab.ml.juno.master;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Locale;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import cab.ml.juno.node.CudaAvailability;
import cab.ml.juno.node.CudaMatVec;
import cab.ml.juno.node.ForwardPassHandler;
import cab.ml.juno.node.ForwardRequest;
import cab.ml.juno.node.GgufReader;
import cab.ml.juno.node.GpuContext;
import cab.ml.juno.node.LlamaConfig;
import cab.ml.juno.node.LlamaTransformerHandler;
import cab.ml.juno.node.ShardContext;
import cab.ml.juno.tokenizer.GgufTokenizer;

/**
 * The GPU forward pass against the CPU forward pass on a real model: the CPU
 * path is the correctness oracle, and this is the test that holds the GPU path
 * to it.
 *
 * <p>The two paths are not expected to agree bit for bit. The GPU keeps K-quant
 * weights packed and quantizes each projection's input to 8 bits (Q8_1) before
 * an integer dot product, keeps other weights in FP16, and attends over an FP16
 * copy of the KV cache; the CPU dequantizes and computes in FP32. So agreement is
 * judged by measures that mean the same at any magnitude, not by a fixed
 * absolute tolerance per element (a few hidden-state dimensions of a Llama model
 * sit above 100, where a 0.2% difference is already 0.3):
 * <ul>
 * <li>hidden state: cosine similarity, and relative L2 error
 * {@code ||gpu - cpu|| / ||cpu||};</li>
 * <li>logits: the same top-1 token, top-5 overlap, and relative L2 error;</li>
 * <li>generation: a greedy decode that must produce the same tokens.</li>
 * </ul>
 * Each bound was calibrated on the reference host: the measured value, and the
 * value with a deliberately planted fault, are recorded beside it.
 *
 * <p>Run: {@code mvn verify -pl juno-master -Pgpu -Dit.model.path=/abs/model.gguf}.
 * Requires CUDA and a Llama-architecture GGUF; skipped otherwise.
 */
@Tag("gpu")
@DisplayName("LlamaTransformerHandler GPU against CPU - end-to-end on a real model (requires CUDA + model file)")
class GpuForwardPassIT {

	private static final String PROMPT = "The capital of France is";
	private static final int GREEDY_STEPS = 16;

	// Bounds, calibrated on the reference host (GTX 1080). "Measured" is GPU against
	// CPU on tinyllama-1.1b Q4_K_M / mistral-7b Q4_K_M; "planted" is the same GPU
	// output deliberately corrupted, on tinyllama, to show the bound catches it.

	/**
	 * Hidden state, relative L2 error. Measured 0.00222 / 0.00036. Planted: a 1% gain
	 * error 0.00785, one attention head's slice zeroed 0.0127.
	 */
	private static final double HIDDEN_REL_L2_MAX = 0.005;

	/**
	 * Hidden state, cosine similarity. Measured 0.9999998 / 1.0000000. Planted: one head's
	 * slice zeroed 0.9999212. (A gain error leaves cosine unchanged; the L2 bound catches it.)
	 */
	private static final double HIDDEN_COSINE_MIN = 0.99995;

	/** Logits, relative L2 error. Measured 0.00904 / 0.00751. Planted: a 5% gain error 0.0515. */
	private static final double LOGITS_REL_L2_MAX = 0.025;

	/**
	 * Logits, how many of the CPU's top 5 tokens the GPU also ranks top 5. Measured 5 / 5; at
	 * least 4 leaves room for a near-tie at rank 5. (Swapping the top two logits keeps 5 of
	 * 5 and moves relative L2 only from 0.00904 to 0.00912; the top-1 check catches it.)
	 */
	private static final int LOGITS_TOP5_OVERLAP_MIN = 4;

	private static GpuContext gpuCtx;
	private static Path modelPath;
	private static LlamaConfig cfg;
	private static GgufTokenizer tokenizer;

	@BeforeAll
	static void setup() throws Exception {
		// Without -Djuno.gpu.test=true, CUDA native libs are never loaded into
		// the coordinator JVM, so no CUDA device FDs are inherited by the node
		// JVMs forked by ClusterHarness (which would crash them on startup).
		assumeTrue(Boolean.getBoolean("juno.gpu.test"),
				"Skipping GpuForwardPassIT — pass -Djuno.gpu.test=true to enable");
		assumeTrue(CudaAvailability.isAvailable(), "Skipping GpuForwardPassIT — no CUDA device available");

		String pathStr = System.getProperty("it.model.path", System.getenv("MODEL_PATH"));
		assumeTrue(pathStr != null && !pathStr.isBlank(),
				"Skipping GpuForwardPassIT — set -Dit.model.path or $MODEL_PATH");

		modelPath = Paths.get(pathStr);
		assumeTrue(modelPath.toFile().exists(), "Skipping GpuForwardPassIT — model file not found: " + modelPath);

		try (GgufReader r = GgufReader.open(modelPath)) {
			cfg = LlamaConfig.from(r);
			tokenizer = GgufTokenizer.load(r);
		}
		gpuCtx = GpuContext.init(0);
		System.out.println("GPU integration test running on: " + CudaAvailability.deviceName(0) + " ("
				+ String.format("%.1f", CudaAvailability.vramBytes(0) / 1e9) + " GB VRAM), model " + modelPath.getFileName());
	}

	@AfterAll
	static void teardown() {
		if (gpuCtx != null)
			gpuCtx.close();
	}

	// ── Tests ─────────────────────────────────────────────────────────────────

	@Test
	@DisplayName("hidden state after the first half of the layers agrees (cosine, relative L2)")
	void first_node_hidden_state_agrees() throws Exception {
		ShardContext ctx = firstHalf();
		int token = tokenizer.encode(PROMPT)[0];
		float[] cpuAct;
		float[] gpuAct;
		ForwardPassHandler cpu = LlamaTransformerHandler.load(modelPath, ctx);
		try {
			cpuAct = cpu.forward(ForwardRequest.withTokens("it-h", new int[] { token }, 0), ctx).activations();
		} finally {
			cpu.releaseGpuResources();
		}
		ForwardPassHandler gpu = LlamaTransformerHandler.load(modelPath, ctx, new CudaMatVec(gpuCtx));
		try {
			gpuAct = gpu.forward(ForwardRequest.withTokens("it-h", new int[] { token }, 0), ctx).activations();
		} finally {
			gpu.releaseGpuResources();
		}
		assertThat(gpuAct).hasSize(cpuAct.length);
		double cos = cosine(gpuAct, cpuAct);
		double rel = relativeL2(gpuAct, cpuAct);
		System.out.printf(Locale.ROOT, "GPU-IT hidden: cosine=%.7f relL2=%.6f maxAbs=%.4f%n", cos, rel,
				maxAbsDiff(gpuAct, cpuAct));
		assertThat(rel).as("hidden state, relative L2 error GPU vs CPU").isLessThanOrEqualTo(HIDDEN_REL_L2_MAX);
		assertThat(cos).as("hidden state, cosine similarity GPU vs CPU").isGreaterThanOrEqualTo(HIDDEN_COSINE_MIN);
	}

	@Test
	@DisplayName("logits of the second half of the layers agree (top-1, top-5 overlap, relative L2)")
	void last_node_logits_agree() throws Exception {
		ShardContext ctx = secondHalf();
		float[] input = new float[cfg.hiddenDim()];
		for (int i = 0; i < input.length; i++)
			input[i] = (float) Math.sin(i * 0.01);
		float[] cpuLogits;
		float[] gpuLogits;
		ForwardPassHandler cpu = LlamaTransformerHandler.load(modelPath, ctx);
		try {
			cpuLogits = cpu.forward(ForwardRequest.withActivations("it-l", input, 0), ctx).logits();
		} finally {
			cpu.releaseGpuResources();
		}
		ForwardPassHandler gpu = LlamaTransformerHandler.load(modelPath, ctx, new CudaMatVec(gpuCtx));
		try {
			gpuLogits = gpu.forward(ForwardRequest.withActivations("it-l", input, 0), ctx).logits();
		} finally {
			gpu.releaseGpuResources();
		}
		assertThat(gpuLogits).hasSize(cfg.vocabSize());
		double rel = relativeL2(gpuLogits, cpuLogits);
		int overlap = topKOverlap(gpuLogits, cpuLogits, 5);
		System.out.printf(Locale.ROOT, "GPU-IT logits: top1 cpu=%d gpu=%d top5overlap=%d relL2=%.6f maxAbs=%.4f%n",
				argmax(cpuLogits), argmax(gpuLogits), overlap, rel, maxAbsDiff(gpuLogits, cpuLogits));
		assertThat(argmax(gpuLogits)).as("top-1 token GPU vs CPU").isEqualTo(argmax(cpuLogits));
		assertThat(overlap).as("top-5 overlap GPU vs CPU").isGreaterThanOrEqualTo(LOGITS_TOP5_OVERLAP_MIN);
		assertThat(rel).as("logits, relative L2 error GPU vs CPU").isLessThanOrEqualTo(LOGITS_REL_L2_MAX);
	}

	@Test
	@DisplayName("greedy decode over the whole model produces the same tokens on GPU and CPU")
	void greedy_decode_agrees() throws Exception {
		ShardContext ctx = wholeModel();
		int[] prompt = tokenizer.encode(PROMPT);
		int[] cpuTokens;
		int[] gpuTokens;
		ForwardPassHandler cpu = LlamaTransformerHandler.load(modelPath, ctx);
		try {
			cpuTokens = greedy(cpu, ctx, prompt, "it-g-cpu");
		} finally {
			cpu.releaseGpuResources();
		}
		ForwardPassHandler gpu = LlamaTransformerHandler.load(modelPath, ctx, new CudaMatVec(gpuCtx));
		try {
			gpuTokens = greedy(gpu, ctx, prompt, "it-g-gpu");
		} finally {
			gpu.releaseGpuResources();
		}
		System.out.println("GPU-IT greedy cpu=" + Arrays.toString(cpuTokens) + " gpu=" + Arrays.toString(gpuTokens));
		// Measured: identical over all 16 steps on tinyllama and mistral-7b.
		assertThat(gpuTokens).as("greedy tokens GPU vs CPU after \"" + PROMPT + "\"").containsExactly(cpuTokens);
	}

	@Test
	@DisplayName("GPU is faster than CPU for the first half of the layers (timing sanity)")
	void gpu_forward_is_faster_than_cpu() throws Exception {
		ShardContext ctx = firstHalf();
		ForwardRequest req = ForwardRequest.withTokens("it-perf", new int[] { tokenizer.encode(PROMPT)[0] }, 0);
		long cpuMs;
		long gpuMs;
		ForwardPassHandler cpu = LlamaTransformerHandler.load(modelPath, ctx);
		try {
			cpu.forward(req, ctx);
			long start = System.nanoTime();
			for (int i = 0; i < 10; i++)
				cpu.forward(req, ctx);
			cpuMs = (System.nanoTime() - start) / 1_000_000;
		} finally {
			cpu.releaseGpuResources();
		}
		ForwardPassHandler gpu = LlamaTransformerHandler.load(modelPath, ctx, new CudaMatVec(gpuCtx));
		try {
			gpu.forward(req, ctx);
			long start = System.nanoTime();
			for (int i = 0; i < 10; i++)
				gpu.forward(req, ctx);
			gpuMs = (System.nanoTime() - start) / 1_000_000;
		} finally {
			gpu.releaseGpuResources();
		}
		System.out.printf("Forward pass 10 runs — CPU: %dms  GPU: %dms  speedup: %.1fx%n", cpuMs, gpuMs,
				(double) cpuMs / gpuMs);
		assertThat(gpuMs).isLessThan(cpuMs);
	}

	@Test
	@DisplayName("isReady() true after load on GPU node")
	void is_ready_after_load() throws Exception {
		ForwardPassHandler gpu = LlamaTransformerHandler.load(modelPath, firstHalf(), new CudaMatVec(gpuCtx));
		try {
			assertThat(gpu.isReady()).isTrue();
		} finally {
			gpu.releaseGpuResources();
		}
	}

	// ── Helpers ───────────────────────────────────────────────────────────────

	private static ShardContext firstHalf() {
		return new ShardContext("gpu-it", 0, cfg.numLayers() / 2, true, false, cfg.vocabSize(), cfg.hiddenDim(),
				cfg.numHeads());
	}

	private static ShardContext secondHalf() {
		return new ShardContext("gpu-it", cfg.numLayers() / 2, cfg.numLayers(), false, true, cfg.vocabSize(),
				cfg.hiddenDim(), cfg.numHeads());
	}

	private static ShardContext wholeModel() {
		return new ShardContext("gpu-it", 0, cfg.numLayers(), true, true, cfg.vocabSize(), cfg.hiddenDim(),
				cfg.numHeads());
	}

	/** Feeds the prompt one position at a time, then decodes {@link #GREEDY_STEPS} tokens by argmax. */
	private static int[] greedy(ForwardPassHandler h, ShardContext ctx, int[] prompt, String kv) {
		int[] out = new int[GREEDY_STEPS];
		int[] one = new int[1];
		float[] logits = null;
		int pos = 0;
		for (int t : prompt) {
			one[0] = t;
			logits = h.forward(ForwardRequest.withTokens(kv, one, pos++), ctx).logits();
		}
		for (int s = 0; s < GREEDY_STEPS; s++) {
			out[s] = argmax(logits);
			one[0] = out[s];
			logits = h.forward(ForwardRequest.withTokens(kv, one, pos++), ctx).logits();
		}
		h.evict(kv);
		return out;
	}

	static double cosine(float[] a, float[] b) {
		double dot = 0, na = 0, nb = 0;
		for (int i = 0; i < a.length; i++) {
			dot += (double) a[i] * b[i];
			na += (double) a[i] * a[i];
			nb += (double) b[i] * b[i];
		}
		return dot / Math.sqrt(na * nb);
	}

	/** {@code ||actual - expected|| / ||expected||}. */
	static double relativeL2(float[] actual, float[] expected) {
		double diff = 0, norm = 0;
		for (int i = 0; i < expected.length; i++) {
			double d = (double) actual[i] - expected[i];
			diff += d * d;
			norm += (double) expected[i] * expected[i];
		}
		return Math.sqrt(diff / norm);
	}

	static double maxAbsDiff(float[] a, float[] b) {
		double m = 0;
		for (int i = 0; i < a.length; i++)
			m = Math.max(m, Math.abs(a[i] - b[i]));
		return m;
	}

	static int argmax(float[] a) {
		int best = 0;
		for (int i = 1; i < a.length; i++)
			if (a[i] > a[best])
				best = i;
		return best;
	}

	/** How many of {@code expected}'s top-k indices are also in {@code actual}'s top-k. */
	static int topKOverlap(float[] actual, float[] expected, int k) {
		int[] ta = topK(actual, k);
		int[] te = topK(expected, k);
		int n = 0;
		for (int e : te)
			for (int a : ta)
				if (a == e)
					n++;
		return n;
	}

	private static int[] topK(float[] a, int k) {
		int[] idx = new int[k];
		Arrays.fill(idx, -1);
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < k; j++) {
				if (idx[j] < 0 || a[i] > a[idx[j]]) {
					System.arraycopy(idx, j, idx, j + 1, k - j - 1);
					idx[j] = i;
					break;
				}
			}
		}
		return idx;
	}
}
