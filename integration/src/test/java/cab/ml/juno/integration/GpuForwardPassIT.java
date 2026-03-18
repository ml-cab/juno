package cab.ml.juno.integration;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Path;
import java.nio.file.Paths;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import cab.ml.juno.node.CpuForwardPassHandler;
import cab.ml.juno.node.CublasMatVec;
import cab.ml.juno.node.CudaAvailability;
import cab.ml.juno.node.ForwardPassHandler;
import cab.ml.juno.node.ForwardRequest;
import cab.ml.juno.node.ForwardResult;
import cab.ml.juno.node.GpuContext;
import cab.ml.juno.node.GpuForwardPassHandler;
import cab.ml.juno.node.ShardContext;

/**
 * GPU forward pass integration test — requires CUDA 12.x + a GGUF model file.
 *
 * This is the AWS smoke test. It loads a real model (TinyLlama or any GGUF),
 * runs the same input through both CpuForwardPassHandler and
 * GpuForwardPassHandler, and asserts numerical equivalence within float32
 * rounding tolerance.
 *
 * Prerequisites on the AWS node: 1. CUDA 12.x installed (nvidia-smi shows the
 * GPU) 2. A GGUF model available at $MODEL_PATH or passed via -Dmodel.path=...
 * 3. mvn test -Dgroups=gpu -Dit.model.path=/path/to/model.gguf -pl integration
 *
 * Recommended AWS instance: g4dn.xlarge (T4, 16 GB VRAM), ~$0.50/hr on-demand.
 * Expected result for TinyLlama-1.1B Q4_K_M: - GPU output matches CPU output
 * within 1e-3 - GPU forward pass is 5–20x faster than CPU for full 22-layer
 * forward
 */
@Tag("gpu")
@DisplayName("GpuForwardPassHandler — end-to-end integration (requires CUDA + model file)")
class GpuForwardPassIT {

	private static final float DELTA = 1e-3f; // float32 rounding across backends

	private static GpuContext gpuCtx;
	private static Path modelPath;

	@BeforeAll
	static void setup() {
		// Guard first — before any JCuda class is touched.
		// Without -Djuno.gpu.test=true, JCuda native libs are never loaded into
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

		gpuCtx = GpuContext.init(0);
		System.out.println("GPU integration test running on: " + CudaAvailability.deviceName(0) + " ("
				+ String.format("%.1f", CudaAvailability.vramBytes(0) / 1e9) + " GB VRAM)");
	}

	@AfterAll
	static void teardown() {
		if (gpuCtx != null)
			gpuCtx.close();
	}

	// ── Helpers ───────────────────────────────────────────────────────────────

	/** First-node shard: embeddings only, no output projection. */
	@SuppressWarnings("unused")
	private static ShardContext firstNodeCtx(int totalLayers, int vocabSize, int hiddenDim) {
		int mid = totalLayers / 2;
		return new ShardContext("gpu-it-n0", 0, mid, true, false, vocabSize, hiddenDim, 32);
	}

	/** Last-node shard: no embeddings, output projection. */
	@SuppressWarnings("unused")
	private static ShardContext lastNodeCtx(int totalLayers, int vocabSize, int hiddenDim) {
		int mid = totalLayers / 2;
		return new ShardContext("gpu-it-n1", mid, totalLayers, false, true, vocabSize, hiddenDim, 32);
	}

	// ── Tests ─────────────────────────────────────────────────────────────────

	@Test
	@DisplayName("GPU activations match CPU activations — first node, single token")
	void first_node_gpu_matches_cpu() throws Exception {
		ShardContext ctx = new ShardContext("gpu-it", 0, 11, true, false, 32000, 2048, 32);

		ForwardPassHandler cpu = CpuForwardPassHandler.load(modelPath, ctx);
		ForwardPassHandler gpu = GpuForwardPassHandler.load(modelPath, ctx, new CublasMatVec(gpuCtx));

		ForwardRequest req = ForwardRequest.withTokens("it-req-1", new int[] { 1 }, 0);

		ForwardResult cpuResult = cpu.forward(req, ctx);
		ForwardResult gpuResult = gpu.forward(req, ctx);

		assertThat(gpuResult.activations()).hasSize(cpuResult.activations().length);
		float[] cpuAct = cpuResult.activations();
		float[] gpuAct = gpuResult.activations();
		for (int i = 0; i < cpuAct.length; i++)
			assertThat(gpuAct[i]).as("activation[%d]", i).isCloseTo(cpuAct[i], within(DELTA));
	}

	@Test
	@DisplayName("GPU logits match CPU logits — last node, vocabSize output")
	void last_node_gpu_logits_match_cpu() throws Exception {
		ShardContext ctx = new ShardContext("gpu-it", 11, 22, false, true, 32000, 2048, 32);
		float[] fakeActivations = new float[2048];
		for (int i = 0; i < fakeActivations.length; i++)
			fakeActivations[i] = (float) Math.sin(i * 0.01);

		ForwardPassHandler cpu = CpuForwardPassHandler.load(modelPath, ctx);
		ForwardPassHandler gpu = GpuForwardPassHandler.load(modelPath, ctx, new CublasMatVec(gpuCtx));

		ForwardRequest req = ForwardRequest.withActivations("it-req-2", fakeActivations, 0);

		ForwardResult cpuResult = cpu.forward(req, ctx);
		ForwardResult gpuResult = gpu.forward(req, ctx);

		assertThat(gpuResult.logits()).hasSize(32000);
		float[] cpuLogits = cpuResult.logits();
		float[] gpuLogits = gpuResult.logits();
		for (int i = 0; i < cpuLogits.length; i++)
			assertThat(gpuLogits[i]).as("logit[%d]", i).isCloseTo(cpuLogits[i], within(DELTA));
	}

	@Test
	@DisplayName("GPU is faster than CPU for full forward pass (timing sanity)")
	void gpu_forward_is_faster_than_cpu() throws Exception {
		ShardContext ctx = new ShardContext("gpu-it", 0, 11, true, false, 32000, 2048, 32);

		ForwardPassHandler cpu = CpuForwardPassHandler.load(modelPath, ctx);
		ForwardPassHandler gpu = GpuForwardPassHandler.load(modelPath, ctx, new CublasMatVec(gpuCtx));

		ForwardRequest req = ForwardRequest.withTokens("it-perf", new int[] { 42 }, 0);

		// Warm up
		cpu.forward(req, ctx);
		gpu.forward(req, ctx);

		// CPU timing
		long cpuStart = System.nanoTime();
		for (int i = 0; i < 10; i++)
			cpu.forward(req, ctx);
		long cpuMs = (System.nanoTime() - cpuStart) / 1_000_000;

		// GPU timing
		long gpuStart = System.nanoTime();
		for (int i = 0; i < 10; i++)
			gpu.forward(req, ctx);
		long gpuMs = (System.nanoTime() - gpuStart) / 1_000_000;

		System.out.printf("Forward pass 10 runs — CPU: %dms  GPU: %dms  speedup: %.1fx%n", cpuMs, gpuMs,
				(double) cpuMs / gpuMs);

		assertThat(gpuMs).isLessThan(cpuMs);
	}

	@Test
	@DisplayName("isReady() true after load on GPU node")
	void is_ready_after_load() throws Exception {
		ShardContext ctx = new ShardContext("gpu-it", 0, 11, true, false, 32000, 2048, 32);
		ForwardPassHandler gpu = GpuForwardPassHandler.load(modelPath, ctx, new CublasMatVec(gpuCtx));
		assertThat(gpu.isReady()).isTrue();
	}
}