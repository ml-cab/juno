package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Path;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import cab.ml.juno.lora.LoraAdapterSet;

/**
 * Sliding-window attention on every handler family with a real model on disk,
 * CPU backend, under a forced patterned window (no sweep model declares one that
 * binds): the KV oracle of {@link SlidingWindowLiveCheck} on each inference
 * handler, and for LoRA training, the training forward's loss against the
 * inference forward's loss under the same window. Each case skips when its model
 * file is absent.
 */
@DisplayName("Sliding-window attention - real models, CPU")
class SlidingWindowLiveTest {

	private static final int N = 24;
	private static final int WIDTH = 6;
	/** A global layer's overwritten rows (values around 50) move the logits by far more than this. */
	private static final double MOVED = 1e-2;

	private static ShardContext shard(Path path) throws Exception {
		try (GgufReader r = GgufReader.open(path)) {
			LlamaConfig cfg = LlamaConfig.from(r);
			return new ShardContext("n0", 0, cfg.numLayers(), true, true, cfg.vocabSize(), cfg.hiddenDim(),
					cfg.numHeads());
		}
	}

	private static LlamaConfig config(Path path) throws Exception {
		try (GgufReader r = GgufReader.open(path)) {
			return LlamaConfig.from(r);
		}
	}

	private static Path present(String file) {
		Path path = SlidingWindowLiveCheck.model(file);
		assumeTrue(path.toFile().exists(), file + " not present");
		return path;
	}

	private static void assertWindowed(SlidingWindowLiveCheck.Loader loader, Path path) throws Exception {
		LlamaConfig cfg = config(path);
		SlidingWindowLiveCheck.Result r = SlidingWindowLiveCheck.run(loader, shard(path), cfg.numLayers(), cfg.kvDim(), N,
				WIDTH);
		assertThat(r.windowedDiff()).as("rows outside a windowed layer's window are not read").isZero();
		assertThat(r.globalDiff()).as("a global layer reads them").isGreaterThan(MOVED);
		assertThat(r.windowEffect()).as("the window changes the output of a prompt longer than it")
				.isGreaterThan(MOVED);
	}

	@Test
	@DisplayName("TinyLlama (LLaMA handler)")
	void tinyllama() throws Exception {
		Path path = present("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
		assertWindowed(() -> LlamaTransformerHandler.load(path, shard(path), CpuMatVec.INSTANCE), path);
	}

	@Test
	@DisplayName("Qwen2.5-3B (LLaMA handler, split-half pairs, Q/K/V biases)")
	void qwen25() throws Exception {
		Path path = present("qwen2.5-3b-instruct-q4_k_m.gguf");
		assertWindowed(() -> LlamaTransformerHandler.load(path, shard(path), CpuMatVec.INSTANCE), path);
	}

	@Test
	@DisplayName("Phi-3.5-mini (Phi-3 handler)")
	void phi35() throws Exception {
		Path path = present("Phi-3.5-mini-instruct-Q4_K_M.gguf");
		assertWindowed(() -> Phi3TransformerHandler.load(path, shard(path), CpuMatVec.INSTANCE), path);
	}

	@Test
	@DisplayName("Qwen3-1.7B (Qwen3 handler)")
	void qwen3() throws Exception {
		Path path = present("Qwen3-1.7B-Q4_K_M.gguf");
		assertWindowed(() -> Qwen3TransformerHandler.load(path, shard(path), CpuMatVec.INSTANCE), path);
	}

	@Test
	@DisplayName("Phi-2 handler (moondream2's Phi-2 text backbone)")
	void phi2() throws Exception {
		Path path = present("moondream2-q5_k.llamafile");
		assertWindowed(() -> Phi2TransformerHandler.load(path, shard(path), CpuMatVec.INSTANCE), path);
	}

	/** Opt-in, as in {@link ContextShiftLiveTest#qwen3Moe}: needs {@code -Djuno.test.largeModels=true} and about 40 GB of heap. */
	@Test
	@EnabledIfSystemProperty(named = "juno.test.largeModels", matches = "true")
	@DisplayName("Qwen3-Coder-30B-A3B (Qwen3-MoE handler)")
	void qwen3Moe() throws Exception {
		Path path = present("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf");
		assertWindowed(() -> Qwen3MoeTransformerHandler.load(path, shard(path), CpuMatVec.INSTANCE), path);
	}

	@Test
	@DisplayName("LoRA playback: TinyLlama with its trained adapter")
	void loraTinyllama() throws Exception {
		Path path = present("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
		Path adapter = present("tinyllama-1.1b-chat-v1.0.Q4_K_M.lora");
		assertWindowed(() -> LoraTrainingHandlerFactory.create(path, shard(path), LoraAdapterSet.load(adapter),
				CpuMatVec.INSTANCE), path);
	}

	@Test
	@DisplayName("LoRA playback: Qwen2.5-3B (delegating handler), Phi-3.5-mini, Qwen3-1.7B")
	void loraOtherFamilies() throws Exception {
		Path qwen25 = SlidingWindowLiveCheck.model("qwen2.5-3b-instruct-q4_k_m.gguf");
		Path phi = SlidingWindowLiveCheck.model("Phi-3.5-mini-instruct-Q4_K_M.gguf");
		Path qwen3 = SlidingWindowLiveCheck.model("Qwen3-1.7B-Q4_K_M.gguf");
		assumeTrue(qwen25.toFile().exists() || phi.toFile().exists() || qwen3.toFile().exists(), "no model present");
		if (qwen25.toFile().exists())
			assertWindowed(() -> LoraTrainingHandlerFactory.create(qwen25, shard(qwen25),
					emptyAdapters(LoraModelLayout.qwen2(config(qwen25))), CpuMatVec.INSTANCE), qwen25);
		if (phi.toFile().exists())
			assertWindowed(() -> LoraTrainingHandlerFactory.create(phi, shard(phi),
					emptyAdapters(LoraModelLayout.phi3(config(phi))), CpuMatVec.INSTANCE), phi);
		if (qwen3.toFile().exists())
			assertWindowed(() -> LoraTrainingHandlerFactory.create(qwen3, shard(qwen3),
					emptyAdapters(LoraModelLayout.qwen3(qwen3Config(qwen3))), CpuMatVec.INSTANCE), qwen3);
	}

	private static Qwen3Config qwen3Config(Path path) throws Exception {
		try (GgufReader r = GgufReader.open(path)) {
			return Qwen3Config.from(r);
		}
	}

	private static LoraAdapterSet emptyAdapters(LoraModelLayout layout) {
		return LoraInitializer.create(layout, LoraProjection.qv(), cab.ml.juno.lora.LoraAdapterConfig.legacy(4, 8f),
				new java.util.Random(3));
	}

	/**
	 * LoRA training: the training forward (which keeps each head's attention weights
	 * for the backward pass) gives the same loss as the inference forward under the
	 * same window, and a different one from no window. Rows outside the window keep
	 * weight 0 in what the backward pass reads, so they get no gradient.
	 */
	@Test
	@DisplayName("LoRA training forward honours the window: loss equals the windowed inference loss")
	void trainingForwardHonoursTheWindow() throws Exception {
		Path tiny = SlidingWindowLiveCheck.model("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
		Path phi = SlidingWindowLiveCheck.model("Phi-3.5-mini-instruct-Q4_K_M.gguf");
		Path qwen3 = SlidingWindowLiveCheck.model("Qwen3-1.7B-Q4_K_M.gguf");
		assumeTrue(tiny.toFile().exists() || phi.toFile().exists() || qwen3.toFile().exists(), "no model present");
		if (tiny.toFile().exists())
			assertTrainingLoss(tiny, () -> emptyAdapters(LoraModelLayout.llama(config(tiny))));
		if (phi.toFile().exists())
			assertTrainingLoss(phi, () -> emptyAdapters(LoraModelLayout.phi3(config(phi))));
		if (qwen3.toFile().exists())
			assertTrainingLoss(qwen3, () -> emptyAdapters(LoraModelLayout.qwen3(qwen3Config(qwen3))));
	}

	interface Adapters {
		LoraAdapterSet get() throws Exception;
	}

	private static void assertTrainingLoss(Path path, Adapters adapters) throws Exception {
		LlamaConfig cfg = config(path);
		int[] tokens = new int[N + 1];
		for (int i = 0; i <= N; i++)
			tokens[i] = ContextShiftLiveCheck.token(i);
		LoraTrainingHandler windowed = (LoraTrainingHandler) SlidingWindowLiveCheck.loadUnder(
				SlidingWindowLiveCheck.periodTwo(WIDTH, cfg.numLayers()),
				() -> LoraTrainingHandlerFactory.create(path, shard(path), adapters.get(), CpuMatVec.INSTANCE));
		float train = windowed.computeGradients(tokens).lossSum();
		float infer = windowed.evaluateLoss(tokens).lossSum();
		LoraTrainingHandler none = (LoraTrainingHandler) SlidingWindowLiveCheck.loadUnder(SlidingWindow.NONE,
				() -> LoraTrainingHandlerFactory.create(path, shard(path), adapters.get(), CpuMatVec.INSTANCE));
		float unwindowed = none.evaluateLoss(tokens).lossSum();
		System.out.printf("%s training: windowed train loss %.5f, windowed inference loss %.5f, no window %.5f%n",
				path.getFileName(), train, infer, unwindowed);
		assertThat((double) Math.abs(train - infer)).as("training and inference forward agree under the window")
				.isLessThanOrEqualTo(1e-3 * Math.abs(infer));
		assertThat((double) Math.abs(infer - unwindowed)).as("the window changes the loss").isGreaterThan(1e-2);
	}
}
