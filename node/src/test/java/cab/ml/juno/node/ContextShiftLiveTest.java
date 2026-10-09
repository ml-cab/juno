package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Path;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/**
 * Context shift on every dense handler family with a real model on disk, CPU
 * backend: the shifted request's next logits match the oracle KV
 * ({@link ContextShiftOracle}) to float rounding, and the greedy token agrees.
 * Each case skips when its model file is absent.
 */
@DisplayName("Context shift - real models, CPU")
class ContextShiftLiveTest {

	private static final int N = 32;
	private static final int KEEP = 4;
	private static final int DISCARD = 12;
	/** Absolute logit tolerance: only the re-rotation's float rounding separates the two requests. */
	private static final double TOLERANCE = 0.01;

	static Path model(String file) {
		Path p = Path.of("models", file);
		return p.toFile().exists() ? p : Path.of("..", "models", file);
	}

	private static ShardContext shard(Path path) throws Exception {
		try (GgufReader r = GgufReader.open(path)) {
			LlamaConfig cfg = LlamaConfig.from(r);
			return new ShardContext("n0", 0, cfg.numLayers(), true, true, cfg.vocabSize(), cfg.hiddenDim(),
					cfg.numHeads());
		}
	}

	private static void assertMatches(ContextShiftLiveCheck.Result r) {
		assertThat(r.sameGreedyToken()).as("greedy next token").isTrue();
		assertThat(r.shiftDiff()).as("max |logit diff| against the oracle").isLessThanOrEqualTo(TOLERANCE);
	}

	private static LlamaConfig llamaConfig(Path path) throws Exception {
		try (GgufReader r = GgufReader.open(path)) {
			return LlamaConfig.from(r);
		}
	}

	/** Shift a LLaMA-family handler and score it with the standard rotation as the oracle. */
	private static void assertLlamaFamily(ForwardPassHandler h, Path path) throws Exception {
		LlamaConfig cfg = llamaConfig(path);
		RopePairing pairing = LlamaTransformerHandler.ropePairingFor(cfg);
		int kvHeads = cfg.numKvHeads();
		int hd = cfg.headDim();
		float theta = cfg.ropeTheta();
		assertMatches(ContextShiftLiveCheck.run(h, shard(path), cfg.kvDim(), N, KEEP, DISCARD,
				(row, pos) -> LoraTrainingMath.ropeBackward(row, pos, kvHeads, hd, theta, pairing),
				(row, pos) -> LlamaTransformerHandler.rope(row, pos, kvHeads, hd, theta, pairing)));
	}

	private static void llamaFamily(String file) throws Exception {
		Path path = model(file);
		assumeTrue(path.toFile().exists(), file + " not present");
		assertLlamaFamily(LlamaTransformerHandler.load(path, shard(path), CpuMatVec.INSTANCE), path);
	}

	/** Shift a Phi-3 handler; the oracle undoes the attention factor the adjoint applies on top of the key's own. */
	private static void assertPhi3(ForwardPassHandler h, Path path) throws Exception {
		LlamaConfig cfg;
		Phi3RopeConfig rope;
		try (GgufReader r = GgufReader.open(path)) {
			cfg = LlamaConfig.from(r);
			rope = Phi3RopeConfig.from(r, cfg);
		}
		int kvHeads = cfg.numKvHeads();
		int hd = cfg.headDim();
		float scale2 = rope.attnFactor() * rope.attnFactor();
		assertThat(h.contextLimit()).as("Phi-3.5 shifts at its 4096-token cap").isEqualTo(4096);
		assertMatches(ContextShiftLiveCheck.run(h, shard(path), cfg.kvDim(), N, KEEP, DISCARD, (row, pos) -> {
			Phi3Rope.ropeExtBackward(row, pos, kvHeads, hd, rope);
			for (int i = 0; i < row.length; i++)
				row[i] /= scale2;
		}, (row, pos) -> Phi3Rope.ropeExt(row, pos, kvHeads, hd, rope)));
	}

	private static Qwen3Config qwen3Config(Path path) throws Exception {
		try (GgufReader r = GgufReader.open(path)) {
			return Qwen3Config.from(r);
		}
	}

	private static void assertQwen3(ForwardPassHandler h, Path path) throws Exception {
		Qwen3Config cfg = qwen3Config(path);
		Qwen3RopeConfig rope = cfg.rope();
		assumeTrue(!rope.yarn(), "oracle below covers the unscaled rotation");
		int kvHeads = cfg.numKvHeads();
		int hd = cfg.headDim();
		assertMatches(ContextShiftLiveCheck.run(h, shard(path), cfg.kvDim(), N, KEEP, DISCARD,
				(row, pos) -> LoraTrainingMath.ropeBackward(row, pos, kvHeads, hd, rope.freqBase(), rope.pairing()),
				(row, pos) -> Qwen3Rope.apply(row, pos, kvHeads, hd, rope)));
	}

	/** Adapters with every B matrix filled, so playback actually changes K and V. */
	private static cab.ml.juno.lora.LoraAdapterSet randomAdapters(LoraModelLayout layout) {
		java.util.Random rnd = new java.util.Random(3);
		cab.ml.juno.lora.LoraAdapterSet set = LoraInitializer.create(layout, LoraProjection.qv(),
				cab.ml.juno.lora.LoraAdapterConfig.legacy(4, 8f), rnd);
		for (cab.ml.juno.lora.LoraAdapter a : set.all()) {
			float[] b = a.b();
			for (int i = 0; i < b.length; i++)
				b[i] = (float) rnd.nextGaussian() * 0.02f;
		}
		return set;
	}

	@Test
	@DisplayName("TinyLlama (LLaMA handler, adjacent pairs)")
	void tinyllama() throws Exception {
		llamaFamily("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
	}

	@Test
	@DisplayName("Qwen2.5-3B (LLaMA handler, split-half pairs)")
	void qwen25() throws Exception {
		llamaFamily("qwen2.5-3b-instruct-q4_k_m.gguf");
	}

	@Test
	@DisplayName("Phi-3.5-mini (Phi-3 handler, LongRoPE short factors); context limit 4096")
	void phi35() throws Exception {
		Path path = model("Phi-3.5-mini-instruct-Q4_K_M.gguf");
		assumeTrue(path.toFile().exists(), "Phi-3.5-mini not present");
		assertPhi3(Phi3TransformerHandler.load(path, shard(path), CpuMatVec.INSTANCE), path);
	}

	@Test
	@DisplayName("Qwen3-1.7B (Qwen3 handler)")
	void qwen3() throws Exception {
		Path path = model("Qwen3-1.7B-Q4_K_M.gguf");
		assumeTrue(path.toFile().exists(), "Qwen3-1.7B not present");
		assertQwen3(Qwen3TransformerHandler.load(path, shard(path), CpuMatVec.INSTANCE), path);
	}

	@Test
	@DisplayName("Phi-2 handler, partial rotation (moondream2's Phi-2 text backbone)")
	void phi2() throws Exception {
		// phi-2.Q4_K_M.gguf stores separate Q/K/V tensors, which this handler does not load; the
		// llamafile's backbone has the fused QKV it expects.
		Path path = model("moondream2-q5_k.llamafile");
		assumeTrue(path.toFile().exists(), "moondream2 not present");
		LlamaConfig cfg;
		int ropeDim;
		try (GgufReader r = GgufReader.open(path)) {
			cfg = LlamaConfig.from(r);
			ropeDim = r.metaInt("phi2.rope.dimension_count", cfg.headDim());
		}
		int kvHeads = cfg.numKvHeads();
		int hd = cfg.headDim();
		float theta = cfg.ropeTheta();
		ShardContext shard = shard(path);
		ForwardPassHandler h = Phi2TransformerHandler.load(path, shard, CpuMatVec.INSTANCE);
		assertMatches(ContextShiftLiveCheck.run(h, shard, cfg.kvDim(), N, KEEP, DISCARD,
				(row, pos) -> Phi2Rope.ropePartial(row, -pos, kvHeads, hd, ropeDim, theta),
				(row, pos) -> Phi2Rope.ropePartial(row, pos, kvHeads, hd, ropeDim, theta)));
	}

	/**
	 * Opt-in: the model is 18.6 GB and its CPU load does not fit the default test heap. Run with
	 * {@code -Djuno.test.largeModels=true} and a heap of about 40 GB.
	 */
	@Test
	@EnabledIfSystemProperty(named = "juno.test.largeModels", matches = "true")
	@DisplayName("Qwen3-Coder-30B-A3B (Qwen3-MoE handler)")
	void qwen3Moe() throws Exception {
		Path path = model("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf");
		assumeTrue(path.toFile().exists(), "Qwen3-Coder-30B-A3B not present");
		assertQwen3(Qwen3MoeTransformerHandler.load(path, shard(path), CpuMatVec.INSTANCE), path);
	}

	@Test
	@DisplayName("LoRA playback, LLaMA family: TinyLlama with its trained adapter")
	void loraTinyllama() throws Exception {
		Path path = model("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
		Path adapter = model("tinyllama-1.1b-chat-v1.0.Q4_K_M.lora");
		assumeTrue(path.toFile().exists() && adapter.toFile().exists(), "TinyLlama or its adapter not present");
		assertLlamaFamily(LoraTrainingHandlerFactory.create(path, shard(path),
				cab.ml.juno.lora.LoraAdapterSet.load(adapter), CpuMatVec.INSTANCE), path);
	}

	@Test
	@DisplayName("LoRA playback, Qwen2 family: Qwen2.5-3B (delegating handler)")
	void loraQwen25() throws Exception {
		Path path = model("qwen2.5-3b-instruct-q4_k_m.gguf");
		assumeTrue(path.toFile().exists(), "Qwen2.5-3B not present");
		assertLlamaFamily(LoraTrainingHandlerFactory.create(path, shard(path),
				randomAdapters(LoraModelLayout.qwen2(llamaConfig(path))), CpuMatVec.INSTANCE), path);
	}

	@Test
	@DisplayName("LoRA playback, Phi-3: Phi-3.5-mini")
	void loraPhi35() throws Exception {
		Path path = model("Phi-3.5-mini-instruct-Q4_K_M.gguf");
		assumeTrue(path.toFile().exists(), "Phi-3.5-mini not present");
		assertPhi3(LoraTrainingHandlerFactory.create(path, shard(path),
				randomAdapters(LoraModelLayout.phi3(llamaConfig(path))), CpuMatVec.INSTANCE), path);
	}

	@Test
	@DisplayName("LoRA playback, Qwen3: Qwen3-1.7B")
	void loraQwen3() throws Exception {
		Path path = model("Qwen3-1.7B-Q4_K_M.gguf");
		assumeTrue(path.toFile().exists(), "Qwen3-1.7B not present");
		assertQwen3(LoraTrainingHandlerFactory.create(path, shard(path),
				randomAdapters(LoraModelLayout.qwen3(qwen3Config(path))), CpuMatVec.INSTANCE), path);
	}
}
