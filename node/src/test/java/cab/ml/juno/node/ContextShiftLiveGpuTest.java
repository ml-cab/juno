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

/**
 * Context shift with the device KV mirror kept by {@link GpuAttentionMirror}
 * (Phi-3 and Qwen3 handlers on CUDA): every mirror in use is rewritten to the
 * shifted length rather than retired, and the next logits, attended on the
 * device, match the oracle KV attended on the host within the device-vs-host
 * difference the same handler shows without a shift.
 */
@Tag("gpu")
@DisplayName("Context shift - real models, CUDA device mirror")
class ContextShiftLiveGpuTest {

	private static final int N = 64;
	private static final int KEEP = 4;
	private static final int DISCARD = 24;

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

	private static ShardContext shard(Path path) throws Exception {
		try (GgufReader r = GgufReader.open(path)) {
			LlamaConfig cfg = LlamaConfig.from(r);
			return new ShardContext("n0", 0, cfg.numLayers(), true, true, cfg.vocabSize(), cfg.hiddenDim(),
					cfg.numHeads());
		}
	}

	private static void assertDeviceShift(ContextShiftLiveCheck.Result r) {
		assertThat(r.watermarks()).as("device mirrors exist").isNotNull();
		assertThat(Arrays.stream(r.watermarks()).min().getAsInt()).as("no mirror retired by the shift")
				.isGreaterThanOrEqualTo(0);
		assertThat(Arrays.stream(r.watermarks()).max().getAsInt()).as("mirrors hold the shifted history")
				.isEqualTo(N - DISCARD);
		assertThat(r.sameGreedyToken()).as("greedy next token").isTrue();
		assertThat(r.shiftDiff()).as("max |logit diff| against the oracle, control %.5f", r.controlDiff())
				.isLessThanOrEqualTo(2 * r.controlDiff() + 0.02);
	}

	@Test
	@DisplayName("Phi-3.5-mini on CUDA")
	void phi35() throws Exception {
		Path path = ContextShiftLiveTest.model("Phi-3.5-mini-instruct-Q4_K_M.gguf");
		assumeTrue(path.toFile().exists(), "Phi-3.5-mini not present");
		LlamaConfig cfg;
		Phi3RopeConfig rope;
		try (GgufReader r = GgufReader.open(path)) {
			cfg = LlamaConfig.from(r);
			rope = Phi3RopeConfig.from(r, cfg);
		}
		int kvHeads = cfg.numKvHeads();
		int hd = cfg.headDim();
		float scale2 = rope.attnFactor() * rope.attnFactor();
		CudaMatVec backend = new CudaMatVec(gpu);
		Phi3TransformerHandler h = Phi3TransformerHandler.load(path, shard(path), backend);
		try {
			assumeTrue(h.gpuAttentionActive(), "GPU attention not active");
			assertDeviceShift(ContextShiftLiveCheck.run(h, shard(path), cfg.kvDim(), N, KEEP, DISCARD, (row, pos) -> {
				Phi3Rope.ropeExtBackward(row, pos, kvHeads, hd, rope);
				for (int i = 0; i < row.length; i++)
					row[i] /= scale2;
			}, (row, pos) -> Phi3Rope.ropeExt(row, pos, kvHeads, hd, rope)));
		} finally {
			h.releaseGpuResources();
			backend.releaseScratch();
		}
	}

	@Test
	@DisplayName("Qwen3-1.7B on CUDA")
	void qwen3() throws Exception {
		Path path = ContextShiftLiveTest.model("Qwen3-1.7B-Q4_K_M.gguf");
		assumeTrue(path.toFile().exists(), "Qwen3-1.7B not present");
		Qwen3Config cfg;
		try (GgufReader r = GgufReader.open(path)) {
			cfg = Qwen3Config.from(r);
		}
		Qwen3RopeConfig rope = cfg.rope();
		int kvHeads = cfg.numKvHeads();
		int hd = cfg.headDim();
		CudaMatVec backend = new CudaMatVec(gpu);
		Qwen3TransformerHandler h = Qwen3TransformerHandler.load(path, shard(path), backend);
		try {
			assumeTrue(h.gpuAttentionActive(), "GPU attention not active");
			assertDeviceShift(ContextShiftLiveCheck.run(h, shard(path), cfg.kvDim(), N, KEEP, DISCARD,
					(row, pos) -> LoraTrainingMath.ropeBackward(row, pos, kvHeads, hd, rope.freqBase(), rope.pairing()),
					(row, pos) -> Qwen3Rope.apply(row, pos, kvHeads, hd, rope)));
		} finally {
			h.releaseGpuResources();
			backend.releaseScratch();
		}
	}
}
