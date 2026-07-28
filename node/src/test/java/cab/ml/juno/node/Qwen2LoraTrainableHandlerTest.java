package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import cab.ml.juno.lora.LoraAdapter;
import cab.ml.juno.lora.LoraAdapterConfig;
import cab.ml.juno.lora.LoraAdapterSet;
import cab.ml.juno.lora.LoraAdamOptimizer;

/**
 * Gate A — Qwen2 / Qwen2.5 dense LoRA training.
 */
@DisplayName("Qwen2LoraTrainableHandler — Gate A")
class Qwen2LoraTrainableHandlerTest {

	private static final int H = 64;
	private static final int HEADS = 4;
	private static final int KV_HEADS = 2;
	private static final int I = 128;
	private static final int VOCAB = 128;
	private static final int LAYERS = 1;

	@Test
	@DisplayName("zero-adapter logits match LlamaTransformerHandler on bias-bearing Qwen2")
	void zeroAdapter_matchesBase(@TempDir Path tmp) throws IOException {
		float[] qBias = filled(H, 0.01f);
		float[] kvBias = filled(KV_HEADS * (H / HEADS), -0.02f);
		Path gguf = buildSyntheticQwen2Gguf(tmp, qBias, kvBias);
		ShardContext ctx = new ShardContext("n0", 0, LAYERS, true, true, VOCAB, H, HEADS);

		LlamaTransformerHandler base = LlamaTransformerHandler.load(gguf, ctx, CpuMatVec.INSTANCE);
		LoraAdapterSet adapters = LoraInitializer.create(
				LlamaConfig.from(GgufReader.open(gguf)), LoraProjection.qv(), 2, 2f, new Random(1));
		// Zero all A/B so LoRA contributes nothing
		zeroAdapters(adapters);

		Qwen2LoraTrainableHandler lora = Qwen2LoraTrainableHandler.load(gguf, ctx, adapters, CpuMatVec.INSTANCE);

		ForwardRequest req = ForwardRequest.withTokens("r", new int[] { 3 }, 0);
		float[] baseLogits = base.forward(req, ctx).logits();
		float[] loraLogits = lora.forward(req, ctx).logits();
		assertThat(loraLogits).hasSameSizeAs(baseLogits);
		for (int i = 0; i < baseLogits.length; i++)
			assertThat(loraLogits[i]).isCloseTo(baseLogits[i], within(1e-4f));
	}

	@Test
	@DisplayName("factory routes qwen2 adapters to Qwen2LoraTrainableHandler")
	void factoryRoutesQwen2(@TempDir Path tmp) throws IOException {
		Path gguf = buildSyntheticQwen2Gguf(tmp, filled(H, 0f), filled(KV_HEADS * (H / HEADS), 0f));
		ShardContext ctx = new ShardContext("n0", 0, LAYERS, true, true, VOCAB, H, HEADS);
		LlamaConfig cfg;
		try (GgufReader r = GgufReader.open(gguf)) {
			cfg = LlamaConfig.from(r);
		}
		LoraAdapterSet adapters = LoraInitializer.create(cfg, LoraProjection.qv(), 2, 2f, new Random(2));
		ForwardPassHandler h = ForwardPassHandlerLoader.load(gguf, ctx, CpuMatVec.INSTANCE, adapters);
		assertThat(h).isInstanceOf(Qwen2LoraTrainableHandler.class);
	}

	@Test
	@DisplayName("tiny synthetic overfit decreases loss")
	void tinyOverfit_decreasesLoss(@TempDir Path tmp) throws IOException {
		Path gguf = buildSyntheticQwen2Gguf(tmp, filled(H, 0.001f), filled(KV_HEADS * (H / HEADS), 0.001f));
		ShardContext ctx = new ShardContext("n0", 0, LAYERS, true, true, VOCAB, H, HEADS);
		LlamaConfig cfg;
		try (GgufReader r = GgufReader.open(gguf)) {
			cfg = LlamaConfig.from(r);
		}
		LoraAdapterSet adapters = LoraInitializer.create(cfg, LoraProjection.allLinear(), 8, 16f, new Random(42));
		for (var e : adapters.asMap().entrySet()) {
			float[] b = e.getValue().b();
			for (int i = 0; i < Math.min(16, b.length); i++)
				b[i] = 0.1f;
		}
		Qwen2LoraTrainableHandler handler = Qwen2LoraTrainableHandler.load(gguf, ctx, adapters, CpuMatVec.INSTANCE);
		int[] tokens = new int[] { 1, 2, 3, 4, 5 };

		adapters.zeroAllGrads();
		handler.computeGradients(tokens);
		float gradNorm = 0f;
		for (var e : adapters.asMap().entrySet())
			for (float v : e.getValue().gradB())
				gradNorm += Math.abs(v);
		assertThat(gradNorm).as("adapter B gradients should be non-zero").isGreaterThan(1e-6f);

		float first = handler.evaluateLoss(tokens).meanLoss();
		// Manual gradient descent on B (avoids Adam hyperparameter sensitivity on tiny models)
		for (int step = 0; step < 50; step++) {
			adapters.zeroAllGrads();
			var gr = handler.computeGradients(tokens);
			adapters.prepareGradientsForOptimizer(gr.predictionCount(), 0f);
			for (var e : adapters.asMap().entrySet()) {
				float[] b = e.getValue().b();
				float[] g = e.getValue().gradB();
				for (int i = 0; i < b.length; i++)
					b[i] -= 0.5f * g[i];
			}
		}
		float last = handler.evaluateLoss(tokens).meanLoss();
		assertThat(last).as("loss %.4f -> %.4f", first, last).isLessThan(first);
	}

	@Test
	@DisplayName("finite-difference adapter gradient for wq")
	void finiteDifference_wq(@TempDir Path tmp) throws IOException {
		Path gguf = buildSyntheticQwen2Gguf(tmp, filled(H, 0f), filled(KV_HEADS * (H / HEADS), 0f));
		ShardContext ctx = new ShardContext("n0", 0, LAYERS, true, true, VOCAB, H, HEADS);
		LlamaConfig cfg;
		try (GgufReader r = GgufReader.open(gguf)) {
			cfg = LlamaConfig.from(r);
		}
		LoraAdapterSet adapters = LoraInitializer.create(cfg, LoraProjection.qv(), 2, 2f, new Random(7));
		Qwen2LoraTrainableHandler handler = Qwen2LoraTrainableHandler.load(gguf, ctx, adapters, CpuMatVec.INSTANCE);

		int[] tokens = new int[] { 1, 2, 3 };
		LoraAdapter a = adapters.get(0, "wq");
		assertThat(a).isNotNull();
		int idx = 0;
		float eps = 1e-3f;
		float orig = a.b()[idx];

		a.b()[idx] = orig + eps;
		float lossPlus = handler.evaluateLoss(tokens).meanLoss();
		a.b()[idx] = orig - eps;
		float lossMinus = handler.evaluateLoss(tokens).meanLoss();
		a.b()[idx] = orig;

		adapters.zeroAllGrads();
		handler.computeGradients(tokens);
		float analytic = a.gradB()[idx];
		float numeric = (lossPlus - lossMinus) / (2f * eps);
		assertThat(analytic).isCloseTo(numeric, within(5e-2f));
	}

	private static void zeroAdapters(LoraAdapterSet adapters) {
		for (var e : adapters.asMap().entrySet()) {
			LoraAdapter a = e.getValue();
			java.util.Arrays.fill(a.a(), 0f);
			java.util.Arrays.fill(a.b(), 0f);
		}
	}

	private static float[] filled(int n, float v) {
		float[] a = new float[n];
		java.util.Arrays.fill(a, v);
		return a;
	}

	private static Path buildSyntheticQwen2Gguf(Path dir, float[] qBias, float[] kvBias) throws IOException {
		Files.createDirectories(dir);
		int kvDim = KV_HEADS * (H / HEADS);
		Phi3TransformerHandlerTest.GgufAssembler gguf = new Phi3TransformerHandlerTest.GgufAssembler();

		gguf.addString("general.architecture", "qwen2");
		gguf.addUInt32("qwen2.embedding_length", H);
		gguf.addUInt32("qwen2.block_count", LAYERS);
		gguf.addUInt32("qwen2.attention.head_count", HEADS);
		gguf.addUInt32("qwen2.attention.head_count_kv", KV_HEADS);
		gguf.addUInt32("qwen2.vocab_size", VOCAB);
		gguf.addUInt32("qwen2.feed_forward_length", I);
		gguf.addFloat32("qwen2.attention.layer_norm_rms_epsilon", 1e-5f);
		gguf.addFloat32("qwen2.rope.freq_base", 10000.0f);

		gguf.addTensor("token_embd.weight", 0, new long[] { VOCAB, H }, varyingF32((long) VOCAB * H, 7));
		gguf.addTensor("output_norm.weight", 0, new long[] { H }, onesF32(H));
		gguf.addTensor("output.weight", 0, new long[] { VOCAB, H }, varyingF32((long) VOCAB * H, 11));

		for (int li = 0; li < LAYERS; li++) {
			String p = "blk." + li + ".";
			gguf.addTensor(p + "attn_norm.weight", 0, new long[] { H }, onesF32(H));
			gguf.addTensor(p + "ffn_norm.weight", 0, new long[] { H }, onesF32(H));
			gguf.addTensor(p + "attn_q.weight", 0, new long[] { H, H }, identityF32(H));
			gguf.addTensor(p + "attn_k.weight", 0, new long[] { kvDim, H }, smallF32((long) kvDim * H));
			gguf.addTensor(p + "attn_v.weight", 0, new long[] { kvDim, H }, smallF32((long) kvDim * H));
			gguf.addTensor(p + "attn_output.weight", 0, new long[] { H, H }, identityF32(H));
			gguf.addTensor(p + "ffn_gate.weight", 0, new long[] { I, H }, smallF32((long) I * H));
			gguf.addTensor(p + "ffn_up.weight", 0, new long[] { I, H }, smallF32((long) I * H));
			gguf.addTensor(p + "ffn_down.weight", 0, new long[] { H, I }, smallF32((long) H * I));
			gguf.addTensor(p + "attn_q.bias", 0, new long[] { H }, f32(qBias));
			gguf.addTensor(p + "attn_k.bias", 0, new long[] { kvDim }, f32(kvBias));
			gguf.addTensor(p + "attn_v.bias", 0, new long[] { kvDim }, f32(kvBias));
		}

		Path out = dir.resolve("synthetic_qwen2_lora.gguf");
		Files.write(out, gguf.build());
		return out;
	}

	private static byte[] zeroF32(long nelems) {
		return new byte[(int) (nelems * 4)];
	}

	private static byte[] onesF32(int n) {
		ByteBuffer bb = ByteBuffer.allocate(n * 4).order(ByteOrder.LITTLE_ENDIAN);
		for (int i = 0; i < n; i++)
			bb.putFloat(1f);
		return bb.array();
	}

	private static byte[] identityF32(int n) {
		ByteBuffer bb = ByteBuffer.allocate(n * n * 4).order(ByteOrder.LITTLE_ENDIAN);
		for (int r = 0; r < n; r++)
			for (int c = 0; c < n; c++)
				bb.putFloat(r == c ? 1f : 0f);
		return bb.array();
	}

	private static byte[] smallF32(long nelems) {
		ByteBuffer bb = ByteBuffer.allocate((int) (nelems * 4)).order(ByteOrder.LITTLE_ENDIAN);
		for (int i = 0; i < nelems; i++)
			bb.putFloat(0.01f);
		return bb.array();
	}

	/** Deterministic pseudo-random small F32 values so different rows are distinguishable. */
	private static byte[] varyingF32(long nelems, long seed) {
		java.util.Random r = new java.util.Random(seed);
		ByteBuffer bb = ByteBuffer.allocate((int) (nelems * 4)).order(ByteOrder.LITTLE_ENDIAN);
		for (int i = 0; i < nelems; i++)
			bb.putFloat((float) (r.nextGaussian() * 0.05));
		return bb.array();
	}

	private static byte[] f32(float[] values) {
		ByteBuffer bb = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
		for (float v : values)
			bb.putFloat(v);
		return bb.array();
	}
}
