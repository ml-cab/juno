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
 * Gate C — Qwen3 dense LoRA training (per-head Q/K norm + qDim ≠ hiddenDim).
 */
@DisplayName("Qwen3LoraTrainableHandler — Gate C")
class Qwen3LoraTrainableHandlerTest {

	private static final int H = 64;
	private static final int HEADS = 4;
	private static final int KV_HEADS = 2;
	private static final int HEAD_DIM = 16; // qDim == H
	private static final int I = 64;
	private static final int VOCAB = 64;
	private static final int LAYERS = 1;

	// ── Zero-adapter parity with Qwen3TransformerHandler ────────────────────

	@Test
	@DisplayName("zero-adapter logits match Qwen3TransformerHandler on identical synthetic weights")
	void zeroAdapter_matchesBase(@TempDir Path tmp) throws IOException {
		Path gguf = buildQwen3F32Gguf(tmp, H, HEADS, KV_HEADS, HEAD_DIM, I, VOCAB, LAYERS);
		ShardContext ctx = new ShardContext("n0", 0, LAYERS, true, true, VOCAB, H, HEADS);

		Qwen3TransformerHandler base = Qwen3TransformerHandler.load(gguf, ctx, CpuMatVec.INSTANCE);
		LoraAdapterSet adapters;
		try (GgufReader r = GgufReader.open(gguf)) {
			Qwen3Config cfg = Qwen3Config.from(r);
			adapters = LoraInitializer.create(LoraModelLayout.qwen3(cfg), LoraProjection.qv(),
					LoraAdapterConfig.legacy(2, 2f), new Random(1));
		}
		zeroAdapters(adapters);

		Qwen3LoraTrainableHandler lora = Qwen3LoraTrainableHandler.load(gguf, ctx, adapters, CpuMatVec.INSTANCE);

		ForwardRequest req = ForwardRequest.withTokens("r", new int[] { 3 }, 0);
		float[] baseLogits = base.forward(req, ctx).logits();
		float[] loraLogits = lora.forward(req, ctx).logits();
		assertThat(loraLogits).hasSameSizeAs(baseLogits);
		for (int j = 0; j < baseLogits.length; j++)
			assertThat(loraLogits[j]).as("logit[%d]", j).isCloseTo(baseLogits[j], within(1e-4f));
	}

	// ── qDim != hiddenDim path ─────────────────────────────────────────────

	@Test
	@DisplayName("handler forwards when key_length picks qDim != hiddenDim (e.g. 128 vs 64)")
	void qDimDiffersFromHiddenDim(@TempDir Path tmp) throws IOException {
		int hiddenDim = 64;
		int heads = 4;
		int kvHeads = 2;
		int keyLen = 32; // qDim = heads*keyLen = 128 ≠ hiddenDim
		Path gguf = buildQwen3F32Gguf(tmp, hiddenDim, heads, kvHeads, keyLen, I, VOCAB, 1);
		ShardContext ctx = new ShardContext("n0", 0, 1, true, true, VOCAB, hiddenDim, heads);

		LoraAdapterSet adapters;
		try (GgufReader r = GgufReader.open(gguf)) {
			Qwen3Config cfg = Qwen3Config.from(r);
			assertThat(cfg.qDim()).isEqualTo(heads * keyLen);
			assertThat(cfg.qDim()).isNotEqualTo(cfg.hiddenDim());
			LoraModelLayout layout = LoraModelLayout.qwen3(cfg);
			assertThat(layout.binding(0, LoraProjection.WQ).outDim()).isEqualTo(cfg.qDim());
			assertThat(layout.binding(0, LoraProjection.WO).inDim()).isEqualTo(cfg.qDim());
			adapters = LoraInitializer.create(layout, LoraProjection.qv(),
					LoraAdapterConfig.legacy(2, 2f), new Random(2));
		}
		Qwen3LoraTrainableHandler handler = Qwen3LoraTrainableHandler.load(gguf, ctx, adapters, CpuMatVec.INSTANCE);

		float[] logits = handler.forward(ForwardRequest.withTokens("q", new int[] { 1 }, 0), ctx).logits();
		assertThat(logits).hasSize(VOCAB);
		for (float v : logits)
			assertThat(v).isFinite();
	}

	@Test
	@DisplayName("factory routes qwen3 adapters to Qwen3LoraTrainableHandler")
	void factoryRoutesQwen3(@TempDir Path tmp) throws IOException {
		Path gguf = buildQwen3F32Gguf(tmp, H, HEADS, KV_HEADS, HEAD_DIM, I, VOCAB, LAYERS);
		ShardContext ctx = new ShardContext("n0", 0, LAYERS, true, true, VOCAB, H, HEADS);
		LoraAdapterSet adapters;
		try (GgufReader r = GgufReader.open(gguf)) {
			adapters = LoraInitializer.create(LoraModelLayout.qwen3(Qwen3Config.from(r)), LoraProjection.qv(),
					LoraAdapterConfig.legacy(2, 2f), new Random(3));
		}
		ForwardPassHandler h = ForwardPassHandlerLoader.load(gguf, ctx, CpuMatVec.INSTANCE, adapters);
		assertThat(h).isInstanceOf(Qwen3LoraTrainableHandler.class);
	}

	@Test
	@DisplayName("per-head RMSNorm backward finite-difference")
	void perHeadRmsNorm_finiteDifference() {
		int nHeads = 2;
		int headDim = 4;
		Random rng = new Random(3);
		float[] x = new float[nHeads * headDim];
		float[] w = new float[headDim];
		float[] g = new float[nHeads * headDim];
		for (int i = 0; i < x.length; i++) {
			x[i] = rng.nextFloat() - 0.5f;
			g[i] = rng.nextFloat() - 0.5f;
		}
		for (int i = 0; i < headDim; i++)
			w[i] = 0.5f + rng.nextFloat();

		float[] analytic = LoraTrainingMath.perHeadRmsNormBackward(x, w, g, nHeads, headDim, 1e-5f);
		float eps = 1e-3f;
		for (int i = 0; i < x.length; i++) {
			float orig = x[i];
			x[i] = orig + eps;
			float lp = forwardPerHeadNormDot(x, w, g, nHeads, headDim);
			x[i] = orig - eps;
			float lm = forwardPerHeadNormDot(x, w, g, nHeads, headDim);
			x[i] = orig;
			float numeric = (lp - lm) / (2f * eps);
			assertThat(analytic[i]).as("index " + i).isCloseTo(numeric, within(2e-2f));
		}
	}

	private static float forwardPerHeadNormDot(float[] x, float[] w, float[] g, int nHeads, int headDim) {
		float[] y = new float[x.length];
		for (int h = 0; h < nHeads; h++) {
			int base = h * headDim;
			float ss = 0f;
			for (int i = 0; i < headDim; i++)
				ss += x[base + i] * x[base + i];
			float scale = (float) (1.0 / Math.sqrt(ss / headDim + 1e-5));
			for (int i = 0; i < headDim; i++)
				y[base + i] = w[i] * x[base + i] * scale;
		}
		float dot = 0f;
		for (int i = 0; i < y.length; i++)
			dot += y[i] * g[i];
		return dot;
	}

	// ── FD grad on wq adapter ──────────────────────────────────────────────

	@Test
	@DisplayName("finite-difference LoRA adapter gradient for wq")
	void finiteDifference_wq(@TempDir Path tmp) throws IOException {
		Path gguf = buildQwen3F32Gguf(tmp, H, HEADS, KV_HEADS, HEAD_DIM, I, VOCAB, LAYERS);
		ShardContext ctx = new ShardContext("n0", 0, LAYERS, true, true, VOCAB, H, HEADS);
		LoraAdapterSet adapters;
		try (GgufReader r = GgufReader.open(gguf)) {
			adapters = LoraInitializer.create(LoraModelLayout.qwen3(Qwen3Config.from(r)), LoraProjection.qv(),
					LoraAdapterConfig.legacy(2, 2f), new Random(7));
		}
		Qwen3LoraTrainableHandler handler = Qwen3LoraTrainableHandler.load(gguf, ctx, adapters, CpuMatVec.INSTANCE);

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

	// ── Tiny overfit ────────────────────────────────────────────────────────

	@Test
	@DisplayName("tiny synthetic overfit decreases loss")
	void tinyOverfit_decreasesLoss(@TempDir Path tmp) throws IOException {
		Path gguf = buildQwen3F32Gguf(tmp, H, HEADS, KV_HEADS, HEAD_DIM, I, VOCAB, LAYERS);
		ShardContext ctx = new ShardContext("n0", 0, LAYERS, true, true, VOCAB, H, HEADS);
		LoraAdapterSet adapters;
		try (GgufReader r = GgufReader.open(gguf)) {
			adapters = LoraInitializer.create(LoraModelLayout.qwen3(Qwen3Config.from(r)),
					LoraProjection.qv(), LoraAdapterConfig.legacy(4, 8f), new Random(42));
		}
		Qwen3LoraTrainableHandler handler = Qwen3LoraTrainableHandler.load(gguf, ctx, adapters, CpuMatVec.INSTANCE);
		LoraAdamOptimizer opt = new LoraAdamOptimizer(1e-1, 0.9, 0.999, 1e-8, 0.0, 1.0);

		int[] tokens = new int[] { 1, 2, 3, 4, 5 };
		float first = handler.evaluateLoss(tokens).meanLoss();
		for (int step = 0; step < 80; step++) {
			adapters.zeroAllGrads();
			var gr = handler.computeGradients(tokens);
			adapters.prepareGradientsForOptimizer(gr.predictionCount(), 0f);
			opt.step(adapters);
		}
		float last = handler.evaluateLoss(tokens).meanLoss();
		float bNorm = 0f;
		for (var e : adapters.asMap().entrySet())
			for (float v : e.getValue().b())
				bNorm += Math.abs(v);
		assertThat(bNorm).as("B should leave zero init").isGreaterThan(0f);
		assertThat(last).as("loss %.4f -> %.4f", first, last).isLessThan(first);
	}

	// ── Helpers ─────────────────────────────────────────────────────────────

	private static void zeroAdapters(LoraAdapterSet adapters) {
		for (var e : adapters.asMap().entrySet()) {
			LoraAdapter a = e.getValue();
			java.util.Arrays.fill(a.a(), 0f);
			java.util.Arrays.fill(a.b(), 0f);
		}
	}

	/**
	 * Build a synthetic dense Qwen3 GGUF with F32 weights so LoRA can meaningfully
	 * change logits (F32 avoids Q4_K block encoding). Supports
	 * {@code key_length != hiddenDim/heads} to exercise the {@code qDim != H} path.
	 */
	static Path buildQwen3F32Gguf(Path dir, int hiddenDim, int heads, int kvHeads, int headDim, int i, int vocab,
			int layers) throws IOException {
		Files.createDirectories(dir);
		int qDim = heads * headDim;
		int kvDim = kvHeads * headDim;
		Phi3TransformerHandlerTest.GgufAssembler gguf = new Phi3TransformerHandlerTest.GgufAssembler();

		gguf.addString("general.architecture", "qwen3");
		gguf.addUInt32("qwen3.embedding_length", hiddenDim);
		gguf.addUInt32("qwen3.block_count", layers);
		gguf.addUInt32("qwen3.attention.head_count", heads);
		gguf.addUInt32("qwen3.attention.head_count_kv", kvHeads);
		gguf.addUInt32("qwen3.attention.key_length", headDim);
		gguf.addUInt32("qwen3.vocab_size", vocab);
		gguf.addUInt32("qwen3.feed_forward_length", i);
		gguf.addFloat32("qwen3.attention.layer_norm_rms_epsilon", 1e-5f);
		gguf.addFloat32("qwen3.rope.freq_base", 10000.0f);

		gguf.addTensor("token_embd.weight", 0, new long[] { vocab, hiddenDim },
				varyingF32((long) vocab * hiddenDim, 7));
		gguf.addTensor("output_norm.weight", 0, new long[] { hiddenDim }, onesF32(hiddenDim));
		gguf.addTensor("output.weight", 0, new long[] { vocab, hiddenDim },
				varyingF32((long) vocab * hiddenDim, 11));

		for (int li = 0; li < layers; li++) {
			String p = "blk." + li + ".";
			gguf.addTensor(p + "attn_norm.weight", 0, new long[] { hiddenDim }, onesF32(hiddenDim));
			gguf.addTensor(p + "ffn_norm.weight", 0, new long[] { hiddenDim }, onesF32(hiddenDim));
			gguf.addTensor(p + "attn_q_norm.weight", 0, new long[] { headDim }, onesF32(headDim));
			gguf.addTensor(p + "attn_k_norm.weight", 0, new long[] { headDim }, onesF32(headDim));

			gguf.addTensor(p + "attn_q.weight", 0, new long[] { qDim, hiddenDim },
					varyingF32((long) qDim * hiddenDim, 20L + li));
			gguf.addTensor(p + "attn_k.weight", 0, new long[] { kvDim, hiddenDim },
					varyingF32((long) kvDim * hiddenDim, 21L + li));
			gguf.addTensor(p + "attn_v.weight", 0, new long[] { kvDim, hiddenDim },
					varyingF32((long) kvDim * hiddenDim, 22L + li));
			gguf.addTensor(p + "attn_output.weight", 0, new long[] { hiddenDim, qDim },
					varyingF32((long) hiddenDim * qDim, 23L + li));
			gguf.addTensor(p + "ffn_gate.weight", 0, new long[] { i, hiddenDim },
					varyingF32((long) i * hiddenDim, 24L + li));
			gguf.addTensor(p + "ffn_up.weight", 0, new long[] { i, hiddenDim },
					varyingF32((long) i * hiddenDim, 25L + li));
			gguf.addTensor(p + "ffn_down.weight", 0, new long[] { hiddenDim, i },
					varyingF32((long) hiddenDim * i, 26L + li));
		}

		Path out = dir.resolve("qwen3_lora_synth.gguf");
		Files.write(out, gguf.build());
		return out;
	}

	private static byte[] onesF32(int n) {
		ByteBuffer bb = ByteBuffer.allocate(n * 4).order(ByteOrder.LITTLE_ENDIAN);
		for (int i = 0; i < n; i++)
			bb.putFloat(1f);
		return bb.array();
	}

	private static byte[] varyingF32(long nelems, long seed) {
		Random r = new Random(seed);
		ByteBuffer bb = ByteBuffer.allocate((int) (nelems * 4)).order(ByteOrder.LITTLE_ENDIAN);
		for (int i = 0; i < nelems; i++)
			bb.putFloat((float) (r.nextGaussian() * 0.05));
		return bb.array();
	}
}
