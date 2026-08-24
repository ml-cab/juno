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
import cab.ml.juno.lora.LoraAdapterSet;
import cab.ml.juno.lora.LoraAdamOptimizer;

/**
 * Gate B — Phi-3 fused-QKV / fused gate-up LoRA training.
 */
@DisplayName("Phi3LoraTrainableHandler — Gate B")
class Phi3LoraTrainableHandlerTest {

	private static final int H = 64;
	private static final int HEADS = 4;
	private static final int KV_HEADS = 2;
	private static final int HEAD_DIM = H / HEADS; // 16
	private static final int KV_DIM = KV_HEADS * HEAD_DIM; // 32
	private static final int I = 64;
	private static final int VOCAB = 64;
	private static final int LAYERS = 1;

	// ── Zero-adapter parity with Phi3TransformerHandler ────────────────────

	@Test
	@DisplayName("zero-adapter logits match Phi3TransformerHandler on identical synthetic weights")
	void zeroAdapter_matchesBase(@TempDir Path tmp) throws IOException {
		Path gguf = buildF32PhiGguf(tmp);
		ShardContext ctx = new ShardContext("n0", 0, LAYERS, true, true, VOCAB, H, HEADS);

		Phi3TransformerHandler base = Phi3TransformerHandler.load(gguf, ctx, CpuMatVec.INSTANCE);
		LoraAdapterSet adapters;
		try (GgufReader r = GgufReader.open(gguf)) {
			adapters = LoraInitializer.create(LoraModelLayout.phi3(LlamaConfig.from(r)),
					LoraProjection.qv(), cab.ml.juno.lora.LoraAdapterConfig.legacy(2, 2f), new Random(1));
		}
		zeroAdapters(adapters);

		Phi3LoraTrainableHandler lora = Phi3LoraTrainableHandler.load(gguf, ctx, adapters, CpuMatVec.INSTANCE);

		ForwardRequest req = ForwardRequest.withTokens("r", new int[] { 3 }, 0);
		float[] baseLogits = base.forward(req, ctx).logits();
		float[] loraLogits = lora.forward(req, ctx).logits();
		assertThat(loraLogits).hasSameSizeAs(baseLogits);
		for (int j = 0; j < baseLogits.length; j++)
			assertThat(loraLogits[j]).as("logit[%d]", j).isCloseTo(baseLogits[j], within(1e-4f));
	}

	@Test
	@DisplayName("factory routes phi3 adapters to Phi3LoraTrainableHandler")
	void factoryRoutesPhi3(@TempDir Path tmp) throws IOException {
		Path gguf = buildF32PhiGguf(tmp);
		ShardContext ctx = new ShardContext("n0", 0, LAYERS, true, true, VOCAB, H, HEADS);
		LoraAdapterSet adapters;
		try (GgufReader r = GgufReader.open(gguf)) {
			adapters = LoraInitializer.create(LoraModelLayout.phi3(LlamaConfig.from(r)), LoraProjection.qv(),
					cab.ml.juno.lora.LoraAdapterConfig.legacy(2, 2f), new Random(2));
		}
		ForwardPassHandler h = ForwardPassHandlerLoader.load(gguf, ctx, CpuMatVec.INSTANCE, adapters);
		assertThat(h).isInstanceOf(Phi3LoraTrainableHandler.class);
	}

	// ── Phi3Rope backward FD with attnFactor=1.5 ────────────────────────────

	@Test
	@DisplayName("Phi3Rope.ropeExtBackward matches finite-difference adjoint with attnFactor=1.5")
	void phi3Rope_backward_matchesFiniteDifference() {
		int nHeads = 2;
		int headDim = 8;
		int pos = 5;
		float[] freqFactors = new float[headDim / 2];
		Random r = new Random(101);
		for (int i = 0; i < freqFactors.length; i++)
			freqFactors[i] = 1f + (float) r.nextDouble() * 0.3f;
		Phi3RopeConfig cfg = new Phi3RopeConfig(10000f, 1.0f, 1.5f, 4096, 4096, freqFactors, freqFactors);

		float[] x = new float[nHeads * headDim];
		for (int i = 0; i < x.length; i++)
			x[i] = (float) r.nextGaussian();
		float[] upstream = new float[nHeads * headDim];
		for (int i = 0; i < upstream.length; i++)
			upstream[i] = (float) r.nextGaussian();

		// dL/dx via ropeExtBackward
		float[] analytic = upstream.clone();
		Phi3Rope.ropeExtBackward(analytic, pos, nHeads, headDim, cfg);

		// FD: for each x[j], perturb, run ropeExt, compute L = <upstream, y>
		float h = 1e-3f;
		for (int j = 0; j < x.length; j++) {
			float orig = x[j];
			x[j] = orig + h;
			float[] yPlus = x.clone();
			Phi3Rope.ropeExt(yPlus, pos, nHeads, headDim, cfg);
			float lPlus = dot(upstream, yPlus);

			x[j] = orig - h;
			float[] yMinus = x.clone();
			Phi3Rope.ropeExt(yMinus, pos, nHeads, headDim, cfg);
			float lMinus = dot(upstream, yMinus);

			x[j] = orig;
			float fd = (lPlus - lMinus) / (2 * h);
			assertThat(analytic[j]).as("grad[%d] analytic=%.4f fd=%.4f", j, analytic[j], fd)
					.isCloseTo(fd, within(1e-2f));
		}
	}

	// ── FD grad on wq adapter ──────────────────────────────────────────────

	@Test
	@DisplayName("finite-difference LoRA adapter gradient for wq")
	void finiteDifference_wq(@TempDir Path tmp) throws IOException {
		Path gguf = buildF32PhiGguf(tmp);
		ShardContext ctx = new ShardContext("n0", 0, LAYERS, true, true, VOCAB, H, HEADS);
		LoraAdapterSet adapters;
		try (GgufReader r = GgufReader.open(gguf)) {
			adapters = LoraInitializer.create(LoraModelLayout.phi3(LlamaConfig.from(r)), LoraProjection.qv(),
					cab.ml.juno.lora.LoraAdapterConfig.legacy(2, 2f), new Random(7));
		}
		Phi3LoraTrainableHandler handler = Phi3LoraTrainableHandler.load(gguf, ctx, adapters, CpuMatVec.INSTANCE);

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

	// ── Tiny overfit decreases loss ────────────────────────────────────────

	@Test
	@DisplayName("tiny synthetic overfit decreases loss")
	void tinyOverfit_decreasesLoss(@TempDir Path tmp) throws IOException {
		Path gguf = buildF32PhiGguf(tmp);
		ShardContext ctx = new ShardContext("n0", 0, LAYERS, true, true, VOCAB, H, HEADS);
		LoraAdapterSet adapters;
		try (GgufReader r = GgufReader.open(gguf)) {
			adapters = LoraInitializer.create(LoraModelLayout.phi3(LlamaConfig.from(r)), LoraProjection.qv(),
					cab.ml.juno.lora.LoraAdapterConfig.legacy(4, 8f), new Random(42));
		}
		Phi3LoraTrainableHandler handler = Phi3LoraTrainableHandler.load(gguf, ctx, adapters, CpuMatVec.INSTANCE);
		LoraAdamOptimizer opt = new LoraAdamOptimizer(1e-2, 0.9, 0.999, 1e-8, 0.0, 1.0);

		int[] tokens = new int[] { 1, 2, 3, 4, 5 };
		float first = Float.NaN;
		float last = Float.NaN;
		for (int step = 0; step < 30; step++) {
			adapters.zeroAllGrads();
			var gr = handler.computeGradients(tokens);
			adapters.prepareGradientsForOptimizer(1, 0f);
			opt.step(adapters);
			last = gr.meanLoss();
			if (step == 0)
				first = last;
		}
		assertThat(last).isLessThan(first);
	}

	// ── Helpers ─────────────────────────────────────────────────────────────

	private static void zeroAdapters(LoraAdapterSet adapters) {
		for (var e : adapters.asMap().entrySet()) {
			LoraAdapter a = e.getValue();
			java.util.Arrays.fill(a.a(), 0f);
			java.util.Arrays.fill(a.b(), 0f);
		}
	}

	private static float dot(float[] a, float[] b) {
		float s = 0f;
		for (int i = 0; i < a.length; i++)
			s += a[i] * b[i];
		return s;
	}

	/**
	 * Build a synthetic Phi-3 GGUF with F32 weights so LoRA can influence logits
	 * without needing Q4_K encoding. Non-zero weights keep the loss non-uniform
	 * and enable finite-difference gradient checks.
	 */
	private static Path buildF32PhiGguf(Path dir) throws IOException {
		Files.createDirectories(dir);
		int kvDim = KV_HEADS * HEAD_DIM;
		Phi3TransformerHandlerTest.GgufAssembler gguf = new Phi3TransformerHandlerTest.GgufAssembler();

		gguf.addString("general.architecture", "phi3");
		gguf.addUInt32("phi3.embedding_length", H);
		gguf.addUInt32("phi3.block_count", LAYERS);
		gguf.addUInt32("phi3.attention.head_count", HEADS);
		gguf.addUInt32("phi3.attention.head_count_kv", KV_HEADS);
		gguf.addUInt32("phi3.vocab_size", VOCAB);
		gguf.addUInt32("phi3.feed_forward_length", I);
		gguf.addFloat32("phi3.attention.layer_norm_rms_epsilon", 1e-5f);
		gguf.addFloat32("phi3.rope.freq_base", 10000.0f);

		gguf.addTensor("token_embd.weight", 0, new long[] { VOCAB, H }, varyingF32((long) VOCAB * H, 7));
		gguf.addTensor("output_norm.weight", 0, new long[] { H }, onesF32(H));
		gguf.addTensor("output.weight", 0, new long[] { VOCAB, H }, varyingF32((long) VOCAB * H, 11));

		for (int li = 0; li < LAYERS; li++) {
			String p = "blk." + li + ".";
			gguf.addTensor(p + "attn_norm.weight", 0, new long[] { H }, onesF32(H));
			gguf.addTensor(p + "ffn_norm.weight", 0, new long[] { H }, onesF32(H));
			long qkvRows = H + kvDim + kvDim;
			gguf.addTensor(p + "attn_qkv.weight", 0, new long[] { qkvRows, H }, varyingF32(qkvRows * H, 20L + li));
			gguf.addTensor(p + "attn_output.weight", 0, new long[] { H, H }, varyingF32((long) H * H, 30L + li));
			gguf.addTensor(p + "ffn_up.weight", 0, new long[] { 2L * I, H }, varyingF32(2L * I * H, 40L + li));
			gguf.addTensor(p + "ffn_down.weight", 0, new long[] { H, I }, varyingF32((long) H * I, 50L + li));
		}

		Path out = dir.resolve("phi3_lora_synth.gguf");
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
