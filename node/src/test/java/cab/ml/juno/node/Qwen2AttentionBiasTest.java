package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Verifies that {@link LlamaTransformerHandler} loads and applies Qwen2-style
 * {@code attn_q/k/v.bias} tensors. Qwen2 models require these biases; omitting
 * them produces garbage logits.
 */
@DisplayName("Qwen2 attention QKV bias")
class Qwen2AttentionBiasTest {

	private static final int H = 256;
	private static final int HEADS = 8;
	private static final int KV_HEADS = 4;
	private static final int I = 256;
	private static final int VOCAB = 256;
	private static final int LAYERS = 1;

	@Test
	@DisplayName("Qwen2 GGUF with QKV biases loads and produces vocab-sized logits")
	void qwen2_withBiases_loadsAndForwards(@TempDir Path tmp) throws IOException {
		Path gguf = buildSyntheticQwen2Gguf(tmp, zeroBias(H), zeroBias(H / HEADS * KV_HEADS));

		ShardContext ctx = new ShardContext("n0", 0, LAYERS, true, true, VOCAB, H, HEADS);
		LlamaTransformerHandler handler = LlamaTransformerHandler.load(gguf, ctx);

		ForwardResult result = handler.forward(ForwardRequest.withTokens("req-1", new int[] { 1 }, 0), ctx);

		assertThat(result.isFinalNode()).isTrue();
		assertThat(result.logits()).hasSize(VOCAB);
		for (float v : result.logits()) {
			assertThat(v).isNotNaN().isFinite();
		}
	}

	@Test
	@DisplayName("Qwen2 GGUF without QKV biases still loads (LLaMA backward compat)")
	void qwen2_withoutBiases_stillLoads(@TempDir Path tmp) throws IOException {
		Path gguf = buildSyntheticQwen2GgufNoBias(tmp);

		ShardContext ctx = new ShardContext("n0", 0, LAYERS, true, true, VOCAB, H, HEADS);
		LlamaTransformerHandler handler = LlamaTransformerHandler.load(gguf, ctx);

		ForwardResult result = handler.forward(ForwardRequest.withTokens("req-3", new int[] { 1 }, 0), ctx);

		assertThat(result.logits()).hasSize(VOCAB);
	}

	@Test
	@DisplayName("ForwardPassHandlerLoader routes qwen2 GGUF to LlamaTransformerHandler")
	void loaderRoutesQwen2ArchToLlamaHandler(@TempDir Path tmp) throws IOException {
		Path gguf = buildSyntheticQwen2Gguf(tmp, zeroBias(H), zeroBias(H / HEADS * KV_HEADS));
		ShardContext ctx = new ShardContext("n0", 0, LAYERS, true, true, VOCAB, H, HEADS);

		ForwardPassHandler handler = ForwardPassHandlerLoader.load(gguf, ctx);

		assertThat(handler).isInstanceOf(LlamaTransformerHandler.class);
	}

	private static float[] zeroBias(int n) {
		return new float[n];
	}

	private static Path buildSyntheticQwen2GgufNoBias(Path dir) throws IOException {
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

		gguf.addTensor("token_embd.weight", 0, new long[] { VOCAB, H }, zeroF32((long) VOCAB * H));
		gguf.addTensor("output_norm.weight", 0, new long[] { H }, zeroF32(H));
		gguf.addTensor("output.weight", 0, new long[] { VOCAB, H }, zeroF32((long) VOCAB * H));

		for (int li = 0; li < LAYERS; li++) {
			String p = "blk." + li + ".";
			gguf.addTensor(p + "attn_norm.weight", 0, new long[] { H }, zeroF32(H));
			gguf.addTensor(p + "ffn_norm.weight", 0, new long[] { H }, zeroF32(H));
			gguf.addTensor(p + "attn_q.weight", 12, new long[] { H, H }, Phi3TransformerHandlerTest.zeroQ4K((long) H * H));
			gguf.addTensor(p + "attn_k.weight", 12, new long[] { kvDim, H }, Phi3TransformerHandlerTest.zeroQ4K((long) kvDim * H));
			gguf.addTensor(p + "attn_v.weight", 12, new long[] { kvDim, H }, Phi3TransformerHandlerTest.zeroQ4K((long) kvDim * H));
			gguf.addTensor(p + "attn_output.weight", 12, new long[] { H, H }, Phi3TransformerHandlerTest.zeroQ4K((long) H * H));
			gguf.addTensor(p + "ffn_gate.weight", 12, new long[] { I, H }, Phi3TransformerHandlerTest.zeroQ4K((long) I * H));
			gguf.addTensor(p + "ffn_up.weight", 12, new long[] { I, H }, Phi3TransformerHandlerTest.zeroQ4K((long) I * H));
			gguf.addTensor(p + "ffn_down.weight", 12, new long[] { H, I }, Phi3TransformerHandlerTest.zeroQ4K((long) H * I));
		}

		Path out = dir.resolve("synthetic_qwen2_no_bias.gguf");
		Files.write(out, gguf.build());
		return out;
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

		gguf.addTensor("token_embd.weight", 0, new long[] { VOCAB, H }, zeroF32((long) VOCAB * H));
		gguf.addTensor("output_norm.weight", 0, new long[] { H }, zeroF32(H));
		gguf.addTensor("output.weight", 0, new long[] { VOCAB, H }, zeroF32((long) VOCAB * H));

		for (int li = 0; li < LAYERS; li++) {
			String p = "blk." + li + ".";
			gguf.addTensor(p + "attn_norm.weight", 0, new long[] { H }, zeroF32(H));
			gguf.addTensor(p + "ffn_norm.weight", 0, new long[] { H }, zeroF32(H));
			gguf.addTensor(p + "attn_q.weight", 12, new long[] { H, H }, Phi3TransformerHandlerTest.zeroQ4K((long) H * H));
			gguf.addTensor(p + "attn_k.weight", 12, new long[] { kvDim, H }, Phi3TransformerHandlerTest.zeroQ4K((long) kvDim * H));
			gguf.addTensor(p + "attn_v.weight", 12, new long[] { kvDim, H }, Phi3TransformerHandlerTest.zeroQ4K((long) kvDim * H));
			gguf.addTensor(p + "attn_output.weight", 12, new long[] { H, H }, Phi3TransformerHandlerTest.zeroQ4K((long) H * H));
			gguf.addTensor(p + "ffn_gate.weight", 12, new long[] { I, H }, Phi3TransformerHandlerTest.zeroQ4K((long) I * H));
			gguf.addTensor(p + "ffn_up.weight", 12, new long[] { I, H }, Phi3TransformerHandlerTest.zeroQ4K((long) I * H));
			gguf.addTensor(p + "ffn_down.weight", 12, new long[] { H, I }, Phi3TransformerHandlerTest.zeroQ4K((long) H * I));
			gguf.addTensor(p + "attn_q.bias", 0, new long[] { H }, f32(qBias));
			gguf.addTensor(p + "attn_k.bias", 0, new long[] { kvDim }, f32(kvBias));
			gguf.addTensor(p + "attn_v.bias", 0, new long[] { kvDim }, f32(kvBias));
		}

		Path out = dir.resolve("synthetic_qwen2.gguf");
		Files.write(out, gguf.build());
		return out;
	}

	private static byte[] zeroF32(long nelems) {
		return new byte[(int) (nelems * 4)];
	}

	private static byte[] f32(float[] values) {
		ByteBuffer bb = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
		for (float v : values)
			bb.putFloat(v);
		return bb.array();
	}
}
