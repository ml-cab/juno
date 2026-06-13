package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Verifies {@link Qwen3TransformerHandler} loads q/k norm tensors and produces
 * finite logits for synthetic qwen3 GGUF.
 */
@DisplayName("Qwen3 attention Q/K norm")
class Qwen3AttentionNormTest {

	private static final int H = 256;
	private static final int HEADS = 8;
	private static final int KV_HEADS = 4;
	private static final int HEAD_DIM = 32;
	private static final int I = 256;
	private static final int VOCAB = 256;
	private static final int LAYERS = 1;

	@Test
	@DisplayName("Qwen3 GGUF with q/k norms loads and produces vocab-sized logits")
	void qwen3_withNorms_loadsAndForwards(@TempDir Path tmp) throws IOException {
		Path gguf = buildSyntheticQwen3Gguf(tmp);

		ShardContext ctx = new ShardContext("n0", 0, LAYERS, true, true, VOCAB, H, HEADS);
		Qwen3TransformerHandler handler = Qwen3TransformerHandler.load(gguf, ctx);

		ForwardResult result = handler.forward(ForwardRequest.withTokens("req-1", new int[] { 1 }, 0), ctx);

		assertThat(result.isFinalNode()).isTrue();
		assertThat(result.logits()).hasSize(VOCAB);
		for (float v : result.logits()) {
			assertThat(v).isNotNaN().isFinite();
		}
	}

	@Test
	@DisplayName("ForwardPassHandlerLoader routes qwen3 GGUF to Qwen3TransformerHandler")
	void loaderRoutesQwen3Arch(@TempDir Path tmp) throws IOException {
		Path gguf = buildSyntheticQwen3Gguf(tmp);
		ShardContext ctx = new ShardContext("n0", 0, LAYERS, true, true, VOCAB, H, HEADS);

		ForwardPassHandler handler = ForwardPassHandlerLoader.load(gguf, ctx);

		assertThat(handler).isInstanceOf(Qwen3TransformerHandler.class);
	}

	@Test
	@DisplayName("Qwen3Config reads attention.key_length for head_dim")
	void configUsesKeyLength(@TempDir Path tmp) throws IOException {
		Path gguf = buildSyntheticQwen3Gguf(tmp);
		try (GgufReader r = GgufReader.open(gguf)) {
			Qwen3Config cfg = Qwen3Config.from(r);
			assertThat(cfg.headDim()).isEqualTo(HEAD_DIM);
		}
	}

	static Path buildSyntheticQwen3Gguf(Path dir) throws IOException {
		Files.createDirectories(dir);
		int kvDim = KV_HEADS * HEAD_DIM;
		Phi3TransformerHandlerTest.GgufAssembler gguf = new Phi3TransformerHandlerTest.GgufAssembler();

		gguf.addString("general.architecture", "qwen3");
		gguf.addUInt32("qwen3.embedding_length", H);
		gguf.addUInt32("qwen3.block_count", LAYERS);
		gguf.addUInt32("qwen3.attention.head_count", HEADS);
		gguf.addUInt32("qwen3.attention.head_count_kv", KV_HEADS);
		gguf.addUInt32("qwen3.attention.key_length", HEAD_DIM);
		gguf.addUInt32("qwen3.vocab_size", VOCAB);
		gguf.addUInt32("qwen3.feed_forward_length", I);
		gguf.addFloat32("qwen3.attention.layer_norm_rms_epsilon", 1e-5f);
		gguf.addFloat32("qwen3.rope.freq_base", 10000.0f);

		gguf.addTensor("token_embd.weight", 0, new long[] { VOCAB, H }, zeroF32((long) VOCAB * H));
		gguf.addTensor("output_norm.weight", 0, new long[] { H }, zeroF32(H));
		gguf.addTensor("output.weight", 0, new long[] { VOCAB, H }, zeroF32((long) VOCAB * H));

		for (int li = 0; li < LAYERS; li++) {
			String p = "blk." + li + ".";
			gguf.addTensor(p + "attn_norm.weight", 0, new long[] { H }, zeroF32(H));
			gguf.addTensor(p + "ffn_norm.weight", 0, new long[] { H }, zeroF32(H));
			gguf.addTensor(p + "attn_q_norm.weight", 0, new long[] { HEAD_DIM }, zeroF32(HEAD_DIM));
			gguf.addTensor(p + "attn_k_norm.weight", 0, new long[] { HEAD_DIM }, zeroF32(HEAD_DIM));
			gguf.addTensor(p + "attn_q.weight", 12, new long[] { H, H },
					Phi3TransformerHandlerTest.zeroQ4K((long) H * H));
			gguf.addTensor(p + "attn_k.weight", 12, new long[] { kvDim, H },
					Phi3TransformerHandlerTest.zeroQ4K((long) kvDim * H));
			gguf.addTensor(p + "attn_v.weight", 12, new long[] { kvDim, H },
					Phi3TransformerHandlerTest.zeroQ4K((long) kvDim * H));
			gguf.addTensor(p + "attn_output.weight", 12, new long[] { H, H },
					Phi3TransformerHandlerTest.zeroQ4K((long) H * H));
			gguf.addTensor(p + "ffn_gate.weight", 12, new long[] { I, H },
					Phi3TransformerHandlerTest.zeroQ4K((long) I * H));
			gguf.addTensor(p + "ffn_up.weight", 12, new long[] { I, H },
					Phi3TransformerHandlerTest.zeroQ4K((long) I * H));
			gguf.addTensor(p + "ffn_down.weight", 12, new long[] { H, I },
					Phi3TransformerHandlerTest.zeroQ4K((long) H * I));
		}

		Path out = dir.resolve("synthetic_qwen3.gguf");
		Files.write(out, gguf.build());
		return out;
	}

	@Test
	@DisplayName("Qwen3Config qDim equals numHeads times headDim when key_length differs from H/numHeads")
	void configQDim_whenKeyLengthDiffers(@TempDir Path tmp) throws IOException {
		Path gguf = buildSyntheticQwen3GgufWithKeyLength(tmp, 256, 8, 4, 64, 32);
		try (GgufReader r = GgufReader.open(gguf)) {
			Qwen3Config cfg = Qwen3Config.from(r);
			assertThat(cfg.headDim()).isEqualTo(64);
			assertThat(cfg.qDim()).isEqualTo(8 * 64);
			assertThat(cfg.qDim()).isNotEqualTo(cfg.hiddenDim());
		}
	}

	@Test
	@DisplayName("Qwen3 with key_length != H/numHeads forwards without index error")
	void qwen3_keyLengthMismatch_forwards(@TempDir Path tmp) throws IOException {
		Path gguf = buildSyntheticQwen3GgufWithKeyLength(tmp, 256, 8, 4, 64, 1);
		ShardContext ctx = new ShardContext("n0", 0, 1, true, true, 256, 256, 8);
		Qwen3TransformerHandler handler = Qwen3TransformerHandler.load(gguf, ctx);
		ForwardResult result = handler.forward(ForwardRequest.withTokens("req-kl", new int[] { 1 }, 0), ctx);
		assertThat(result.logits()).hasSize(256);
	}

	static Path buildSyntheticQwen3GgufWithKeyLength(Path dir, int H, int heads, int kvHeads, int headDim, int layers)
			throws IOException {
		Files.createDirectories(dir);
		int qDim = heads * headDim;
		int kvDim = kvHeads * headDim;
		int I = 256;
		int VOCAB = 256;
		Phi3TransformerHandlerTest.GgufAssembler gguf = new Phi3TransformerHandlerTest.GgufAssembler();

		gguf.addString("general.architecture", "qwen3");
		gguf.addUInt32("qwen3.embedding_length", H);
		gguf.addUInt32("qwen3.block_count", layers);
		gguf.addUInt32("qwen3.attention.head_count", heads);
		gguf.addUInt32("qwen3.attention.head_count_kv", kvHeads);
		gguf.addUInt32("qwen3.attention.key_length", headDim);
		gguf.addUInt32("qwen3.vocab_size", VOCAB);
		gguf.addUInt32("qwen3.feed_forward_length", I);
		gguf.addFloat32("qwen3.attention.layer_norm_rms_epsilon", 1e-5f);
		gguf.addFloat32("qwen3.rope.freq_base", 10000.0f);

		gguf.addTensor("token_embd.weight", 0, new long[] { VOCAB, H }, zeroF32((long) VOCAB * H));
		gguf.addTensor("output_norm.weight", 0, new long[] { H }, zeroF32(H));
		gguf.addTensor("output.weight", 0, new long[] { VOCAB, H }, zeroF32((long) VOCAB * H));

		for (int li = 0; li < layers; li++) {
			String p = "blk." + li + ".";
			gguf.addTensor(p + "attn_norm.weight", 0, new long[] { H }, zeroF32(H));
			gguf.addTensor(p + "ffn_norm.weight", 0, new long[] { H }, zeroF32(H));
			gguf.addTensor(p + "attn_q_norm.weight", 0, new long[] { headDim }, zeroF32(headDim));
			gguf.addTensor(p + "attn_k_norm.weight", 0, new long[] { headDim }, zeroF32(headDim));
			gguf.addTensor(p + "attn_q.weight", 12, new long[] { qDim, H },
					Phi3TransformerHandlerTest.zeroQ4K((long) qDim * H));
			gguf.addTensor(p + "attn_k.weight", 12, new long[] { kvDim, H },
					Phi3TransformerHandlerTest.zeroQ4K((long) kvDim * H));
			gguf.addTensor(p + "attn_v.weight", 12, new long[] { kvDim, H },
					Phi3TransformerHandlerTest.zeroQ4K((long) kvDim * H));
			gguf.addTensor(p + "attn_output.weight", 12, new long[] { H, qDim },
					Phi3TransformerHandlerTest.zeroQ4K((long) H * qDim));
			gguf.addTensor(p + "ffn_gate.weight", 12, new long[] { I, H },
					Phi3TransformerHandlerTest.zeroQ4K((long) I * H));
			gguf.addTensor(p + "ffn_up.weight", 12, new long[] { I, H },
					Phi3TransformerHandlerTest.zeroQ4K((long) I * H));
			gguf.addTensor(p + "ffn_down.weight", 12, new long[] { H, I },
					Phi3TransformerHandlerTest.zeroQ4K((long) H * I));
		}

		Path out = dir.resolve("synthetic_qwen3_keylen.gguf");
		Files.write(out, gguf.build());
		return out;
	}

	private static byte[] zeroF32(long nelems) {
		return new byte[(int) (nelems * 4)];
	}
}
