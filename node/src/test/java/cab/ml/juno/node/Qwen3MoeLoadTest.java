package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Verifies {@link Qwen3MoeTransformerHandler} loads MoE tensors and forwards
 * without error on a minimal synthetic qwen3moe GGUF.
 */
@DisplayName("Qwen3-MoE load and forward")
class Qwen3MoeLoadTest {

	private static final int H = 256;
	private static final int HEADS = 8;
	private static final int KV_HEADS = 4;
	private static final int HEAD_DIM = 32;
	private static final int EXP_FF = 128;
	private static final int EXPERTS = 4;
	private static final int TOP_K = 2;
	private static final int VOCAB = 256;
	private static final int LAYERS = 1;

	@Test
	@DisplayName("Qwen3-MoE synthetic GGUF loads and produces finite logits")
	void qwen3moe_loadsAndForwards(@TempDir Path tmp) throws IOException {
		Path gguf = buildSyntheticQwen3MoeGguf(tmp);

		ShardContext ctx = new ShardContext("n0", 0, LAYERS, true, true, VOCAB, H, HEADS);
		Qwen3MoeTransformerHandler handler = Qwen3MoeTransformerHandler.load(gguf, ctx);

		ForwardResult result = handler.forward(ForwardRequest.withTokens("req-moe", new int[] { 1 }, 0), ctx);

		assertThat(result.logits()).hasSize(VOCAB);
		for (float v : result.logits()) {
			assertThat(v).isNotNaN().isFinite();
		}
	}

	@Test
	@DisplayName("ForwardPassHandlerLoader routes qwen3moe GGUF to Qwen3MoeTransformerHandler")
	void loaderRoutesQwen3MoeArch(@TempDir Path tmp) throws IOException {
		Path gguf = buildSyntheticQwen3MoeGguf(tmp);
		ShardContext ctx = new ShardContext("n0", 0, LAYERS, true, true, VOCAB, H, HEADS);

		ForwardPassHandler handler = ForwardPassHandlerLoader.load(gguf, ctx);

		assertThat(handler).isInstanceOf(Qwen3MoeTransformerHandler.class);
	}

	static Path buildSyntheticQwen3MoeGguf(Path dir) throws IOException {
		Files.createDirectories(dir);
		int kvDim = KV_HEADS * HEAD_DIM;
		Phi3TransformerHandlerTest.GgufAssembler gguf = new Phi3TransformerHandlerTest.GgufAssembler();

		gguf.addString("general.architecture", "qwen3moe");
		gguf.addUInt32("qwen3moe.embedding_length", H);
		gguf.addUInt32("qwen3moe.block_count", LAYERS);
		gguf.addUInt32("qwen3moe.attention.head_count", HEADS);
		gguf.addUInt32("qwen3moe.attention.head_count_kv", KV_HEADS);
		gguf.addUInt32("qwen3moe.attention.key_length", HEAD_DIM);
		gguf.addUInt32("qwen3moe.vocab_size", VOCAB);
		gguf.addUInt32("qwen3moe.feed_forward_length", EXP_FF);
		gguf.addUInt32("qwen3moe.expert_count", EXPERTS);
		gguf.addUInt32("qwen3moe.expert_used_count", TOP_K);
		gguf.addUInt32("qwen3moe.expert_feed_forward_length", EXP_FF);
		gguf.addFloat32("qwen3moe.attention.layer_norm_rms_epsilon", 1e-5f);
		gguf.addFloat32("qwen3moe.rope.freq_base", 1000000.0f);
		gguf.addString("qwen3moe.rope.scaling.type", "yarn");
		gguf.addFloat32("qwen3moe.rope.scaling.factor", 4.0f);
		gguf.addUInt32("qwen3moe.rope.scaling.original_context_length", 32768);

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

			// MoE: [numExperts, rows, cols] flattened
			gguf.addTensor(p + "ffn_gate_inp.weight", 12, new long[] { EXPERTS, H },
					Phi3TransformerHandlerTest.zeroQ4K((long) EXPERTS * H));
			gguf.addTensor(p + "ffn_gate_exps.weight", 12, new long[] { EXPERTS, EXP_FF, H },
					Phi3TransformerHandlerTest.zeroQ4K((long) EXPERTS * EXP_FF * H));
			gguf.addTensor(p + "ffn_up_exps.weight", 12, new long[] { EXPERTS, EXP_FF, H },
					Phi3TransformerHandlerTest.zeroQ4K((long) EXPERTS * EXP_FF * H));
			gguf.addTensor(p + "ffn_down_exps.weight", 12, new long[] { EXPERTS, H, EXP_FF },
					Phi3TransformerHandlerTest.zeroQ4K((long) EXPERTS * H * EXP_FF));
		}

		Path out = dir.resolve("synthetic_qwen3moe.gguf");
		Files.write(out, gguf.build());
		return out;
	}

	private static byte[] zeroF32(long nelems) {
		return new byte[(int) (nelems * 4)];
	}
}
