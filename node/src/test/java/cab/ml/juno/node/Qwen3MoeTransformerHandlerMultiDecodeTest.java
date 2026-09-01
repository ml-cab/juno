package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Batched multi-request decode must match N serial {@link #forward} calls.
 */
@DisplayName("Qwen3MoeTransformerHandler — multi-request decode batching")
class Qwen3MoeTransformerHandlerMultiDecodeTest {

	private static final int H = 256;
	private static final int HEADS = 8;
	private static final int VOCAB = 256;
	private static final int LAYERS = 2;

	private Qwen3MoeTransformerHandler handler(Path gguf) throws IOException {
		ShardContext ctx = new ShardContext("n1", 0, LAYERS, true, true, VOCAB, H, HEADS);
		return Qwen3MoeTransformerHandler.load(gguf, ctx);
	}

	@Test
	@DisplayName("forwardMultiDecode logits match serial forward at independent positions")
	void multi_decode_matches_serial_forward(@TempDir Path tmp) throws IOException {
		Path gguf = buildSyntheticQwen3MoeGguf(tmp, LAYERS);
		ShardContext ctx = new ShardContext("n1", 0, LAYERS, true, true, VOCAB, H, HEADS);

		List<String> ids = List.of("req-a", "req-b", "req-c");
		int[] tokens = { 3, 17, 42 };
		int[] positions = { 0, 1, 2 };

		Qwen3MoeTransformerHandler handler = handler(gguf);
		for (int i = 0; i < ids.size(); i++) {
			for (int p = 0; p < positions[i]; p++)
				handler.forward(ForwardRequest.withTokens(ids.get(i), new int[] { tokens[i] + p }, p), ctx);
		}

		float[][] serial = new float[ids.size()][];
		for (int i = 0; i < ids.size(); i++)
			serial[i] = handler.forward(ForwardRequest.withTokens(ids.get(i), new int[] { tokens[i] }, positions[i]),
					ctx).logits();

		Qwen3MoeTransformerHandler batchedHandler = handler(gguf);
		for (int i = 0; i < ids.size(); i++) {
			for (int p = 0; p < positions[i]; p++)
				batchedHandler.forward(ForwardRequest.withTokens(ids.get(i), new int[] { tokens[i] + p }, p), ctx);
		}

		float[][] batched = batchedHandler
				.forwardMultiDecode(MultiDecodeForwardRequest.withTokens(ids, tokens, positions), ctx).logits();

		assertThat(batched.length).isEqualTo(ids.size());
		for (int i = 0; i < ids.size(); i++) {
			assertThat(batched[i].length).isEqualTo(VOCAB);
			for (int v = 0; v < VOCAB; v++)
				assertThat(batched[i][v]).as("req %d vocab %d", i, v).isCloseTo(serial[i][v], within(1e-4f));
		}
	}

	@Test
	@DisplayName("forwardMultiDecode updates KV for every request independently")
	void multi_decode_updates_kv_per_request(@TempDir Path tmp) throws IOException {
		Path gguf = buildSyntheticQwen3MoeGguf(tmp, LAYERS);
		ShardContext ctx = new ShardContext("n1", 0, LAYERS, true, true, VOCAB, H, HEADS);
		Qwen3MoeTransformerHandler handler = handler(gguf);

		List<String> ids = List.of("x", "y");
		handler.forwardMultiDecode(MultiDecodeForwardRequest.withTokens(ids, new int[] { 5, 9 }, new int[] { 0, 0 }),
				ctx);
		handler.forwardMultiDecode(
				MultiDecodeForwardRequest.withTokens(ids, new int[] { 6, 10 }, new int[] { 1, 1 }), ctx);

		assertThat(handler.kvCacheAllocatedSlots("x")).isGreaterThanOrEqualTo(2);
		assertThat(handler.kvCacheAllocatedSlots("y")).isGreaterThanOrEqualTo(2);
	}

	static Path buildSyntheticQwen3MoeGguf(Path dir, int layers) throws IOException {
		int kvDim = 4 * 32;
		int expFf = 128;
		int experts = 4;
		Phi3TransformerHandlerTest.GgufAssembler gguf = new Phi3TransformerHandlerTest.GgufAssembler();

		gguf.addString("general.architecture", "qwen3moe");
		gguf.addUInt32("qwen3moe.embedding_length", H);
		gguf.addUInt32("qwen3moe.block_count", layers);
		gguf.addUInt32("qwen3moe.attention.head_count", HEADS);
		gguf.addUInt32("qwen3moe.attention.head_count_kv", 4);
		gguf.addUInt32("qwen3moe.attention.key_length", 32);
		gguf.addUInt32("qwen3moe.vocab_size", VOCAB);
		gguf.addUInt32("qwen3moe.feed_forward_length", expFf);
		gguf.addUInt32("qwen3moe.expert_count", experts);
		gguf.addUInt32("qwen3moe.expert_used_count", 2);
		gguf.addUInt32("qwen3moe.expert_feed_forward_length", expFf);
		gguf.addFloat32("qwen3moe.attention.layer_norm_rms_epsilon", 1e-5f);
		gguf.addFloat32("qwen3moe.rope.freq_base", 1000000.0f);

		gguf.addTensor("token_embd.weight", 0, new long[] { VOCAB, H }, zeroF32((long) VOCAB * H));
		gguf.addTensor("output_norm.weight", 0, new long[] { H }, zeroF32(H));
		gguf.addTensor("output.weight", 0, new long[] { VOCAB, H }, zeroF32((long) VOCAB * H));

		for (int li = 0; li < layers; li++) {
			String p = "blk." + li + ".";
			gguf.addTensor(p + "attn_norm.weight", 0, new long[] { H }, zeroF32(H));
			gguf.addTensor(p + "ffn_norm.weight", 0, new long[] { H }, zeroF32(H));
			gguf.addTensor(p + "attn_q_norm.weight", 0, new long[] { 32 }, zeroF32(32));
			gguf.addTensor(p + "attn_k_norm.weight", 0, new long[] { 32 }, zeroF32(32));
			gguf.addTensor(p + "attn_q.weight", 12, new long[] { H, H },
					Phi3TransformerHandlerTest.zeroQ4K((long) H * H));
			gguf.addTensor(p + "attn_k.weight", 12, new long[] { kvDim, H },
					Phi3TransformerHandlerTest.zeroQ4K((long) kvDim * H));
			gguf.addTensor(p + "attn_v.weight", 12, new long[] { kvDim, H },
					Phi3TransformerHandlerTest.zeroQ4K((long) kvDim * H));
			gguf.addTensor(p + "attn_output.weight", 12, new long[] { H, H },
					Phi3TransformerHandlerTest.zeroQ4K((long) H * H));
			gguf.addTensor(p + "ffn_gate_inp.weight", 12, new long[] { experts, H },
					Phi3TransformerHandlerTest.zeroQ4K((long) experts * H));
			gguf.addTensor(p + "ffn_gate_exps.weight", 12, new long[] { experts, expFf, H },
					Phi3TransformerHandlerTest.zeroQ4K((long) experts * expFf * H));
			gguf.addTensor(p + "ffn_up_exps.weight", 12, new long[] { experts, expFf, H },
					Phi3TransformerHandlerTest.zeroQ4K((long) experts * expFf * H));
			gguf.addTensor(p + "ffn_down_exps.weight", 12, new long[] { experts, H, expFf },
					Phi3TransformerHandlerTest.zeroQ4K((long) experts * H * expFf));
		}

		Path out = dir.resolve("synthetic_qwen3moe_multidecode.gguf");
		java.nio.file.Files.write(out, gguf.build());
		return out;
	}

	private static byte[] zeroF32(long nelems) {
		return new byte[(int) (nelems * 4)];
	}
}
