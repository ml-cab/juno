package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Batched multi-request decode must match N serial {@link #forward} calls.
 */
@DisplayName("Phi2TransformerHandler — multi-request decode batching")
class Phi2TransformerHandlerMultiDecodeTest {

	private static final int H = 256;
	private static final int HEADS = 8;
	private static final int KV_HEADS = 8;
	private static final int I = 256;
	private static final int VOCAB = 256;
	private static final int LAYERS = 2;

	private Phi2TransformerHandler handler(Path gguf) throws IOException {
		ShardContext ctx = new ShardContext("n1", 0, LAYERS, true, true, VOCAB, H, HEADS);
		return Phi2TransformerHandler.load(gguf, ctx);
	}

	@Test
	@DisplayName("forwardMultiDecode logits match serial forward at independent positions")
	void multi_decode_matches_serial_forward(@TempDir Path tmp) throws IOException {
		Path gguf = buildSyntheticPhi2Gguf(tmp);
		ShardContext ctx = new ShardContext("n1", 0, LAYERS, true, true, VOCAB, H, HEADS);

		List<String> ids = List.of("req-a", "req-b", "req-c");
		int[] tokens = { 3, 17, 42 };
		int[] positions = { 0, 1, 2 };

		Phi2TransformerHandler handler = handler(gguf);
		for (int i = 0; i < ids.size(); i++) {
			for (int p = 0; p < positions[i]; p++)
				handler.forward(ForwardRequest.withTokens(ids.get(i), new int[] { tokens[i] + p }, p), ctx);
		}

		float[][] serial = new float[ids.size()][];
		for (int i = 0; i < ids.size(); i++)
			serial[i] = handler.forward(ForwardRequest.withTokens(ids.get(i), new int[] { tokens[i] }, positions[i]),
					ctx).logits();

		Phi2TransformerHandler batchedHandler = handler(gguf);
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
		Path gguf = buildSyntheticPhi2Gguf(tmp);
		ShardContext ctx = new ShardContext("n1", 0, LAYERS, true, true, VOCAB, H, HEADS);
		Phi2TransformerHandler handler = handler(gguf);

		List<String> ids = List.of("x", "y");
		handler.forwardMultiDecode(MultiDecodeForwardRequest.withTokens(ids, new int[] { 5, 9 }, new int[] { 0, 0 }),
				ctx);
		handler.forwardMultiDecode(
				MultiDecodeForwardRequest.withTokens(ids, new int[] { 6, 10 }, new int[] { 1, 1 }), ctx);

		assertThat(handler.kvCacheAllocatedSlots("x")).isGreaterThanOrEqualTo(2);
		assertThat(handler.kvCacheAllocatedSlots("y")).isGreaterThanOrEqualTo(2);
	}

	static Path buildSyntheticPhi2Gguf(Path dir) throws IOException {
		int kvDim = KV_HEADS * (H / HEADS);
		Phi3TransformerHandlerTest.GgufAssembler gguf = new Phi3TransformerHandlerTest.GgufAssembler();

		gguf.addString("general.architecture", "phi2");
		gguf.addUInt32("phi2.embedding_length", H);
		gguf.addUInt32("phi2.block_count", LAYERS);
		gguf.addUInt32("phi2.attention.head_count", HEADS);
		gguf.addUInt32("phi2.attention.head_count_kv", KV_HEADS);
		gguf.addUInt32("phi2.vocab_size", VOCAB);
		gguf.addUInt32("phi2.feed_forward_length", I);
		gguf.addFloat32("phi2.attention.layer_norm_epsilon", 1e-5f);
		gguf.addFloat32("phi2.rope.freq_base", 10000.0f);
		gguf.addUInt32("phi2.rope.dimension_count", H / HEADS);

		gguf.addTensor("token_embd.weight", 0, new long[] { VOCAB, H }, zeroF32((long) VOCAB * H));
		gguf.addTensor("output_norm.weight", 0, new long[] { H }, zeroF32(H));
		gguf.addTensor("output.weight", 0, new long[] { VOCAB, H }, zeroF32((long) VOCAB * H));

		for (int li = 0; li < LAYERS; li++) {
			String p = "blk." + li + ".";
			gguf.addTensor(p + "attn_norm.weight", 0, new long[] { H }, zeroF32(H));
			long qkvRows = H + kvDim + kvDim;
			gguf.addTensor(p + "attn_qkv.weight", 12, new long[] { qkvRows, H },
					Phi3TransformerHandlerTest.zeroQ4K(qkvRows * H));
			gguf.addTensor(p + "attn_output.weight", 12, new long[] { H, H },
					Phi3TransformerHandlerTest.zeroQ4K((long) H * H));
			gguf.addTensor(p + "ffn_up.weight", 12, new long[] { I, H },
					Phi3TransformerHandlerTest.zeroQ4K((long) I * H));
			gguf.addTensor(p + "ffn_down.weight", 12, new long[] { H, I },
					Phi3TransformerHandlerTest.zeroQ4K((long) H * I));
		}

		Path out = dir.resolve("synthetic_phi2.gguf");
		Files.write(out, gguf.build());
		return out;
	}

	private static byte[] zeroF32(long nelems) {
		return new byte[(int) (nelems * 4)];
	}
}
