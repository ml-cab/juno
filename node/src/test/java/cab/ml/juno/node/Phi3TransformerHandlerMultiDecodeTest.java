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
@DisplayName("Phi3TransformerHandler — multi-request decode batching")
class Phi3TransformerHandlerMultiDecodeTest {

	private static final int H = 256;
	private static final int HEADS = 8;
	private static final int KV_HEADS = 8;
	private static final int I = 256;
	private static final int VOCAB = 256;
	private static final int LAYERS = 2;

	private Phi3TransformerHandler handler(Path gguf) throws IOException {
		ShardContext ctx = new ShardContext("n1", 0, LAYERS, true, true, VOCAB, H, HEADS);
		return Phi3TransformerHandler.load(gguf, ctx);
	}

	@Test
	@DisplayName("forwardMultiDecode logits match serial forward at independent positions")
	void multi_decode_matches_serial_forward(@TempDir Path tmp) throws IOException {
		Path gguf = Phi3TransformerHandlerTest.buildSyntheticPhiGguf(tmp, H, HEADS, KV_HEADS, I, LAYERS, VOCAB);
		ShardContext ctx = new ShardContext("n1", 0, LAYERS, true, true, VOCAB, H, HEADS);

		List<String> ids = List.of("req-a", "req-b", "req-c");
		int[] tokens = { 3, 17, 42 };
		int[] positions = { 0, 1, 2 };

		Phi3TransformerHandler handler = handler(gguf);
		for (int i = 0; i < ids.size(); i++) {
			for (int p = 0; p < positions[i]; p++)
				handler.forward(ForwardRequest.withTokens(ids.get(i), new int[] { tokens[i] + p }, p), ctx);
		}

		float[][] serial = new float[ids.size()][];
		for (int i = 0; i < ids.size(); i++)
			serial[i] = handler.forward(ForwardRequest.withTokens(ids.get(i), new int[] { tokens[i] }, positions[i]),
					ctx).logits();

		Phi3TransformerHandler batchedHandler = handler(gguf);
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
		Path gguf = Phi3TransformerHandlerTest.buildSyntheticPhiGguf(tmp, H, HEADS, KV_HEADS, I, LAYERS, VOCAB);
		ShardContext ctx = new ShardContext("n1", 0, LAYERS, true, true, VOCAB, H, HEADS);
		Phi3TransformerHandler handler = handler(gguf);

		List<String> ids = List.of("x", "y");
		int[] tokens = { 5, 9 };
		int[] positions = { 0, 0 };

		handler.forwardMultiDecode(MultiDecodeForwardRequest.withTokens(ids, tokens, positions), ctx);
		handler.forwardMultiDecode(
				MultiDecodeForwardRequest.withTokens(ids, new int[] { 6, 10 }, new int[] { 1, 1 }), ctx);

		assertThat(handler.kvCacheAllocatedSlots("x")).isGreaterThanOrEqualTo(2);
		assertThat(handler.kvCacheAllocatedSlots("y")).isGreaterThanOrEqualTo(2);
	}
}
