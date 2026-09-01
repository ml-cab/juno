package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.registry.ShardAssignment;
import cab.ml.juno.registry.ShardMap;

/**
 * Batched multi-request decode must match N serial {@link #forward} calls on
 * the same handler (logits within tolerance).
 */
@DisplayName("LlamaTransformerHandler — multi-request decode batching")
class LlamaTransformerHandlerMultiDecodeTest {

	private static final int VOCAB = 256;
	private static final int H = 32;
	private static final int NH = 4;
	private static final int KVH = 4;
	private static final int LAYERS = 2;

	private LlamaTransformerHandler handler() {
		return LlamaTransformerHandler.newTestInstance(VOCAB, H, NH, KVH, LAYERS, 0, LAYERS, true, true, null);
	}

	private ShardContext ctx() {
		ShardAssignment a = new ShardAssignment("n1", "localhost", 0, 0, LAYERS, true, true);
		return ShardContext.from(a, VOCAB, H, NH);
	}

	@Test
	@DisplayName("forwardMultiDecode logits match serial forward at independent positions")
	void multi_decode_matches_serial_forward() {
		LlamaTransformerHandler handler = handler();
		ShardContext ctx = ctx();

		List<String> ids = List.of("req-a", "req-b", "req-c");
		int[] tokens = { 3, 17, 42 };
		int[] positions = { 0, 1, 2 };

		// Warm KV for each stream with prior positions (prefill/decode history)
		for (int i = 0; i < ids.size(); i++) {
			for (int p = 0; p < positions[i]; p++)
				handler.forward(ForwardRequest.withTokens(ids.get(i), new int[] { tokens[i] + p }, p), ctx);
		}

		float[][] serial = new float[ids.size()][];
		for (int i = 0; i < ids.size(); i++)
			serial[i] = handler.forward(ForwardRequest.withTokens(ids.get(i), new int[] { tokens[i] }, positions[i]),
					ctx).logits();

		// Fresh handler so batched path does not see serial KV updates
		LlamaTransformerHandler batchedHandler = handler();
		for (int i = 0; i < ids.size(); i++) {
			for (int p = 0; p < positions[i]; p++)
				batchedHandler.forward(ForwardRequest.withTokens(ids.get(i), new int[] { tokens[i] + p }, p), ctx);
		}

		MultiDecodeForwardRequest batchReq = MultiDecodeForwardRequest.withTokens(ids, tokens, positions);
		float[][] batched = batchedHandler.forwardMultiDecode(batchReq, ctx).logits();

		assertThat(batched.length).isEqualTo(ids.size());
		for (int i = 0; i < ids.size(); i++)
			assertThat(batched[i].length).isEqualTo(VOCAB);
		for (int i = 0; i < ids.size(); i++) {
			for (int v = 0; v < VOCAB; v++)
				assertThat(batched[i][v]).as("req %d vocab %d", i, v).isCloseTo(serial[i][v], within(1e-4f));
		}
	}

	@Test
	@DisplayName("forwardMultiDecode updates KV for every request independently")
	void multi_decode_updates_kv_per_request() {
		LlamaTransformerHandler handler = handler();
		ShardContext ctx = ctx();

		List<String> ids = List.of("x", "y");
		int[] tokens = { 5, 9 };
		int[] positions = { 0, 0 };

		handler.forwardMultiDecode(MultiDecodeForwardRequest.withTokens(ids, tokens, positions), ctx);

		// Second token per stream — KV from step 0 must be visible
		handler.forwardMultiDecode(
				MultiDecodeForwardRequest.withTokens(ids, new int[] { 6, 10 }, new int[] { 1, 1 }), ctx);

		assertThat(handler.kvCacheAllocatedSlots("x")).isGreaterThanOrEqualTo(2);
		assertThat(handler.kvCacheAllocatedSlots("y")).isGreaterThanOrEqualTo(2);
	}
}
