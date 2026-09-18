package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.registry.ShardAssignment;

/**
 * {@code forwardVerify}'s per-position logits (one batched GEMM over the whole
 * draft window) must match calling {@code forward} serially, one draft
 * position at a time — the token-identity invariant speculative decoding
 * depends on (a hypothetically zero-cost kernel must not change which token
 * gets emitted, only how fast it is computed).
 */
@DisplayName("LlamaTransformerHandler — forwardVerify parity")
class LlamaTransformerHandlerVerifyParityTest {

	private static final int VOCAB = 256;
	private static final int H = 32;
	private static final int NH = 4;
	private static final int KVH = 4;
	private static final int LAYERS = 2;
	private static final int PROMPT_LEN = 17;
	private static final int DRAFT_LEN = 4;

	private LlamaTransformerHandler handler() {
		return LlamaTransformerHandler.newTestInstance(VOCAB, H, NH, KVH, LAYERS, 0, LAYERS, true, true, null);
	}

	private ShardContext ctx() {
		ShardAssignment a = new ShardAssignment("n1", "localhost", 0, 0, LAYERS, true, true);
		return ShardContext.from(a, VOCAB, H, NH);
	}

	private int[] tokens(int count, int offset) {
		int[] ids = new int[count];
		for (int i = 0; i < ids.length; i++)
			ids[i] = 3 + ((i + offset) % 17);
		return ids;
	}

	/** Prefill a handler up to (but not including) startPosition, serially. */
	private void prefill(LlamaTransformerHandler handler, ShardContext ctx, String requestId, int upToExclusive) {
		int[] prompt = tokens(upToExclusive, 0);
		for (int p = 0; p < upToExclusive; p++) {
			handler.forward(ForwardRequest.withTokens(requestId, new int[] { prompt[p] }, p), ctx);
		}
	}

	@Test
	@DisplayName("batched verify over a draft window matches serial forward() per position")
	void batched_verify_matches_serial_forward() {
		ShardContext ctx = ctx();
		int[] draft = tokens(DRAFT_LEN, PROMPT_LEN);

		// Serial reference: one forward() call per draft position, same handler
		// instance carrying KV state forward across calls (true decode path).
		LlamaTransformerHandler serialHandler = handler();
		prefill(serialHandler, ctx, "serial", PROMPT_LEN);
		float[][] serialLogits = new float[DRAFT_LEN][];
		for (int d = 0; d < DRAFT_LEN; d++) {
			ForwardResult res = serialHandler.forward(
					ForwardRequest.withTokens("serial", new int[] { draft[d] }, PROMPT_LEN + d), ctx);
			serialLogits[d] = res.logits();
		}

		// Batched verify: one call over the whole draft window.
		LlamaTransformerHandler verifyHandler = handler();
		prefill(verifyHandler, ctx, "verify", PROMPT_LEN);
		VerifyBatchResult result = verifyHandler.forwardVerify(
				BatchForwardRequest.withTokens("verify", draft, PROMPT_LEN), ctx);
		assertThat(result.isFinalNode()).isTrue();
		float[][] batchedLogits = result.logitsPerPosition(VOCAB);

		for (int d = 0; d < DRAFT_LEN; d++) {
			for (int v = 0; v < VOCAB; v++) {
				assertThat(batchedLogits[d][v]).as("position %d, vocab %d", d, v)
						.isCloseTo(serialLogits[d][v], within(1e-4f));
			}
		}
	}

	@Test
	@DisplayName("intermediate-node verify keeps every position's activations, not just the last")
	void intermediate_node_keeps_all_positions() {
		// First shard only: no output projection, so the result carries activations.
		LlamaTransformerHandler firstShard = LlamaTransformerHandler.newTestInstance(VOCAB, H, NH, KVH, LAYERS, 0,
				LAYERS, true, false, null);
		ShardAssignment a = new ShardAssignment("n1", "localhost", 0, 0, LAYERS, true, false);
		ShardContext ctx = ShardContext.from(a, VOCAB, H, NH);

		int[] draft = tokens(DRAFT_LEN, PROMPT_LEN);
		prefill(firstShard, ctx, "activations", PROMPT_LEN);
		VerifyBatchResult result = firstShard.forwardVerify(
				BatchForwardRequest.withTokens("activations", draft, PROMPT_LEN), ctx);

		assertThat(result.isFinalNode()).isFalse();
		assertThat(result.activations()).hasSize(DRAFT_LEN * H);
	}
}
