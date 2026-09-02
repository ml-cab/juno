package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.util.Arrays;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.registry.ShardAssignment;
import cab.ml.juno.registry.ShardMap;

/**
 * Chunked {@code prefillBatch} windows must leave the same KV state as one
 * full-window prefill — first-decode logits must match within tolerance.
 */
@DisplayName("LlamaTransformerHandler — prefill chunk parity")
class LlamaTransformerHandlerPrefillChunkParityTest {

	private static final int VOCAB = 256;
	private static final int H = 32;
	private static final int NH = 4;
	private static final int KVH = 4;
	private static final int LAYERS = 2;
	private static final int PROMPT_LEN = 65; // 64 prefill positions + 1 decode token
	private static final int CHUNK = 32;

	private LlamaTransformerHandler handler() {
		return LlamaTransformerHandler.newTestInstance(VOCAB, H, NH, KVH, LAYERS, 0, LAYERS, true, true, null);
	}

	private ShardContext ctx() {
		ShardAssignment a = new ShardAssignment("n1", "localhost", 0, 0, LAYERS, true, true);
		return ShardContext.from(a, VOCAB, H, NH);
	}

	private int[] promptTokens() {
		int[] ids = new int[PROMPT_LEN];
		for (int i = 0; i < ids.length; i++)
			ids[i] = 3 + (i % 17);
		return ids;
	}

	private float[] firstDecodeLogits(LlamaTransformerHandler handler, ShardContext ctx, String requestId,
			int[] promptIds, int chunkSize) {
		int prefillEnd = promptIds.length - 1;
		for (int pos = 0; pos < prefillEnd;) {
			int chunkEnd = Math.min(pos + chunkSize, prefillEnd);
			int[] window = Arrays.copyOfRange(promptIds, pos, chunkEnd);
			handler.forwardBatch(BatchForwardRequest.withTokens(requestId, window, pos), ctx);
			pos = chunkEnd;
		}
		int lastPromptTok = promptIds[prefillEnd];
		return handler.forward(ForwardRequest.withTokens(requestId, new int[] { lastPromptTok }, prefillEnd), ctx)
				.logits();
	}

	@Test
	@DisplayName("chunked forwardBatch prefill matches single-window prefill logits")
	void chunked_prefill_matches_whole_window() {
		int[] prompt = promptTokens();
		ShardContext ctx = ctx();

		float[] whole = firstDecodeLogits(handler(), ctx, "whole", prompt, PROMPT_LEN);
		float[] chunked = firstDecodeLogits(handler(), ctx, "chunked", prompt, CHUNK);

		assertThat(chunked.length).isEqualTo(VOCAB);
		for (int v = 0; v < VOCAB; v++)
			assertThat(chunked[v]).as("vocab %d", v).isCloseTo(whole[v], within(1e-4f));
	}

	@Test
	@DisplayName("prefill-batch size 1 matches whole-window prefill logits")
	void per_token_chunks_match_whole_window() {
		int[] prompt = promptTokens();
		ShardContext ctx = ctx();

		float[] whole = firstDecodeLogits(handler(), ctx, "whole-1", prompt, PROMPT_LEN);
		float[] perTok = firstDecodeLogits(handler(), ctx, "one-1", prompt, 1);

		for (int v = 0; v < VOCAB; v++)
			assertThat(perTok[v]).as("vocab %d", v).isCloseTo(whole[v], within(1e-4f));
	}
}
