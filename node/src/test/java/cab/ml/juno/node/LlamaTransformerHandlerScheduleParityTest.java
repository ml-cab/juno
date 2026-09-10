package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.offset;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.KvPageSizeOptions;
import cab.ml.juno.kvcache.ServeScheduleOptions;

/**
 * Static dense and continuous paged KV paths must agree on greedy logits.
 */
@DisplayName("LlamaTransformerHandler — schedule KV parity")
class LlamaTransformerHandlerScheduleParityTest {

	private static final int VOCAB = 64;
	private static final int H = 32;
	private static final int HEADS = 4;
	private static final int KV_HEADS = 2;
	private static final int LAYERS = 4;

	private String prevSched;
	private String prevPage;

	@AfterEach
	void restore() {
		restoreProp(ServeScheduleOptions.ENV, prevSched);
		restoreProp(KvPageSizeOptions.ENV, prevPage);
	}

	@Test
	@DisplayName("continuous paged forward matches static dense (f16)")
	void continuous_matches_static() {
		prevSched = System.getProperty(ServeScheduleOptions.ENV);
		prevPage = System.getProperty(KvPageSizeOptions.ENV);

		ShardContext ctx = new ShardContext("n0", 0, LAYERS, true, true, VOCAB, H, HEADS);
		ForwardRequest req = ForwardRequest.withTokens("r1", new int[] { 3, 7, 11 }, 1);

		System.setProperty(ServeScheduleOptions.ENV, "static");
		System.clearProperty(KvPageSizeOptions.ENV);
		float[] dense = LlamaTransformerHandler.newTestInstance(
				VOCAB, H, HEADS, KV_HEADS, LAYERS, 0, LAYERS, true, true, null)
				.forward(req, ctx).logits();

		System.setProperty(ServeScheduleOptions.ENV, "continuous");
		System.setProperty(KvPageSizeOptions.ENV, "4");
		float[] paged = LlamaTransformerHandler.newTestInstance(
				VOCAB, H, HEADS, KV_HEADS, LAYERS, 0, LAYERS, true, true, null)
				.forward(req, ctx).logits();

		assertThat(paged.length).isEqualTo(VOCAB);
		for (int i = 0; i < VOCAB; i++)
			assertThat(paged[i]).isCloseTo(dense[i], offset(1e-5f));
	}

	private static void restoreProp(String key, String prev) {
		if (prev == null)
			System.clearProperty(key);
		else
			System.setProperty(key, prev);
	}
}
