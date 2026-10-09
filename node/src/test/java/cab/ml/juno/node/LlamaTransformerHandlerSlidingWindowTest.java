package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.offset;

import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.KvPageSizeOptions;
import cab.ml.juno.kvcache.ServeScheduleOptions;
import cab.ml.juno.kvcache.SessionKvTensor;

/**
 * Sliding-window attention on the LLaMA handler (synthetic model, CPU), with a
 * patterned window: period 2 over four layers, so layers 0 and 2 attend within
 * {@link #WINDOW} keys and layers 1 and 3 globally. The oracle is the KV itself:
 * overwriting the rows a windowed layer must not read leaves the next token's
 * logits bit-identical, and overwriting the same rows of a global layer moves them.
 */
@DisplayName("LlamaTransformerHandler - sliding-window attention")
class LlamaTransformerHandlerSlidingWindowTest {

	private static final int VOCAB = 64;
	private static final int H = 32;
	private static final int HEADS = 4;
	private static final int KV_HEADS = 2;
	private static final int LAYERS = 4;
	private static final int N = 20;
	private static final int WINDOW = 6;
	private static final float WEIGHT_RANGE = 2.0f;
	private static final SlidingWindow PERIOD_TWO = SlidingWindow.of(WINDOW, new boolean[] { true, false, true, false });

	private final String prevSched = System.getProperty(ServeScheduleOptions.ENV);
	private final String prevPage = System.getProperty(KvPageSizeOptions.ENV);

	@AfterEach
	void restore() {
		restoreProp(ServeScheduleOptions.ENV, prevSched);
		restoreProp(KvPageSizeOptions.ENV, prevPage);
	}

	private static void restoreProp(String key, String prev) {
		if (prev == null)
			System.clearProperty(key);
		else
			System.setProperty(key, prev);
	}

	private static void schedule(String kv) {
		if (kv.equals("paged")) {
			System.setProperty(ServeScheduleOptions.ENV, "continuous");
			System.setProperty(KvPageSizeOptions.ENV, "4");
		} else {
			System.setProperty(ServeScheduleOptions.ENV, "static");
			System.clearProperty(KvPageSizeOptions.ENV);
		}
	}

	private static LlamaTransformerHandler handler(SlidingWindow window) {
		return LlamaTransformerHandler.newTestInstance(VOCAB, H, HEADS, KV_HEADS, LAYERS, 0, LAYERS, true, true, null,
				WEIGHT_RANGE, window);
	}

	private static ShardContext ctx() {
		return new ShardContext("n0", 0, LAYERS, true, true, VOCAB, H, HEADS);
	}

	private static int token(int i) {
		return (i * 7 + 3) % VOCAB;
	}

	private static int[] prompt() {
		int[] t = new int[N];
		for (int i = 0; i < N; i++)
			t[i] = token(i);
		return t;
	}

	private static float[] step(ForwardPassHandler h, String id, int tok, int pos) {
		return h.forward(ForwardRequest.withTokens(id, new int[] { tok }, pos), ctx()).logits();
	}

	private static float[] prefill(ForwardPassHandler h, String id) {
		return h.forwardBatch(BatchForwardRequest.withTokens(id, prompt(), 0), ctx()).lastLogits();
	}

	/** Overwrites K and V rows {@code [0, rows)} of {@code layer} with large finite values. */
	private static void overwrite(LlamaTransformerHandler h, String id, int layer, int rows) {
		SessionKvTensor[][] kv = h.hostKv(id);
		int kvDim = (H / HEADS) * KV_HEADS;
		Random rng = new Random(layer * 31L + rows);
		float[] row = new float[kvDim];
		for (int pos = 0; pos < rows; pos++)
			for (int which = 0; which < 2; which++) {
				for (int i = 0; i < kvDim; i++)
					row[i] = (float) rng.nextGaussian() * 50f;
				kv[which][layer].writeToken(pos, row);
			}
	}

	private static double maxAbsDiff(float[] a, float[] b) {
		double m = 0;
		for (int i = 0; i < a.length; i++)
			m = Math.max(m, Math.abs(a[i] - b[i]));
		return m;
	}

	@ParameterizedTest(name = "{0} KV")
	@ValueSource(strings = { "dense", "paged" })
	@DisplayName("decode: a windowed layer never reads the rows before its window; a global layer does")
	void decodeReadsOnlyTheWindowOnWindowedLayers(String kv) {
		schedule(kv);
		LlamaTransformerHandler h = handler(PERIOD_TWO);
		for (String id : new String[] { "a", "w0", "w2", "g1", "g3" })
			prefill(h, id);
		// The next token sits at position N, so its context is N + 1 keys and the window starts at N + 1 - WINDOW.
		int outside = N + 1 - WINDOW;
		overwrite(h, "w0", 0, outside);
		overwrite(h, "w2", 2, outside);
		overwrite(h, "g1", 1, outside);
		overwrite(h, "g3", 3, outside);
		float[] clean = step(h, "a", token(N), N);
		assertThat(step(h, "w0", token(N), N)).as("layer 0 is windowed").containsExactly(clean);
		assertThat(step(h, "w2", token(N), N)).as("layer 2 is windowed").containsExactly(clean);
		assertThat(maxAbsDiff(step(h, "g1", token(N), N), clean)).as("layer 1 is global").isGreaterThan(1e-3);
		assertThat(maxAbsDiff(step(h, "g3", token(N), N), clean)).as("layer 3 is global").isGreaterThan(1e-3);
	}

	@ParameterizedTest(name = "{0} KV")
	@ValueSource(strings = { "dense", "paged" })
	@DisplayName("prefill windows each row: a prefilled prompt matches the same prompt decoded token by token")
	void prefillMatchesTokenByToken(String kv) {
		schedule(kv);
		LlamaTransformerHandler h = handler(PERIOD_TWO);
		float[] batched = prefill(h, "p");
		float[] stepped = null;
		for (int p = 0; p < N; p++)
			stepped = step(h, "s", token(p), p);
		for (int i = 0; i < VOCAB; i++)
			assertThat(batched[i]).as("logit %d", i).isCloseTo(stepped[i], offset(1e-4f));

		float[] unwindowed = prefill(handler(SlidingWindow.NONE), "u");
		assertThat(maxAbsDiff(batched, unwindowed)).as("the window changes a prompt longer than it")
				.isGreaterThan(1e-3);
	}

	@Test
	@DisplayName("multi-stream decode windows each stream at its own position")
	void multiDecodeMatchesSingleDecode() {
		schedule("dense");
		LlamaTransformerHandler h = handler(PERIOD_TWO);
		int[] shortPrompt = new int[N - 7];
		System.arraycopy(prompt(), 0, shortPrompt, 0, shortPrompt.length);
		for (String id : new String[] { "a", "b" })
			h.forwardBatch(BatchForwardRequest.withTokens(id, id.equals("a") ? prompt() : shortPrompt, 0), ctx());
		for (String id : new String[] { "sa", "sb" })
			h.forwardBatch(BatchForwardRequest.withTokens(id, id.equals("sa") ? prompt() : shortPrompt, 0), ctx());
		float[][] multi = h.forwardMultiDecode(MultiDecodeForwardRequest.withTokens(List.of("a", "b"),
				new int[] { token(N), token(N - 7) }, new int[] { N, N - 7 }), ctx()).logits();
		float[] a = step(h, "sa", token(N), N);
		float[] b = step(h, "sb", token(N - 7), N - 7);
		for (int i = 0; i < VOCAB; i++) {
			assertThat(multi[0][i]).as("stream a logit %d", i).isCloseTo(a[i], offset(1e-4f));
			assertThat(multi[1][i]).as("stream b logit %d", i).isCloseTo(b[i], offset(1e-4f));
		}
	}

	@Test
	@DisplayName("a window no shorter than the context is bit-identical to no window (prefill and decode)")
	void wideWindowIsNoOp() {
		schedule("dense");
		LlamaTransformerHandler none = handler(SlidingWindow.NONE);
		LlamaTransformerHandler wide = handler(SlidingWindow.of(4096, null));
		assertThat(prefill(wide, "x")).containsExactly(prefill(none, "x"));
		for (int p = N; p < N + 4; p++)
			assertThat(step(wide, "x", token(p), p)).as("decode at " + p).containsExactly(step(none, "x", token(p), p));
	}

	@Test
	@DisplayName("a pipeline shard takes its layers' windows by global index")
	void shardWindowsByGlobalLayer() {
		LlamaTransformerHandler shard = LlamaTransformerHandler.newTestInstance(VOCAB, H, HEADS, KV_HEADS, LAYERS, 1, 3,
				false, false, null, WEIGHT_RANGE, PERIOD_TWO);
		// Global layers 1 (global) and 2 (windowed).
		assertThat(shard.attentionWindows()).containsExactly(0, WINDOW);
		assertThat(handler(PERIOD_TWO).attentionWindows()).containsExactly(WINDOW, 0, WINDOW, 0);
	}
}
