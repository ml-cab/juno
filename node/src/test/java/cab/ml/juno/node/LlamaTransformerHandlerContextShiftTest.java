package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.offset;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.DenseKvTensor;
import cab.ml.juno.kvcache.KvPageSizeOptions;
import cab.ml.juno.kvcache.ServeScheduleOptions;
import cab.ml.juno.kvcache.SessionKvTensor;

/**
 * A shifted request must continue exactly as a request holding the expected
 * shifted KV ({@link ContextShiftOracle}): same next-token logits, on dense
 * (static) and paged (continuous) KV.
 */
@DisplayName("LlamaTransformerHandler — context shift")
class LlamaTransformerHandlerContextShiftTest {

	private static final int VOCAB = 64;
	private static final int H = 32;
	private static final int HEADS = 4;
	private static final int KV_HEADS = 2;
	private static final int LAYERS = 3;
	private static final int N = 40;
	private static final int KEEP = 4;
	private static final int DISCARD = 18;
	/** Wide enough that attention depends on where each key sits; see the negative control. */
	private static final float WEIGHT_RANGE = 2.0f;

	private final String prevSched = System.getProperty(ServeScheduleOptions.ENV);
	private final String prevPage = System.getProperty(KvPageSizeOptions.ENV);

	@AfterEach
	void restore() {
		restoreProp(ServeScheduleOptions.ENV, prevSched);
		restoreProp(KvPageSizeOptions.ENV, prevPage);
	}

	private static int token(int i) {
		return (i * 7 + 3) % VOCAB;
	}

	private static LlamaTransformerHandler handler() {
		return LlamaTransformerHandler.newTestInstance(VOCAB, H, HEADS, KV_HEADS, LAYERS, 0, LAYERS, true, true, null,
				WEIGHT_RANGE);
	}

	private static ShardContext ctx() {
		return new ShardContext("n0", 0, LAYERS, true, true, VOCAB, H, HEADS);
	}

	private static float[] step(ForwardPassHandler h, String id, int tok, int pos) {
		return h.forward(ForwardRequest.withTokens(id, new int[] { tok }, pos), ctx()).logits();
	}

	/**
	 * The shifted request's next logits against an oracle request holding the
	 * expected shifted KV ({@link ContextShiftOracle}); both are first fed the same
	 * N tokens, so the oracle starts from identical KV.
	 */
	private static void assertShiftMatchesOracle() {
		LlamaTransformerHandler h = handler();
		for (int p = 0; p < N; p++) {
			step(h, "s", token(p), p);
			step(h, "o", token(p), p);
		}
		int kvDim = (H / HEADS) * KV_HEADS;
		int headDim = H / HEADS;
		SessionKvTensor[][] o = h.hostKv("o");
		float[][] k = ContextShiftOracle.expected(ContextShiftOracle.snapshot(o[0], N), kvDim, N, KEEP, DISCARD, true,
				(row, pos) -> LoraTrainingMath.ropeBackward(row, pos, KV_HEADS, headDim, 10000f, RopePairing.ADJACENT),
				(row, pos) -> LlamaTransformerHandler.rope(row, pos, KV_HEADS, headDim, 10000f, RopePairing.ADJACENT));
		float[][] v = ContextShiftOracle.expected(ContextShiftOracle.snapshot(o[1], N), kvDim, N, KEEP, DISCARD, false,
				null, null);
		ContextShiftOracle.inject(o[0], k, kvDim, N - DISCARD);
		ContextShiftOracle.inject(o[1], v, kvDim, N - DISCARD);
		float[] expected = step(h, "o", token(N), N - DISCARD);

		h.shiftKv("s", N, KEEP, DISCARD);
		float[] afterShift = step(h, "s", token(N), N - DISCARD);

		for (int i = 0; i < VOCAB; i++)
			assertThat(afterShift[i]).as("logit %d", i).isCloseTo(expected[i], offset(1e-4f));
	}

	@Test
	@DisplayName("dense KV: shifted request matches the oracle KV")
	void denseShiftMatchesFresh() {
		System.setProperty(ServeScheduleOptions.ENV, "static");
		System.clearProperty(KvPageSizeOptions.ENV);
		assertShiftMatchesOracle();
	}

	@Test
	@DisplayName("paged KV: shifted request matches the oracle KV")
	void pagedShiftMatchesFresh() {
		System.setProperty(ServeScheduleOptions.ENV, "continuous");
		System.setProperty(KvPageSizeOptions.ENV, "4");
		assertShiftMatchesOracle();
	}

	@Test
	@DisplayName("negative control: keys moved without re-rotation give measurably different logits")
	void unrotatedKeysAreDetected() {
		System.setProperty(ServeScheduleOptions.ENV, "static");
		System.clearProperty(KvPageSizeOptions.ENV);
		LlamaTransformerHandler h = handler();
		for (int p = 0; p < N; p++) {
			step(h, "s", token(p), p);
			step(h, "u", token(p), p);
		}
		int kvDim = (H / HEADS) * KV_HEADS;
		SessionKvTensor[][] u = h.hostKv("u");
		ContextShiftOracle.inject(u[0], ContextShiftOracle.expected(ContextShiftOracle.snapshot(u[0], N), kvDim, N,
				KEEP, DISCARD, false, null, null), kvDim, N - DISCARD);
		ContextShiftOracle.inject(u[1], ContextShiftOracle.expected(ContextShiftOracle.snapshot(u[1], N), kvDim, N,
				KEEP, DISCARD, false, null, null), kvDim, N - DISCARD);
		float[] unrotated = step(h, "u", token(N), N - DISCARD);
		h.shiftKv("s", N, KEEP, DISCARD);
		float[] shifted = step(h, "s", token(N), N - DISCARD);
		double maxDiff = 0;
		for (int i = 0; i < VOCAB; i++)
			maxDiff = Math.max(maxDiff, Math.abs(shifted[i] - unrotated[i]));
		assertThat(maxDiff).as("the 1e-4 tolerance must be able to see a missing rotation").isGreaterThan(1e-3);
	}

	@Test
	@DisplayName("context limit is the KV cap for the LLaMA family")
	void contextLimitIsKvCap() {
		assertThat(handler().contextLimit()).isEqualTo(DenseKvTensor.MAX_SEQ_LEN);
	}

	@Test
	@DisplayName("shifting a request with no KV fails closed")
	void unknownRequestFailsClosed() {
		assertThatThrownBy(() -> handler().shiftKv("nobody", 10, 2, 4)).isInstanceOf(IllegalStateException.class)
				.hasMessageContaining("nobody");
	}

	@Test
	@DisplayName("a handler without context-shift support fails closed")
	void unsupportedHandlerFailsClosed() {
		ForwardPassHandler bare = new ForwardPassHandler() {
			@Override
			public ForwardResult forward(ForwardRequest request, ShardContext context) {
				throw new AssertionError();
			}

			@Override
			public boolean isReady() {
				return true;
			}
		};
		assertThatThrownBy(() -> bare.shiftKv("r", 10, 2, 4)).isInstanceOf(UnsupportedOperationException.class)
				.hasMessageContaining("context shift");
	}

	private static void restoreProp(String key, String prev) {
		if (prev == null)
			System.clearProperty(key);
		else
			System.setProperty(key, prev);
	}
}
