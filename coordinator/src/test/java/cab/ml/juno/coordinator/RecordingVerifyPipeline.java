package cab.ml.juno.coordinator;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import cab.ml.juno.node.InferencePipeline;

/**
 * Test double for {@link InferencePipeline} whose predicted token at a given
 * absolute sequence position is a fixed, deterministic function of that
 * position — not of call order or call count.
 *
 * <p>This matters for speculative-decode tests specifically: a batched
 * {@link #verifyDraft} call computes logits for every position in its window
 * up front, including positions whose result ends up discarded by the caller
 * (the draft diverged before reaching them). A real model recomputing the same
 * position later (e.g. via a fresh {@link #forward} call after divergence)
 * gives the same answer, since it depends on context/position, not on "which
 * call this is" — a call-counting test double would instead treat that
 * discarded computation as having "consumed" a token that a later real call
 * then never gets to see, corrupting the comparison against non-speculative
 * decoding. Scripting by position avoids that pitfall entirely.
 *
 * @param scriptByPosition absolute sequence position (0-based, counting from
 *                         the start of the prompt) of the token BEING
 *                         PREDICTED -&gt; token id to predict there. Positions
 *                         not present return {@link #DEFAULT_TOKEN}.
 */
record RecordingVerifyPipeline(Map<Integer, Integer> scriptByPosition, List<Integer> verifyDraftWindowSizes)
		implements InferencePipeline {

	static final int VOCAB_SIZE = 1000;
	static final int DEFAULT_TOKEN = 42;

	RecordingVerifyPipeline(Map<Integer, Integer> scriptByPosition) {
		this(scriptByPosition, new ArrayList<>());
	}

	private int scriptedTokenAt(int position) {
		return scriptByPosition.getOrDefault(position, DEFAULT_TOKEN);
	}

	private static float[] logitsFor(int winner) {
		float[] logits = new float[VOCAB_SIZE];
		logits[winner] = 100.0f;
		return logits;
	}

	@Override
	public float[] forward(String requestId, int[] tokens, int startPos) {
		return logitsFor(scriptedTokenAt(startPos + 1));
	}

	@Override
	public float[][] verifyDraft(String requestId, int[] draftTokens, int startPosition) {
		verifyDraftWindowSizes.add(draftTokens.length);
		float[][] logits = new float[draftTokens.length][];
		for (int i = 0; i < draftTokens.length; i++) {
			logits[i] = logitsFor(scriptedTokenAt(startPosition + i + 1));
		}
		return logits;
	}

	@Override
	public int vocabSize() {
		return VOCAB_SIZE;
	}
}
