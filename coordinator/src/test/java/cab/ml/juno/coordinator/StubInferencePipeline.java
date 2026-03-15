package cab.ml.juno.coordinator;

import java.util.concurrent.atomic.AtomicInteger;

import cab.ml.juno.node.InferencePipeline;

/**
 * Test double for InferencePipeline.
 *
 * Returns a fixed logit distribution on each forward() call. By default puts
 * all probability mass on token index 42 (arbitrary non-special token).
 * Configurable to return a specific "winner" token, or to cycle through a
 * sequence.
 */
final class StubInferencePipeline implements InferencePipeline {

	static final int VOCAB_SIZE = 1000;
	static final int DEFAULT_TOKEN = 42;

	private final int[] tokenSequence; // if set, returns these in order
	private final AtomicInteger callCount = new AtomicInteger(0);

	/** Always returns logits pointing at DEFAULT_TOKEN. */
	StubInferencePipeline() {
		this.tokenSequence = null;
	}

	/** Cycles through the given token sequence, then returns DEFAULT_TOKEN. */
	StubInferencePipeline(int... tokenSequence) {
		this.tokenSequence = tokenSequence;
	}

	@Override
	public float[] forward(String requestId, int[] tokens, int startPos) {
		float[] logits = new float[VOCAB_SIZE];

		int winnerToken;
		if (tokenSequence != null) {
			int idx = callCount.getAndIncrement();
			winnerToken = idx < tokenSequence.length ? tokenSequence[idx] : DEFAULT_TOKEN;
		} else {
			winnerToken = DEFAULT_TOKEN;
		}

		logits[winnerToken] = 100.0f; // overwhelmingly high logit → always selected
		return logits;
	}

	@Override
	public int vocabSize() {
		return VOCAB_SIZE;
	}

	int callCount() {
		return callCount.get();
	}
}
