package cab.ml.juno.coordinator;

import java.util.Arrays;
import java.util.Optional;

import cab.ml.juno.node.ForwardPassHandler;
import cab.ml.juno.node.ForwardRequest;
import cab.ml.juno.node.ForwardResult;
import cab.ml.juno.node.ShardContext;

/**
 * Single-stage {@link ForwardPassHandler} test double whose embedding hidden
 * vector is filled with {@code startPosition + 1} at every element — lets
 * {@link EmbeddingsHandlerTest} tell {@code mean} / {@code cls} / {@code last}
 * pooling apart by their actual value instead of only their shape, unlike the
 * position-invariant {@code CyclicForwardPassHandler} used elsewhere.
 *
 * <p>{@link #forward} always picks token 42 as the winner, purely so the
 * regular chat-completions route keeps working in the same test server
 * (proving {@code /v1/embeddings} does not break it) — chat generation
 * itself is not what this double is testing.
 */
final class PositionValueForwardPassHandler implements ForwardPassHandler {

	private static final int WINNER_TOKEN = 42;

	@Override
	public ForwardResult forward(ForwardRequest request, ShardContext context) {
		float[] logits = new float[context.vocabSize()];
		logits[WINNER_TOKEN] = 100.0f;
		return ForwardResult.logits(request.requestId(), logits, 0);
	}

	@Override
	public boolean isReady() {
		return true;
	}

	@Override
	public Optional<float[]> lastRmsHiddenForEmbedding(ForwardRequest request, ShardContext context) {
		float[] v = new float[context.hiddenDim()];
		Arrays.fill(v, (float) (request.startPosition() + 1));
		return Optional.of(v);
	}
}
