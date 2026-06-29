package cab.ml.juno.vision;

import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;

import cab.ml.juno.node.ForwardPassHandler;
import cab.ml.juno.node.ForwardRequest;
import cab.ml.juno.node.ForwardResult;
import cab.ml.juno.node.ShardContext;

/**
 * Minimal ForwardPassHandler test double for the vision module.
 *
 * Replaces the dependency on node's test-jar (CyclicForwardPassHandler),
 * which would create a Maven reactor cycle: node -> vision -> node.
 *
 * Records the last ForwardRequest and ShardContext seen, returns a
 * zero-filled activations result for intermediate nodes and a logits
 * result with full mass on {@code winnerToken} for the last node.
 */
final class StubForwardPassHandler implements ForwardPassHandler {

    ForwardRequest lastRequest;
    ShardContext lastContext;

    private final int winnerToken;
    private final AtomicInteger callCount = new AtomicInteger();

    StubForwardPassHandler() {
        this.winnerToken = 0;
    }

    StubForwardPassHandler(int winnerToken) {
        this.winnerToken = winnerToken;
    }

    @Override
    public ForwardResult forward(ForwardRequest request, ShardContext context) {
        this.lastRequest = request;
        this.lastContext = context;
        callCount.incrementAndGet();

        if (context.hasOutputProjection()) {
            float[] logits = new float[context.vocabSize()];
            logits[winnerToken] = 100.0f;
            return ForwardResult.logits(request.requestId(), logits, 0L);
        }
        float[] activations = new float[context.hiddenDim()];
        return ForwardResult.activations(request.requestId(), activations, 0L);
    }

    @Override
    public boolean isReady() {
        return true;
    }

    @Override
    public Optional<float[]> lastRmsHiddenForEmbedding(ForwardRequest request, ShardContext context) {
        return Optional.empty();
    }

    int callCount() {
        return callCount.get();
    }
}