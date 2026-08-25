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
class StubForwardPassHandler implements ForwardPassHandler {

    ForwardRequest lastRequest;
    ShardContext lastContext;

    private final int winnerToken;
    private final int hiddenDim; // used only by embedToken(); 0 = "not configured"
    private final AtomicInteger callCount = new AtomicInteger();

    StubForwardPassHandler() {
        this(0, 0);
    }

    StubForwardPassHandler(int winnerToken) {
        this(winnerToken, 0);
    }

    StubForwardPassHandler(int winnerToken, int hiddenDim) {
        this.winnerToken = winnerToken;
        this.hiddenDim = hiddenDim;
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

    /** Deterministic fake embedding: emb[d] = tokenId * 1000 + d. Requires the
     * hiddenDim constructor param to have been set. */
    @Override
    public float[] embedToken(int tokenId) {
        if (hiddenDim <= 0) {
            throw new IllegalStateException(
                    "StubForwardPassHandler.embedToken() called but hiddenDim was not configured — "
                    + "use the StubForwardPassHandler(int winnerToken, int hiddenDim) constructor");
        }
        float[] emb = new float[hiddenDim];
        for (int d = 0; d < hiddenDim; d++) {
            emb[d] = tokenId * 1000f + d;
        }
        return emb;
    }

    int callCount() {
        return callCount.get();
    }
}
