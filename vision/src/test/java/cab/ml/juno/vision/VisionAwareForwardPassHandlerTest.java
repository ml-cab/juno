package cab.ml.juno.vision;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.node.ForwardRequest;
import cab.ml.juno.node.ShardContext;

/**
 * Tests for VisionAwareForwardPassHandler.
 *
 * Uses StubForwardPassHandler — a test double that lives in vision's own
 * test sources. No dependency on node:tests classifier (which would create
 * a Maven reactor cycle).
 */
@DisplayName("VisionAwareForwardPassHandler — embedding splice and delegation")
class VisionAwareForwardPassHandlerTest {

    private static final int IMAGE_TOKEN_ID = 32000;
    private static final int HIDDEN_DIM     = 64;
    private static final int VOCAB_SIZE     = 200;
    private static final int NUM_HEADS      = 2;

    // ShardContext for the first node (hasEmbeddings=true)
    private static final ShardContext FIRST_NODE_CTX =
            new ShardContext("n1", 0, 11, true, false, VOCAB_SIZE, HIDDEN_DIM, NUM_HEADS);

    // ShardContext for a subsequent node (hasEmbeddings=false)
    private static final ShardContext MID_NODE_CTX =
            new ShardContext("n2", 11, 22, false, false, VOCAB_SIZE, HIDDEN_DIM, NUM_HEADS);

    private StubForwardPassHandler inner;
    private VisionAwareForwardPassHandler handler;

    @BeforeEach
    void setup() {
        inner   = new StubForwardPassHandler();
        handler = new VisionAwareForwardPassHandler(inner, IMAGE_TOKEN_ID, HIDDEN_DIM);
    }

    // ── Text-only pass-through ────────────────────────────────────────────────

    @Test
    @DisplayName("text-only request: delegated with original token-ids request unchanged")
    void text_only_request_delegates_unchanged() {
        ForwardRequest req = ForwardRequest.withTokens("req-text", new int[]{1, 2, 3}, 0);

        handler.forward(req, FIRST_NODE_CTX);

        // No vision embeddings registered → inner handler called with the original token request
        assertThat(inner.lastRequest).isSameAs(req);
    }

    @Test
    @DisplayName("non-first node: passes through regardless of registered embeddings")
    void non_first_node_passes_through() {
        float[][] patches = buildPatches(4, HIDDEN_DIM);
        handler.registerVisionEmbeddings("req-mid", patches);

        float[] act = new float[HIDDEN_DIM];
        ForwardRequest req = ForwardRequest.withActivations("req-mid", act, 5);

        handler.forward(req, MID_NODE_CTX);

        assertThat(inner.lastRequest).isSameAs(req);
    }

    // ── Vision embedding splice ───────────────────────────────────────────────

    @Test
    @DisplayName("image token at last position: inner handler receives withActivations request")
    void image_token_replaced_with_patch_embedding() {
        float[][] patches = buildPatches(1, HIDDEN_DIM);
        // Mark patch 0 with sentinel values
        for (int d = 0; d < HIDDEN_DIM; d++) patches[0][d] = d + 1.0f;

        handler.registerVisionEmbeddings("req-img", patches);

        // Token sequence: [1, IMAGE_TOKEN_ID] → last token is image → use patch[0]
        ForwardRequest req = ForwardRequest.withTokens("req-img",
                new int[]{1, IMAGE_TOKEN_ID}, 0);

        handler.forward(req, FIRST_NODE_CTX);

        // Inner handler must have received an activations request (not token request)
        assertThat(inner.lastRequest.isFirstNode()).isFalse();

        // Activations must match patch[0] sentinel values
        float[] act = inner.lastRequest.activations();
        assertThat(act).hasSize(HIDDEN_DIM);
        for (int d = 0; d < HIDDEN_DIM; d++) {
            assertThat(act[d]).isEqualTo(d + 1.0f);
        }
    }

    @Test
    @DisplayName("second image token selects patch[1], not patch[0]")
    void second_image_token_selects_correct_patch() {
        float[][] patches = buildPatches(2, HIDDEN_DIM);
        patches[0][0] = 100f;
        patches[1][0] = 200f;

        handler.registerVisionEmbeddings("req-p1", patches);

        // Sequence: [IMAGE_TOKEN_ID, IMAGE_TOKEN_ID] → last token is 2nd image → patch[1]
        ForwardRequest req = ForwardRequest.withTokens("req-p1",
                new int[]{IMAGE_TOKEN_ID, IMAGE_TOKEN_ID}, 1);

        handler.forward(req, FIRST_NODE_CTX);

        assertThat(inner.lastRequest.activations()[0]).isEqualTo(200f);
    }

    @Test
    @DisplayName("patch vector is defensively copied — mutation after forward does not affect inner handler's activations")
    void patch_vector_is_defensively_copied() {
        float[][] patches = buildPatches(1, HIDDEN_DIM);
        patches[0][0] = 77f;

        handler.registerVisionEmbeddings("req-copy", patches);
        ForwardRequest req = ForwardRequest.withTokens("req-copy",
                new int[]{IMAGE_TOKEN_ID}, 0);
        handler.forward(req, FIRST_NODE_CTX);

        float[] activations = inner.lastRequest.activations();
        float original = activations[0];

        // Mutate the stored patch after the call
        patches[0][0] = 999f;

        // Already-passed activations must be unaffected
        assertThat(activations[0]).isEqualTo(original);
    }

    // ── Release ───────────────────────────────────────────────────────────────

    @Test
    @DisplayName("after release, request falls back to text-only path")
    void after_release_text_path_used() {
        float[][] patches = buildPatches(1, HIDDEN_DIM);
        handler.registerVisionEmbeddings("req-rel", patches);
        handler.releaseVisionEmbeddings("req-rel");

        // Now forwards as text-only
        ForwardRequest req = ForwardRequest.withTokens("req-rel",
                new int[]{IMAGE_TOKEN_ID}, 0);
        handler.forward(req, FIRST_NODE_CTX);

        // Inner handler receives the original token request (no activation substitution)
        assertThat(inner.lastRequest).isSameAs(req);
    }

    @Test
    @DisplayName("releaseVisionEmbeddings on unknown id is safe (no exception)")
    void release_unknown_id_safe() {
        handler.releaseVisionEmbeddings("nonexistent-request-id");
        // No exception expected
    }

    // ── Delegation of isReady ─────────────────────────────────────────────────

    @Test
    @DisplayName("isReady delegates to inner handler")
    void is_ready_delegates() {
        assertThat(handler.isReady()).isTrue();
    }

    // ── Construction guards ───────────────────────────────────────────────────

    @Test
    @DisplayName("null textHandler throws IllegalArgumentException")
    void null_text_handler_rejected() {
        assertThatThrownBy(() -> new VisionAwareForwardPassHandler(null, IMAGE_TOKEN_ID, HIDDEN_DIM))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("textHandler");
    }

    @Test
    @DisplayName("hiddenDim < 1 throws IllegalArgumentException")
    void invalid_hidden_dim_rejected() {
        assertThatThrownBy(() -> new VisionAwareForwardPassHandler(inner, IMAGE_TOKEN_ID, 0))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("hiddenDim");
    }

    @Test
    @DisplayName("registerVisionEmbeddings with blank requestId throws")
    void blank_request_id_rejected() {
        float[][] patches = buildPatches(1, HIDDEN_DIM);
        assertThatThrownBy(() -> handler.registerVisionEmbeddings("  ", patches))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    @DisplayName("registerVisionEmbeddings with empty patches throws")
    void empty_patches_rejected() {
        assertThatThrownBy(() -> handler.registerVisionEmbeddings("req", new float[0][]))
                .isInstanceOf(IllegalArgumentException.class);
    }

    // ── Helper ────────────────────────────────────────────────────────────────

    private static float[][] buildPatches(int count, int dim) {
        float[][] patches = new float[count][dim];
        for (int i = 0; i < count; i++)
            for (int d = 0; d < dim; d++)
                patches[i][d] = 0.1f * i;
        return patches;
    }
}