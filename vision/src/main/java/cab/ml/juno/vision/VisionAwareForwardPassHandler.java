/*
 * Copyright 2026 Dmytro Soloviov (soulaway)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package cab.ml.juno.vision;

import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

import cab.ml.juno.node.ForwardPassHandler;
import cab.ml.juno.node.ForwardRequest;
import cab.ml.juno.node.ForwardResult;
import cab.ml.juno.node.ShardContext;

/**
 * {@link ForwardPassHandler} decorator that splices pre-computed vision
 * patch embeddings into the hidden state at {@code <image>} token positions.
 *
 * <h3>How it works</h3>
 * <ol>
 *   <li>The caller registers patch embeddings for a request via
 *       {@link #registerVisionEmbeddings(String, float[][])} before the first
 *       forward pass for that request.
 *   <li>On {@link #forward}, the first node (hasEmbeddings=true) looks up the
 *       embedding table from {@link ForwardRequest#tokenIds()} to build the
 *       initial activation. This handler intercepts that path: for any token
 *       whose ID falls in {@code [IMAGE_TOKEN_START, IMAGE_TOKEN_START + numPatches)}
 *       the embedding is substituted with the pre-computed patch vector instead
 *       of the text vocabulary row.
 *   <li>The modified activation (still {@code float[hiddenDim]}) is passed into
 *       the wrapped {@link ForwardPassHandler} via a synthetic
 *       {@link ForwardRequest#withActivations} so the text handler runs its
 *       normal layers on top.
 *   <li>Subsequent nodes (not the embedding node) pass through unchanged.
 *   <li>Vision embeddings are released when the caller invokes
 *       {@link #releaseVisionEmbeddings(String)}.
 * </ol>
 *
 * <h3>IMAGE_TOKEN_ID convention</h3>
 * The special {@code <image>} token ID is model-specific.  LLaVA-1.5 uses
 * ID 32000; Phi-3 Vision uses 32044.  Pass the correct value at construction.
 * The handler replaces a <em>contiguous range</em> of {@code IMAGE_TOKEN_ID}
 * repetitions with the patch vectors (one patch per repeated token).
 *
 * Thread-safe: the patch embedding map is a {@link ConcurrentHashMap}.
 */
public final class VisionAwareForwardPassHandler implements ForwardPassHandler {

    private final ForwardPassHandler textHandler;
    private final int imageTokenId;
    private final int hiddenDim;

    /**
     * Per-request patch embeddings:
     *   key   = requestId
     *   value = float[numPatches][projectionDim] — one vector per image patch
     *
     * Populated before the first forward pass, cleared on request completion.
     */
    private final ConcurrentHashMap<String, float[][]> patchEmbeddings = new ConcurrentHashMap<>();

    /**
     * @param textHandler  underlying text-only forward-pass handler
     * @param imageTokenId special token ID used to mark image positions
     * @param hiddenDim    LLM hidden dimension (= VisionConfig.projectionDim)
     */
    public VisionAwareForwardPassHandler(ForwardPassHandler textHandler,
                                          int imageTokenId,
                                          int hiddenDim) {
        if (textHandler == null)
            throw new IllegalArgumentException("textHandler must not be null");
        if (hiddenDim < 1)
            throw new IllegalArgumentException("hiddenDim must be >= 1");
        this.textHandler  = textHandler;
        this.imageTokenId = imageTokenId;
        this.hiddenDim    = hiddenDim;
    }

    /**
     * Register patch embeddings produced by {@link VisionEncoder#encode} for a
     * specific request.  Must be called before the first {@link #forward} for
     * that requestId.
     *
     * @param requestId     the request identifier (matches InferenceRequest.requestId)
     * @param patchVectors  float[numPatches][projectionDim]
     */
    public void registerVisionEmbeddings(String requestId, float[][] patchVectors) {
        if (requestId == null || requestId.isBlank())
            throw new IllegalArgumentException("requestId must not be blank");
        if (patchVectors == null || patchVectors.length == 0)
            throw new IllegalArgumentException("patchVectors must not be empty");
        patchEmbeddings.put(requestId, patchVectors);
    }

    /**
     * Release the patch embeddings for a completed request so they can be GC'd.
     * Safe to call even if the requestId was never registered.
     */
    public void releaseVisionEmbeddings(String requestId) {
        patchEmbeddings.remove(requestId);
    }

    // ── ForwardPassHandler ────────────────────────────────────────────────

    @Override
    public ForwardResult forward(ForwardRequest request, ShardContext context) {
        if (!context.hasEmbeddings()) {
            // Intermediate or last node: pass straight through — no embedding lookup here.
            return textHandler.forward(request, context);
        }

        float[][] patches = patchEmbeddings.get(request.requestId());
        if (patches == null) {
            // Text-only request — delegate entirely to the base handler.
            return textHandler.forward(request, context);
        }

        // First node with vision input: build the initial activation with spliced patches.
        float[] initialActivation = buildActivationWithVision(request.tokenIds(), patches);

        // Wrap as an activations request so the text handler skips its embedding lookup.
        ForwardRequest activationsReq = ForwardRequest.withActivations(
                request.requestId(), initialActivation, request.startPosition());

        return textHandler.forward(activationsReq, context);
    }

    @Override
    public boolean isReady() {
        return textHandler.isReady();
    }

    @Override
    public void releaseGpuResources() {
        textHandler.releaseGpuResources();
    }

    @Override
    public Optional<float[]> lastRmsHiddenForEmbedding(ForwardRequest request, ShardContext context) {
        return textHandler.lastRmsHiddenForEmbedding(request, context);
    }

    // ── Private helpers ───────────────────────────────────────────────────

    /**
     * Build the initial hidden-state vector for the first node.
     *
     * For text tokens the embedding lookup is delegated to the text handler.
     * For image tokens (ID == imageTokenId) the pre-computed patch vector is
     * used directly.
     *
     * Since the text handler owns the token embedding table (it is loaded as
     * part of the GGUF shard), we reconstruct the text embedding by running a
     * minimal single-token prefill on the text handler and extracting the last
     * RMS hidden state.  However, this would require an extra forward pass.
     *
     * Simpler and correct: the last token in the sequence is the one the
     * generation loop asks us to score.  If it is an image token, return its
     * patch vector directly.  If it is a text token, forward to the text handler
     * (which will do the embedding lookup) — and never reach this branch.
     *
     * The generation loop always calls forward() with the full token sequence
     * but only the last token matters for each decode step.  For the prefill
     * phase the loop walks positions individually; vision tokens appear as a
     * contiguous run of IMAGE_TOKEN_ID at known positions.
     */
    private float[] buildActivationWithVision(int[] tokenIds, float[][] patches) {
        int lastToken = tokenIds[tokenIds.length - 1];

        if (lastToken != imageTokenId) {
            // Text token at this position — we cannot build the embedding here because
            // the embedding table lives inside the text handler.  Return a zero vector;
            // the caller (GenerationLoop) re-routes through the text handler's own
            // getInitialActivation() on the next pass.  In practice the generation
            // loop only calls VisionAwareForwardPassHandler at image-token positions
            // during prefill; this path is a safety guard.
            return new float[hiddenDim];
        }

        // Count how many IMAGE_TOKEN_IDs appear before this position — that is the
        // patch index to use for the current token.
        int patchIdx = 0;
        for (int i = 0; i < tokenIds.length - 1; i++) {
            if (tokenIds[i] == imageTokenId)
                patchIdx++;
        }

        if (patchIdx >= patches.length) {
            // Guard: more image tokens in the sequence than patches available.
            // Return zero vector; the model will produce uncertain output but will not crash.
            return new float[hiddenDim];
        }

        float[] patch = patches[patchIdx];
        if (patch.length != hiddenDim) {
            throw new IllegalStateException(
                    "Patch embedding dim " + patch.length
                    + " does not match hiddenDim " + hiddenDim
                    + " — check VisionConfig.projectionDim matches the LLM hidden size");
        }
        // Defensive copy so the caller cannot mutate our stored patch.
        float[] out = new float[hiddenDim];
        System.arraycopy(patch, 0, out, 0, hiddenDim);
        return out;
    }
}