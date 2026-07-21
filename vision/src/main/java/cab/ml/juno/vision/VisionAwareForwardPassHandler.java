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
import java.util.logging.Logger;

import cab.ml.juno.node.BatchForwardRequest;
import cab.ml.juno.node.BatchForwardResult;
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

    private static final Logger log = Logger.getLogger(VisionAwareForwardPassHandler.class.getName());

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

    /**
     * Batched prefill: build the initial activation matrix for the whole window
     * with vision patch vectors spliced in for image-token positions, then delegate
     * the full batch to the wrapped handler in one call.
     *
     * <p>Building the activation matrix is O(windowSize * hiddenDim) — negligible
     * next to the transformer matmuls that follow. The key benefit is that the
     * wrapped handler receives a single {@link BatchForwardRequest} for the entire
     * window instead of being called once per token as in the old loop.
     */
    @Override
    public BatchForwardResult forwardBatch(BatchForwardRequest request, ShardContext context) {
        // Unconditional — proves this exact class/build actually executed for this
        // request, regardless of anything downstream. If you don't see this line
        // in the log immediately after "Prefill: calling pipeline.prefillBatch()",
        // the running jar does NOT contain this build; stop and rebuild before
        // trusting anything else in the log. (Added 2026-07-20 after the
        // text-embedding-stats line silently never appeared across two full runs
        // due to a stale build — this line cannot have the same failure mode,
        // since it has no condition gating it at all.)
        log.info("[vision] forwardBatch ENTER requestId=" + request.requestId() + " windowSize="
                + request.windowSize() + " hasEmbeddings=" + context.hasEmbeddings());

        if (!context.hasEmbeddings()) {
            log.info("[vision] forwardBatch PASSTHROUGH (not the embeddings node) requestId="
                    + request.requestId());
            return textHandler.forwardBatch(request, context);
        }

        float[][] patches = patchEmbeddings.get(request.requestId());
        if (patches == null) {
            log.info("[vision] forwardBatch PASSTHROUGH (no patches registered for this requestId — "
                    + "plain text request) requestId=" + request.requestId());
            return textHandler.forwardBatch(request, context);
        }

        int W = request.windowSize();
        int[] tokenIds = request.tokenIds();
        float[] flatActivations = buildWindowActivationsWithVision(tokenIds, patches, W);

        BatchForwardRequest activationsReq = BatchForwardRequest.withActivations(
                request.requestId(), flatActivations, W, request.startPosition());

        log.info("[vision] forwardBatch delegating to wrapped text handler requestId=" + request.requestId()
                + " windowSize=" + W);
        return textHandler.forwardBatch(activationsReq, context);
    }

    /**
     * Build a flattened {@code float[windowSize * hiddenDim]} activation matrix.
     * Image-token positions use the pre-computed patch vector; text-token
     * positions use the wrapped handler's real embedding-table row via
     * {@link ForwardPassHandler#embedToken(int)} — NOT a zero vector. (Prior to
     * 2026-07-12 this used a zero vector for every text token, which silently
     * fed the model 600+ meaningless positions and produced incoherent output
     * even once the image itself was correctly seen.)
     *
     * <p>Logs an unconditional, per-call aggregate of every real text-token
     * embedding encountered (min/max/mean/std/L2 norm, plus image- vs
     * text-token counts) — directly comparable to {@code VisionEncoder}'s own
     * per-request patch-embedding stats log. Unconditional (not gated behind
     * a "log once" flag) specifically so it cannot silently fail to appear.
     */
    private float[] buildWindowActivationsWithVision(int[] tokenIds, float[][] patches, int W) {
        float[] flat = new float[W * hiddenDim];
        int patchIdx = 0;
        int textTokenCount = 0;
        int imageTokenCount = 0;
        float min = Float.POSITIVE_INFINITY, max = Float.NEGATIVE_INFINITY;
        double sum = 0, sumSq = 0;
        double normSum = 0;
        float minNorm = Float.POSITIVE_INFINITY, maxNorm = Float.NEGATIVE_INFINITY;

        for (int b = 0; b < W; b++) {
            if (tokenIds[b] == imageTokenId) {
                imageTokenCount++;
                if (patchIdx < patches.length) {
                    float[] patch = patches[patchIdx++];
                    if (patch.length != hiddenDim)
                        throw new IllegalStateException("Patch embedding dim " + patch.length
                                + " does not match hiddenDim " + hiddenDim);
                    System.arraycopy(patch, 0, flat, b * hiddenDim, hiddenDim);
                }
                // else: more image tokens than patches — leave zero (safety guard)
            } else {
                textTokenCount++;
                float[] textEmbedding = textHandler.embedToken(tokenIds[b]);
                if (textEmbedding.length != hiddenDim)
                    throw new IllegalStateException("Text embedding dim " + textEmbedding.length
                            + " does not match hiddenDim " + hiddenDim);

                double normSq = 0;
                for (float v : textEmbedding) {
                    if (v < min) min = v;
                    if (v > max) max = v;
                    sum += v;
                    sumSq += (double) v * v;
                    normSq += (double) v * v;
                }
                float norm = (float) Math.sqrt(normSq);
                normSum += norm;
                if (norm < minNorm) minNorm = norm;
                if (norm > maxNorm) maxNorm = norm;

                System.arraycopy(textEmbedding, 0, flat, b * hiddenDim, hiddenDim);
            }
        }

        if (textTokenCount > 0) {
            long totalElems = (long) textTokenCount * hiddenDim;
            double mean = sum / totalElems;
            double std = Math.sqrt(sumSq / totalElems - mean * mean);
            double meanNorm = normSum / textTokenCount;
            log.info(String.format(
                    "[vision] Real text-token embeddings stats (ALL %d text tokens in this window, dim=%d): "
                            + "min=%.4f max=%.4f mean=%.4f std=%.4f | per-token L2 norm: min=%.4f mean=%.4f "
                            + "max=%.4f  <-- compare against VisionEncoder's per-patch L2 norm log for this "
                            + "same request",
                    textTokenCount, hiddenDim, min, max, mean, std, minNorm, meanNorm, maxNorm));
        } else {
            log.warning("[vision] Window contains ZERO text tokens (all " + imageTokenCount
                    + " positions are image tokens) — no comparison stats available. This would be unusual "
                    + "for a real chat prompt and is worth double-checking if seen.");
        }
        log.info("[vision] Window token composition: imageTokens=" + imageTokenCount + " textTokens="
                + textTokenCount + " totalWindow=" + W + " patchesAvailable=" + patches.length
                + " patchesConsumed=" + patchIdx);

        return flat;
    }

    @Override
    public ForwardResult forward(ForwardRequest request, ShardContext context) {
        log.info("[vision] forward ENTER requestId=" + request.requestId() + " hasEmbeddings="
                + context.hasEmbeddings());

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
     * Build the initial hidden-state vector for the first node, for a single
     * token position (used by {@code PrefillMode.SINGLE} and — for the very
     * last prompt token that starts decode — by every prefill mode).
     *
     * Image tokens (ID == imageTokenId) use the pre-computed patch vector.
     * Text tokens use the wrapped handler's real embedding-table row via
     * {@link ForwardPassHandler#embedToken(int)} — NOT a zero vector, matching
     * the fix applied to {@link #buildWindowActivationsWithVision}.
     */
    private float[] buildActivationWithVision(int[] tokenIds, float[][] patches) {
        int lastToken = tokenIds[tokenIds.length - 1];

        if (lastToken != imageTokenId) {
            return textHandler.embedToken(lastToken);
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