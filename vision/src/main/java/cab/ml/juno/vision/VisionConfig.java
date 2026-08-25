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

import cab.ml.juno.node.GgufReader;

/**
 * Vision encoder configuration read from GGUF metadata.
 *
 * Covers the CLIP / SigLIP encoder embedded in multimodal models
 * (LLaVA-1.5, Phi-3 Vision, Qwen-VL). Reads the {@code clip.*}
 * and {@code vision.*} key namespaces used by llama.cpp-format GGUFs.
 *
 * Key relationships:
 * <ul>
 *   <li>{@link #patchSize} and {@link #imageSize} determine the patch grid:
 *       {@code numPatches = (imageSize / patchSize)^2}.
 *   <li>{@link #hiddenSize} is the CLIP encoder residual dimension; it is
 *       separate from the LLM hidden dimension.
 *   <li>{@link #projectionDim} is the output of the vision projector (MLP or
 *       linear) that maps CLIP embeddings into the LLM token embedding space.
 *   <li>{@link #numLayers} is the number of CLIP transformer layers.
 *   <li>{@link #numHeads} is the number of CLIP attention heads.
 * </ul>
 */
public record VisionConfig(
        int imageSize,      // input image resolution (square: imageSize × imageSize)
        int patchSize,      // ViT patch size (e.g. 14 for CLIP-L/14)
        int hiddenSize,     // CLIP encoder hidden dimension
        int numLayers,      // number of CLIP transformer blocks
        int numHeads,       // number of CLIP attention heads
        int intermediateSize, // CLIP FFN intermediate dimension
        int projectionDim,  // output dimension of the vision projector (= LLM hiddenDim)
        float layerNormEps, // layer-norm epsilon for the CLIP encoder
        boolean useGelu,    // true: standard (tanh-approx) GELU. false: quick_gelu
                             // (x * sigmoid(1.702x), OpenAI CLIP's original activation).
                             // Read from clip.use_gelu — llama.cpp's own flag for exactly
                             // this distinction. 2026-07-20: found via ./juno gguf-info
                             // that llava-v1.5-7b-mmproj-Q4_0.gguf declares this false;
                             // VisionEncoder previously always used standard GELU
                             // regardless, silently using the wrong activation in every
                             // one of the 23 transformer blocks. See CHANGELOG.
        float[] imageMean,  // per-channel pixel normalisation mean (RGB order)
        float[] imageStd    // per-channel pixel normalisation std (RGB order)
                             // 2026-07-29: ImagePatchEmbedder previously hardcoded the
                             // OpenAI CLIP constants (0.4815/0.4578/0.4082,
                             // 0.2686/0.2613/0.2758) for every model. SigLIP-family
                             // encoders (moondream2's mmproj) are trained with
                             // image_mean=image_std=[0.5,0.5,0.5] instead; using the
                             // CLIP constants silently mis-scales every pixel fed to a
                             // SigLIP encoder. Resolved here from clip.vision.image_mean
                             // / image_std when the GGUF declares them, else defaulted
                             // by encoder family (CLS token present → CLIP; absent →
                             // SigLIP), matching VisionEncoder's own CLS-token detection.
) {

    /** OpenAI CLIP normalisation constants (ImageNet-derived). */
    static final float[] OPENAI_CLIP_MEAN = {0.48145466f, 0.4578275f, 0.40821073f};
    static final float[] OPENAI_CLIP_STD  = {0.26862954f, 0.26130258f, 0.27577711f};

    /** SigLIP / "imagenet standard" normalisation constants. */
    static final float[] SIGLIP_MEAN = {0.5f, 0.5f, 0.5f};
    static final float[] SIGLIP_STD  = {0.5f, 0.5f, 0.5f};

    /**
     * Derive from an open {@link GgufReader}.
     *
     * Reads {@code clip.*} keys first (llama.cpp mmproj convention), then
     * falls back to {@code vision.*} keys used by older Phi-3 Vision GGUFs.
     */
    public static VisionConfig from(GgufReader r) {
        // Prefer clip.* namespace (llama.cpp mmproj standard)
        int imageSize       = r.metaInt("clip.vision.image_size",
                              r.metaInt("vision.image_size",       336));
        int patchSize       = r.metaInt("clip.vision.patch_size",
                              r.metaInt("vision.patch_size",        14));
        int hiddenSize      = r.metaInt("clip.vision.embedding_length",
                              r.metaInt("vision.embedding_length", 1024));
        int numLayers       = r.metaInt("clip.vision.block_count",
                              r.metaInt("vision.block_count",        24));
        int numHeads        = r.metaInt("clip.vision.attention.head_count",
                              r.metaInt("vision.attention.head_count", 16));
        int intermediateSize= r.metaInt("clip.vision.feed_forward_length",
                              r.metaInt("vision.feed_forward_length", 4096));
        int projectionDim   = r.metaInt("clip.vision.projection_dim",
                              r.metaInt("vision.projection_dim",   4096));
        float eps           = r.metaFloat("clip.vision.attention.layer_norm_epsilon",
                              r.metaFloat("vision.attention.layer_norm_epsilon", 1e-5f));
        // Default true (standard GELU) when the key is absent, matching this
        // codebase's prior unconditional behavior for files that don't declare it.
        boolean useGelu      = r.metaBool("clip.use_gelu", true);

        // CLS-token presence distinguishes CLIP-style (has v.class_embd) from
        // SigLIP-style (no CLS token) encoders — same signal VisionEncoder
        // itself uses. Each family trains with different normalisation
        // constants; the GGUF's own declared values (when present) always win.
        boolean hasClsToken = r.hasTensor("v.class_embd");
        float[] defaultMean = hasClsToken ? OPENAI_CLIP_MEAN : SIGLIP_MEAN;
        float[] defaultStd  = hasClsToken ? OPENAI_CLIP_STD  : SIGLIP_STD;
        float[] imageMean   = r.metaFloatArray("clip.vision.image_mean", defaultMean);
        float[] imageStd    = r.metaFloatArray("clip.vision.image_std", defaultStd);

        return new VisionConfig(imageSize, patchSize, hiddenSize, numLayers,
                numHeads, intermediateSize, projectionDim, eps, useGelu, imageMean, imageStd);
    }

    /**
     * Total number of image patches produced by the ViT patch embedding.
     * Does not include the CLS token.
     */
    public int numPatches() {
        int grid = imageSize / patchSize;
        return grid * grid;
    }

    /**
     * {@code numPatches() + 1} — the sequence length used by CLIP-style encoders
     * that prepend a CLS token ({@code v.class_embd} present).
     *
     * <p>For SigLIP-style encoders (e.g. moondream2) that have no CLS token,
     * the actual sequence length is {@link #numPatches()} and this method over-
     * counts by one. Prefer {@link VisionEncoder#encode}'s internal logic over
     * calling this method directly when sizing buffers — {@link VisionEncoder}
     * derives sequence length from whether {@code v.class_embd} was loaded.
     */
    public int numVisionTokens() {
        return numPatches() + 1;
    }

    /** Head dimension = hiddenSize / numHeads. */
    public int headDim() {
        return hiddenSize / numHeads;
    }

    /**
     * Build a synthetic config for unit tests — no GGUF file needed.
     * useGelu defaults to true (standard GELU), matching this class's
     * pre-2026-07-20 behavior, so existing test call sites are unaffected.
     * imageMean/imageStd default to the OpenAI CLIP constants, matching this
     * class's pre-2026-07-29 behavior, so existing test call sites are
     * unaffected.
     */
    static VisionConfig synthetic(int imageSize, int patchSize, int hiddenSize,
                                   int numLayers, int numHeads, int projectionDim) {
        return synthetic(imageSize, patchSize, hiddenSize, numLayers, numHeads, projectionDim, true);
    }

    /** Same as {@link #synthetic(int, int, int, int, int, int)} with an explicit useGelu. */
    static VisionConfig synthetic(int imageSize, int patchSize, int hiddenSize,
                                   int numLayers, int numHeads, int projectionDim, boolean useGelu) {
        return synthetic(imageSize, patchSize, hiddenSize, numLayers, numHeads, projectionDim, useGelu,
                OPENAI_CLIP_MEAN, OPENAI_CLIP_STD);
    }

    /** Same as {@link #synthetic(int, int, int, int, int, int, boolean)} with explicit normalisation. */
    static VisionConfig synthetic(int imageSize, int patchSize, int hiddenSize,
                                   int numLayers, int numHeads, int projectionDim, boolean useGelu,
                                   float[] imageMean, float[] imageStd) {
        int intermediateSize = hiddenSize * 4;
        float eps = 1e-5f;
        return new VisionConfig(imageSize, patchSize, hiddenSize, numLayers,
                numHeads, intermediateSize, projectionDim, eps, useGelu, imageMean, imageStd);
    }

    @Override
    public String toString() {
        return String.format(
                "VisionConfig{image=%d patch=%d hidden=%d layers=%d heads=%d ffn=%d proj=%d eps=%.1e useGelu=%b "
                        + "mean=[%.4f,%.4f,%.4f] std=[%.4f,%.4f,%.4f]}",
                imageSize, patchSize, hiddenSize, numLayers, numHeads,
                intermediateSize, projectionDim, layerNormEps, useGelu,
                imageMean[0], imageMean[1], imageMean[2], imageStd[0], imageStd[1], imageStd[2]);
    }
}