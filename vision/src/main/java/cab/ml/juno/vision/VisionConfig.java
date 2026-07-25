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
        boolean useGelu     // true: standard (tanh-approx) GELU. false: quick_gelu
                             // (x * sigmoid(1.702x), OpenAI CLIP's original activation).
                             // Read from clip.use_gelu — llama.cpp's own flag for exactly
                             // this distinction. 2026-07-20: found via ./juno gguf-info
                             // that llava-v1.5-7b-mmproj-Q4_0.gguf declares this false;
                             // VisionEncoder previously always used standard GELU
                             // regardless, silently using the wrong activation in every
                             // one of the 23 transformer blocks. See CHANGELOG.
) {

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

        return new VisionConfig(imageSize, patchSize, hiddenSize, numLayers,
                numHeads, intermediateSize, projectionDim, eps, useGelu);
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
     */
    static VisionConfig synthetic(int imageSize, int patchSize, int hiddenSize,
                                   int numLayers, int numHeads, int projectionDim) {
        return synthetic(imageSize, patchSize, hiddenSize, numLayers, numHeads, projectionDim, true);
    }

    /** Same as {@link #synthetic(int, int, int, int, int, int)} with an explicit useGelu. */
    static VisionConfig synthetic(int imageSize, int patchSize, int hiddenSize,
                                   int numLayers, int numHeads, int projectionDim, boolean useGelu) {
        int intermediateSize = hiddenSize * 4;
        float eps = 1e-5f;
        return new VisionConfig(imageSize, patchSize, hiddenSize, numLayers,
                numHeads, intermediateSize, projectionDim, eps, useGelu);
    }

    @Override
    public String toString() {
        return String.format(
                "VisionConfig{image=%d patch=%d hidden=%d layers=%d heads=%d ffn=%d proj=%d eps=%.1e useGelu=%b}",
                imageSize, patchSize, hiddenSize, numLayers, numHeads,
                intermediateSize, projectionDim, layerNormEps, useGelu);
    }
}