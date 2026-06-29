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
        float layerNormEps  // layer-norm epsilon for the CLIP encoder
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

        return new VisionConfig(imageSize, patchSize, hiddenSize, numLayers,
                numHeads, intermediateSize, projectionDim, eps);
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
     * Number of tokens passed to the LLM: numPatches + 1 CLS token.
     * Some models strip the CLS token before projection; this value is the
     * upper bound. {@link VisionEncoder} documents whether CLS is retained.
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
     */
    static VisionConfig synthetic(int imageSize, int patchSize, int hiddenSize,
                                   int numLayers, int numHeads, int projectionDim) {
        int intermediateSize = hiddenSize * 4;
        float eps = 1e-5f;
        return new VisionConfig(imageSize, patchSize, hiddenSize, numLayers,
                numHeads, intermediateSize, projectionDim, eps);
    }

    @Override
    public String toString() {
        return String.format(
                "VisionConfig{image=%d patch=%d hidden=%d layers=%d heads=%d ffn=%d proj=%d eps=%.1e}",
                imageSize, patchSize, hiddenSize, numLayers, numHeads,
                intermediateSize, projectionDim, layerNormEps);
    }
}