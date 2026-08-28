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

package cab.ml.juno.node;

/**
 * Phi-2 partial rotary embeddings — port of llama.cpp {@code ggml_rope_ext}
 * for {@code GGML_ROPE_TYPE_NEOX} (mode 2), restricted to the first
 * {@code ropeDim} dimensions of each head.
 *
 * <h3>Why this class exists</h3>
 * Phi-2's HuggingFace reference implementation ({@code modeling_phi.py}) builds
 * Q/K with {@code rotate_half}, the GPT-NeoX convention: for a rotary width
 * {@code ropeDim}, dimension {@code i} is paired with dimension
 * {@code i + ropeDim/2} (a split-half pairing), not with its neighbour
 * {@code i+1} (adjacent-pair pairing). The GGUF converter for {@code phi2}
 * does not permute the {@code attn_qkv} weight rows to the interleaved layout
 * that original LLaMA conversion uses, so the raw tensor layout in the GGUF
 * requires split-half pairing at inference time. Dimensions
 * {@code [ropeDim, headDim)} are never rotated and pass through unchanged
 * (partial rotary).
 *
 * <p>This is the same NeoX/adjacent-pair distinction already documented and
 * fixed for {@link Phi3Rope} on 2026-06-11 (see
 * {@code docs/phi3-inference-handoff.md}, section C). {@link Phi2TransformerHandler}
 * was written afterwards with the adjacent-pair convention despite the sibling
 * fix already existing in the same codebase; this class corrects that
 * regression for the {@code phi2} architecture.
 *
 * <p>Unlike {@link Phi3Rope}, Phi-2 has no long-context YARN scaling and no
 * per-dimension {@code rope_factors} tensors, so the frequency computation is
 * the plain {@code freq = ropeTheta ^ (-2i / ropeDim)} — this class only
 * differs from the old {@code Phi2TransformerHandler.ropePartial} in the
 * *pairing*, not the frequency formula.
 */
final class Phi2Rope {

    private Phi2Rope() {
    }

    /**
     * Apply partial NeoX-style RoPE in-place to {@code x[nHeads * headDim]}.
     *
     * @param x         Q or K vector, laid out as {@code nHeads} contiguous
     *                  blocks of {@code headDim} (modified in place)
     * @param pos       absolute sequence position for this token
     * @param nHeads    number of heads represented in {@code x}
     * @param headDim   dimension per head
     * @param ropeDim   number of leading dimensions per head that are rotated
     *                  ({@code phi2.rope.dimension_count}, typically
     *                  {@code headDim / 2}); must be even and {@code <= headDim}
     * @param ropeTheta RoPE base frequency ({@code phi2.rope.freq_base})
     */
    static void ropePartial(float[] x, int pos, int nHeads, int headDim,
                             int ropeDim, float ropeTheta) {
        int half = ropeDim / 2;
        for (int h = 0; h < nHeads; h++) {
            int base = h * headDim;
            for (int i = 0; i < half; i++) {
                double freq  = 1.0 / Math.pow(ropeTheta, (2.0 * i) / ropeDim);
                double angle = pos * freq;
                float  cosA  = (float) Math.cos(angle);
                float  sinA  = (float) Math.sin(angle);
                float  x0    = x[base + i];
                float  x1    = x[base + i + half];
                x[base + i]        = x0 * cosA - x1 * sinA;
                x[base + i + half] = x0 * sinA + x1 * cosA;
            }
            // Dims [ropeDim .. headDim-1] are intentionally left unchanged.
        }
    }
}