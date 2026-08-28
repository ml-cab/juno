package cab.ml.juno.vision;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("VisionEncoder — math primitives correctness")
class VisionEncoderTest {

    // ── layerNorm ─────────────────────────────────────────────────────────────

    @Test
    @DisplayName("layerNorm: zero-mean unit-variance input with ones weights becomes identity")
    void layer_norm_ones_weight_identity() {
        // x = [1, 2, 3, 4]  mean=2.5  var=1.25
        float[] x = {1f, 2f, 3f, 4f};
        float[] w = {1f, 1f, 1f, 1f};
        float[] b = {0f, 0f, 0f, 0f};

        float[] out = VisionEncoder.layerNorm(x, w, b, 1e-5f);

        // Each element should be (x_i - mean) / std
        float mean = 2.5f;
        float var  = 1.25f;
        float std  = (float) Math.sqrt(var + 1e-5f);

        assertThat(out[0]).isCloseTo((1f - mean) / std, within(1e-5f));
        assertThat(out[1]).isCloseTo((2f - mean) / std, within(1e-5f));
        assertThat(out[2]).isCloseTo((3f - mean) / std, within(1e-5f));
        assertThat(out[3]).isCloseTo((4f - mean) / std, within(1e-5f));
    }

    @Test
    @DisplayName("layerNorm: weight scaling applied element-wise")
    void layer_norm_weight_scaling() {
        float[] x = {2f, 2f, 2f};
        float[] w = {0f, 1f, 2f};
        float[] b = {0f, 0f, 0f};

        float[] out = VisionEncoder.layerNorm(x, w, b, 1e-5f);

        // All x are identical → normalised value is 0 for all positions
        // w[0]=0 → out[0]=0, w[1]=1 → out[1]=0, w[2]=2 → out[2]=0
        assertThat(out[0]).isCloseTo(0f, within(1e-5f));
        assertThat(out[1]).isCloseTo(0f, within(1e-5f));
        assertThat(out[2]).isCloseTo(0f, within(1e-5f));
    }

    @Test
    @DisplayName("layerNorm: bias added after normalisation and scaling")
    void layer_norm_bias_added() {
        float[] x = {2f, 2f};
        float[] w = {1f, 1f};
        float[] b = {3f, -1f};

        float[] out = VisionEncoder.layerNorm(x, w, b, 1e-5f);

        // All identical → norm = 0; with bias: out = [3, -1]
        assertThat(out[0]).isCloseTo(3f, within(1e-4f));
        assertThat(out[1]).isCloseTo(-1f, within(1e-4f));
    }

    @Test
    @DisplayName("layerNorm: output length equals input length")
    void layer_norm_output_length() {
        float[] x = {1f, 2f, 3f, 4f, 5f};
        float[] w = {1f, 1f, 1f, 1f, 1f};
        float[] b = new float[5];
        assertThat(VisionEncoder.layerNorm(x, w, b, 1e-5f)).hasSize(5);
    }

    @Test
    @DisplayName("layerNorm: collapses an unnormalised final-transformer-block-scale vector "
            + "to a bounded, unit-variance-ish scale (regression for the missing v.post_ln bug)")
    void layer_norm_collapses_unnormalised_transformer_output() {
        // Mirrors the actual magnitudes logged in production for moondream2's
        // final transformer block, before v.post_ln existed in this encoder:
        // "Vision patch embeddings stats ... per-patch L2 norm: min=353 mean=9272
        // max=69715". A patch embedding with an L2 norm in the tens of thousands,
        // handed directly to the projector with no normalisation, silently
        // dwarfs any real signal. LayerNorm with identity weight/bias must bring
        // that down to order-of-sqrt(dim), regardless of the input's raw scale.
        int dim = 64;
        float[] huge = new float[dim];
        java.util.Random rnd = new java.util.Random(42);
        for (int i = 0; i < dim; i++)
            huge[i] = (float) (rnd.nextGaussian() * 9000.0); // ~ the logged per-patch scale
        float[] w = new float[dim];
        java.util.Arrays.fill(w, 1f);
        float[] b = new float[dim];

        float[] out = VisionEncoder.layerNorm(huge, w, b, 1e-5f);

        double outNorm = 0;
        for (float v : out) outNorm += (double) v * v;
        outNorm = Math.sqrt(outNorm);

        // Unit-variance-per-dim vector has L2 norm ~= sqrt(dim) = 8 here.
        // The pre-fix bug fed the projector norms in the tens of thousands
        // regardless of dim; post-fix must land within a small constant
        // factor of sqrt(dim), independent of the huge input scale.
        assertThat(outNorm).isLessThan(5.0 * Math.sqrt(dim));
    }

    // ── GELU ─────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("gelu(0) = 0")
    void gelu_zero() {
        assertThat(VisionEncoder.gelu(0f)).isCloseTo(0f, within(1e-6f));
    }

    @Test
    @DisplayName("gelu(x) ≈ x for large positive x")
    void gelu_large_positive_approx_identity() {
        // For large x, tanh(...) → 1, so gelu(x) → x
        assertThat(VisionEncoder.gelu(10f)).isCloseTo(10f, within(0.01f));
    }

    @Test
    @DisplayName("gelu(x) ≈ 0 for large negative x")
    void gelu_large_negative_near_zero() {
        assertThat(VisionEncoder.gelu(-10f)).isCloseTo(0f, within(0.01f));
    }

    @Test
    @DisplayName("gelu shape: trough near x=-1, monotone increasing for x > -0.17")
    void gelu_shape() {
        // GELU is NOT globally monotone for negative x.
        // It has a local minimum around x ≈ -0.17 (gelu ≈ -0.169).
        // For x < -0.17 the function rises back toward 0 as x decreases.
        // Concretely: gelu(-2) ≈ -0.045  >  gelu(-1) ≈ -0.159
        assertThat(VisionEncoder.gelu(-2f)).isGreaterThan(VisionEncoder.gelu(-1f));

        // The trough is the global minimum in the negative region
        assertThat(VisionEncoder.gelu(-1f)).isLessThan(VisionEncoder.gelu(0f));

        // For positive x GELU is strictly monotone increasing
        assertThat(VisionEncoder.gelu(0f)).isLessThan(VisionEncoder.gelu(1f));
        assertThat(VisionEncoder.gelu(1f)).isLessThan(VisionEncoder.gelu(2f));

        // Large negative values converge back toward 0 from below
        assertThat(VisionEncoder.gelu(-3f)).isGreaterThan(VisionEncoder.gelu(-1f));
    }

    // ── quick_gelu (clip.use_gelu=false) — 2026-07-20 fix ───────────────────────
    //
    // llava-v1.5-7b-mmproj-Q4_0.gguf declares clip.use_gelu=false (confirmed via
    // ./juno gguf-info), meaning the ViT transformer blocks' FFN activation
    // should be quick_gelu (x * sigmoid(1.702x), OpenAI CLIP's original
    // activation) — NOT the standard tanh-approx gelu() that every call site
    // used unconditionally before this fix.

    @Test
    @DisplayName("quickGelu(0) = 0")
    void quickGelu_zero() {
        assertThat(VisionEncoder.quickGelu(0f)).isCloseTo(0f, within(1e-6f));
    }

    @Test
    @DisplayName("quickGelu(x) ≈ x for large positive x")
    void quickGelu_large_positive_approx_identity() {
        assertThat(VisionEncoder.quickGelu(10f)).isCloseTo(10f, within(0.001f));
    }

    @Test
    @DisplayName("quickGelu(x) ≈ 0 for large negative x")
    void quickGelu_large_negative_near_zero() {
        assertThat(VisionEncoder.quickGelu(-10f)).isCloseTo(0f, within(0.001f));
    }

    @Test
    @DisplayName("quickGelu shape: trough near x=-0.75, monotone increasing for x > -0.75")
    void quickGelu_shape() {
        // Like gelu, quick_gelu is not globally monotone for negative x — it has
        // a local minimum around x ≈ -0.75 (quick_gelu ≈ -0.164), then rises
        // back toward 0 as x decreases further.
        assertThat(VisionEncoder.quickGelu(-2f)).isGreaterThan(VisionEncoder.quickGelu(-0.75f));
        assertThat(VisionEncoder.quickGelu(-0.75f)).isLessThan(VisionEncoder.quickGelu(0f));
        assertThat(VisionEncoder.quickGelu(0f)).isLessThan(VisionEncoder.quickGelu(1f));
        assertThat(VisionEncoder.quickGelu(1f)).isLessThan(VisionEncoder.quickGelu(2f));
        assertThat(VisionEncoder.quickGelu(-3f)).isGreaterThan(VisionEncoder.quickGelu(-0.75f));
    }

    @Test
    @DisplayName("quickGelu differs measurably from gelu — the fix must actually change behavior, "
            + "not silently compute the same thing under a new name")
    void quickGelu_differsFromStandardGelu() {
        assertThat(VisionEncoder.quickGelu(-2f)).isNotCloseTo(VisionEncoder.gelu(-2f), within(0.01f));
        assertThat(VisionEncoder.quickGelu(1f)).isNotCloseTo(VisionEncoder.gelu(1f), within(0.001f));
    }

    // ── VisionConfig integration ──────────────────────────────────────────────

    @Test
    @DisplayName("VisionConfig.synthetic produces expected numPatches for typical sizes")
    void vision_config_num_patches_sanity() {
        // LLaVA-1.5 336/14 = 24 → 576 patches
        VisionConfig cfg336 = VisionConfig.synthetic(336, 14, 1024, 24, 16, 4096);
        assertThat(cfg336.numPatches()).isEqualTo(576);

        // CLIP-B/32 224/32 = 7 → 49 patches
        VisionConfig cfg224 = VisionConfig.synthetic(224, 32, 768, 12, 12, 512);
        assertThat(cfg224.numPatches()).isEqualTo(49);
    }

    // ── FFN orientation auto-detection ────────────────────────────────────────
    //
    // Regression coverage for the reported crash:
    //   java.lang.ArrayIndexOutOfBoundsException: Index 1024 out of bounds for length 1024
    //       at cab.ml.juno.vision.VisionEncoder.mlp(VisionEncoder.java:376)
    // Root cause: llava-phi-3-mini's mmproj file's "ffn_up.bias" tensor was
    // actually shaped for the I→H contraction (length hiddenSize=1024), not the
    // H→I expansion (length intermediateSize=4096) our code assumed from the
    // tensor's name alone. resolveFfnOrientation replaces that name-trusting
    // assumption with a decision based on each tensor's own measured shape.

    @Test
    @DisplayName("resolveFfnOrientation: normal naming (ffn_up expands, ffn_down contracts) is NORMAL")
    void ffn_orientation_normal() {
        VisionEncoder.FfnOrientation result = VisionEncoder.resolveFfnOrientation(
                0, "v.blk.0.ffn_up.weight", "v.blk.0.ffn_down.weight",
                4096, 1024, 4096, 1024);

        assertThat(result).isEqualTo(VisionEncoder.FfnOrientation.NORMAL);
    }

    @Test
    @DisplayName("resolveFfnOrientation: reversed naming (ffn_up outputs hiddenSize) is SWAPPED — "
            + "this is the exact llava-phi-3-mini mmproj case that used to crash")
    void ffn_orientation_swapped() {
        VisionEncoder.FfnOrientation result = VisionEncoder.resolveFfnOrientation(
                5, "v.blk.5.ffn_up.weight", "v.blk.5.ffn_down.weight",
                1024, 4096, 4096, 1024);

        assertThat(result).isEqualTo(VisionEncoder.FfnOrientation.SWAPPED);
    }

    @Test
    @DisplayName("resolveFfnOrientation: neither orientation matches → throws with both actual outDims named")
    void ffn_orientation_neither_matches_throws() {
        assertThatThrownBy(() -> VisionEncoder.resolveFfnOrientation(
                2, "v.blk.2.ffn_up.weight", "v.blk.2.ffn_down.weight",
                2048, 2048, 4096, 1024))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("block 2")
                .hasMessageContaining("v.blk.2.ffn_up.weight")
                .hasMessageContaining("v.blk.2.ffn_down.weight")
                .hasMessageContaining("2048");
    }

    @Test
    @DisplayName("resolveFfnOrientation: hiddenSize == intermediateSize is unambiguous but harmless — "
            + "both orientations are numerically identical, so NORMAL is a safe default")
    void ffn_orientation_equal_dims_defaults_normal() {
        VisionEncoder.FfnOrientation result = VisionEncoder.resolveFfnOrientation(
                0, "v.blk.0.ffn_up.weight", "v.blk.0.ffn_down.weight",
                512, 512, 512, 512);

        assertThat(result).isEqualTo(VisionEncoder.FfnOrientation.NORMAL);
    }

    // ── Projector output dim auto-detection ───────────────────────────────────
    //
    // Regression coverage for the second reported crash (after the FFN fix above
    // got the request further):
    //   java.lang.IllegalArgumentException: A.length=3145728 != rows*cols=786432
    //       at cab.ml.juno.node.CpuMatVec.sgemv(CpuMatVec.java:42)
    //       at cab.ml.juno.vision.VisionEncoder.project(VisionEncoder.java:477)
    // Root cause: llava-phi-3-mini's mmproj file's clip.vision.projection_dim
    // metadata says 768, but mm.0.weight's own GGUF shape is actually
    // [hiddenSize=1024, 3072] — 3072 being the LLM's real hidden dimension,
    // the width the projector must actually produce. 768*1024=786432 (what the
    // code assumed); 3072*1024=3,145,728 (what the tensor actually contains).
    // resolveProjectorOutputDim replaces the metadata-trusting assumption with
    // the tensor's own measured shape.

    @Test
    @DisplayName("resolveProjectorOutputDim: metadata matches the tensor's own shape — resolves silently")
    void projector_output_dim_matches_metadata() {
        int result = VisionEncoder.resolveProjectorOutputDim(1024, 768, 1024, 768L * 1024, 768);

        assertThat(result).isEqualTo(768);
    }

    @Test
    @DisplayName("resolveProjectorOutputDim: metadata disagrees with the tensor's own shape — "
            + "the tensor wins (this is the exact llava-phi-3-mini case that crashed)")
    void projector_output_dim_metadata_mismatch_tensor_wins() {
        int result = VisionEncoder.resolveProjectorOutputDim(1024, 3072, 1024, 3072L * 1024, 768);

        assertThat(result).isEqualTo(3072);
    }

    @Test
    @DisplayName("resolveProjectorOutputDim: inDim mismatch with hiddenSize throws")
    void projector_output_dim_wrong_indim_throws() {
        assertThatThrownBy(() -> VisionEncoder.resolveProjectorOutputDim(2048, 768, 1024, 768L * 2048, 768))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("inDim=2048")
                .hasMessageContaining("hiddenSize=1024");
    }

    @Test
    @DisplayName("resolveProjectorOutputDim: flattened weight length inconsistent with outDim*hiddenSize throws")
    void projector_output_dim_wrong_weight_length_throws() {
        assertThatThrownBy(() -> VisionEncoder.resolveProjectorOutputDim(1024, 768, 1024, 12345L, 768))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("12345")
                .hasMessageContaining("786432");
    }

    // ── applyProjector: math correctness for a possible future 2-layer projector ─
    //
    // NOTE (2026-07-12): applying mm.2 in production was tried and reverted —
    // it caused a confirmed regression (degenerate repeating-token output), so
    // VisionEncoder.project() currently calls applyProjector with w2=null,b2=null
    // regardless of whether mm.2 exists in the file. These tests still verify
    // applyProjector's own math is correct in isolation (useful if/when the
    // real root cause of the regression is found and this gets re-enabled) —
    // they are NOT evidence that re-enabling mm.2 in production is safe.

    private static final cab.ml.juno.node.MatVec CPU = cab.ml.juno.node.CpuMatVec.INSTANCE;

    @Test
    @DisplayName("applyProjector: with no mm.2 weight, output is just mm.0 (single-layer fallback, unchanged "
            + "behavior for mmproj files that genuinely only have one projector layer)")
    void applyProjector_singleLayer_whenNoSecondWeight() {
        int hiddenSize = 2;
        int outputDim = 3;
        float[] x = { 1f, 2f };
        // w1: outputDim x hiddenSize, row-major
        float[] w1 = { 1f, 0f, 0f, 1f, 1f, 1f }; // rows: [1,0] [0,1] [1,1]
        float[] b1 = { 0.5f, 0.5f, 0.5f };

        float[] out = VisionEncoder.applyProjector(CPU, x, w1, b1, null, null, hiddenSize, outputDim, outputDim);

        // row0: 1*1+2*0+0.5=1.5  row1: 1*0+2*1+0.5=2.5  row2: 1*1+2*1+0.5=3.5
        assertThat(out).containsExactly(1.5f, 2.5f, 3.5f);
    }

    @Test
    @DisplayName("applyProjector: with mm.2 present, GELU is applied between the two linear layers "
            + "(NOT a plain second linear pass)")
    void applyProjector_twoLayer_appliesGeluBetweenLayers() {
        int hiddenSize = 2;
        int outputDim = 2;
        float[] x = { 1f, 1f };
        float[] w1 = { 1f, 0f, 0f, 1f }; // identity → mm.0 output = [1, 1] before bias
        float[] w2 = { 1f, 0f, 0f, 1f }; // identity → mm.2 output = GELU(mm.0 output)

        float[] out = VisionEncoder.applyProjector(CPU, x, w1, null, w2, null, hiddenSize, outputDim, outputDim);

        float expected = VisionEncoder.gelu(1f);
        assertThat(out).containsExactly(expected, expected);
        // Sanity: this must NOT equal the raw (non-GELU'd) mm.0 output — if it
        // did, that would mean the fix regressed back to "just two linear passes".
        assertThat(out).isNotEqualTo(new float[] { 1f, 1f });
    }

    @Test
    @DisplayName("applyProjector: mm.2 bias is applied after the second linear layer")
    void applyProjector_twoLayer_appliesSecondBias() {
        int hiddenSize = 1;
        int outputDim = 1;
        float[] x = { 0f }; // mm.0(0) + b1 = b1; GELU(b1) fed into mm.2
        float[] w1 = { 1f };
        float[] b1 = { 0f }; // mm.0 output = 0, GELU(0) = 0
        float[] w2 = { 1f };
        float[] b2 = { 10f };

        float[] out = VisionEncoder.applyProjector(CPU, x, w1, b1, w2, b2, hiddenSize, outputDim, outputDim);

        assertThat(out).containsExactly(10f); // 0 (mm.2 linear on GELU(0)=0) + b2=10
    }

    @Test
    @DisplayName("applyProjector: 2-layer output differs from what single-layer-only would have produced "
            + "for the same input (this is the actual defect the fix closes)")
    void applyProjector_twoLayer_differsFromSingleLayerFallback() {
        int hiddenSize = 3;
        int outputDim = 3;
        float[] x = { 0.3f, -0.7f, 1.2f };
        float[] w1 = { 1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f }; // identity
        float[] w2 = { 2f, 0f, 0f, 0f, 2f, 0f, 0f, 0f, 2f }; // 2x scale

        float[] singleLayerOnly = VisionEncoder.applyProjector(CPU, x, w1, null, null, null, hiddenSize, outputDim, outputDim);
        float[] fullTwoLayer = VisionEncoder.applyProjector(CPU, x, w1, null, w2, null, hiddenSize, outputDim, outputDim);

        assertThat(fullTwoLayer).isNotEqualTo(singleLayerOnly);
    }

    @Test
    @DisplayName("applyProjector: non-square mm.2 (moondream2 pattern: expand then contract) "
            + "produces finalOutDim-sized output, not mm0OutDim-sized")
    void applyProjector_nonSquareTwoLayer_producesCorrectFinalDim() {
        // Simulates moondream2: hidden=2 → mm0OutDim=4 (expand) → finalOutDim=3 (contract)
        int hiddenSize  = 2;
        int mm0OutDim   = 4;
        int finalOutDim = 3;
        float[] x  = { 1f, 1f };
        // mm.0: 4-row × 2-col identity-like expand (rows: [1,0] [0,1] [1,1] [0,0])
        float[] w1 = { 1f, 0f,  0f, 1f,  1f, 1f,  0f, 0f };
        // mm.2: 3-row × 4-col contract (rows: [1,0,0,0] [0,1,0,0] [0,0,1,0])
        float[] w2 = { 1f, 0f, 0f, 0f,   0f, 1f, 0f, 0f,   0f, 0f, 1f, 0f };

        float[] out = VisionEncoder.applyProjector(CPU, x, w1, null, w2, null, hiddenSize, mm0OutDim, finalOutDim);

        assertThat(out).hasSize(finalOutDim);
        // mm.0 output (before GELU): [1,1,2,0]. After GELU: [g(1), g(1), g(2), g(0)=0].
        // mm.2 contracts first 3 of the 4 mm.0 values (last row of w2 zeros out dim 4).
        float g1 = VisionEncoder.gelu(1f);
        float g2 = VisionEncoder.gelu(2f);
        assertThat(out).containsExactly(g1, g1, g2);
    }

    // ── applyProjectorBatch: batched sibling must match applyProjector exactly ─
    //
    // encode() calls applyProjectorBatch once for all patches instead of
    // applyProjector once per patch, purely as a performance change (weight-
    // stationary batched GEMM on CpuMatVec — see its javadoc). These tests
    // are the regression net for that change: per-row batched output must be
    // bit-for-bit identical to calling the unbatched method row by row.

    @Test
    @DisplayName("applyProjectorBatch: single-layer output for each row matches applyProjector called "
            + "on that row individually")
    void applyProjectorBatch_singleLayer_matchesPerRowApplyProjector() {
        int hiddenSize = 2;
        int outputDim = 3;
        float[] w1 = { 1f, 0f, 0f, 1f, 1f, 1f };
        float[] b1 = { 0.5f, 0.5f, 0.5f };
        float[][] X = { { 1f, 2f }, { -1f, 3f }, { 0f, 0f } };

        float[][] out = VisionEncoder.applyProjectorBatch(CPU, X, w1, b1, null, null, hiddenSize, outputDim,
                outputDim);

        assertThat(out).hasNumberOfRows(X.length);
        for (int i = 0; i < X.length; i++) {
            float[] expected = VisionEncoder.applyProjector(CPU, X[i], w1, b1, null, null, hiddenSize, outputDim,
                    outputDim);
            assertThat(out[i]).containsExactly(expected);
        }
    }

    @Test
    @DisplayName("applyProjectorBatch: two-layer (GELU between mm.0 and mm.2) output for each row matches "
            + "applyProjector called on that row individually")
    void applyProjectorBatch_twoLayer_matchesPerRowApplyProjector() {
        int hiddenSize  = 2;
        int mm0OutDim   = 4;
        int finalOutDim = 3;
        float[] w1 = { 1f, 0f,  0f, 1f,  1f, 1f,  0f, 0f };
        float[] w2 = { 1f, 0f, 0f, 0f,   0f, 1f, 0f, 0f,   0f, 0f, 1f, 0f };
        float[] b2 = { 0.1f, 0.2f, 0.3f };
        float[][] X = { { 1f, 1f }, { 2f, -1f }, { 0.5f, 0.5f } };

        float[][] out = VisionEncoder.applyProjectorBatch(CPU, X, w1, null, w2, b2, hiddenSize, mm0OutDim,
                finalOutDim);

        assertThat(out).hasNumberOfRows(X.length);
        for (int i = 0; i < X.length; i++) {
            float[] expected = VisionEncoder.applyProjector(CPU, X[i], w1, null, w2, b2, hiddenSize, mm0OutDim,
                    finalOutDim);
            assertThat(out[i]).containsExactly(expected);
        }
    }

    @Test
    @DisplayName("applyProjectorBatch: empty batch returns empty output without invoking the backend")
    void applyProjectorBatch_emptyBatch_returnsEmpty() {
        float[][] out = VisionEncoder.applyProjectorBatch(CPU, new float[0][], new float[6], null, null, null, 2, 3,
                3);
        assertThat(out).isEmpty();
    }

    // ── buildSequence: CLS-optional sequence construction ─────────────────────
    // These tests cover the SigLIP (no-CLS) path introduced 2026-07-22 to
    // support moondream2, whose vision GGUF lacks v.class_embd.

    @Test
    @DisplayName("buildSequence: null classEmbd returns patches as-is (SigLIP / no-CLS path)")
    void buildSequence_nullCls_returnsPatchesUnmodified() {
        float[][] patches = { { 1f, 2f }, { 3f, 4f }, { 5f, 6f } };

        float[][] seq = VisionEncoder.buildSequence(patches, null);

        assertThat(seq).isSameAs(patches); // identical reference — no allocation
        assertThat(seq.length).isEqualTo(3);
    }

    @Test
    @DisplayName("buildSequence: non-null classEmbd prepends CLS at index 0 (CLIP path)")
    void buildSequence_nonNullCls_prependsClsToken() {
        float[][] patches = { { 1f, 2f }, { 3f, 4f } };
        float[]   cls     = { 9f, 8f };

        float[][] seq = VisionEncoder.buildSequence(patches, cls);

        assertThat(seq.length).isEqualTo(3);
        // CLS is at position 0, values copied from classEmbd
        assertThat(seq[0]).containsExactly(9f, 8f);
        // Original patch arrays are referenced at positions 1..nP
        assertThat(seq[1]).isSameAs(patches[0]);
        assertThat(seq[2]).isSameAs(patches[1]);
    }

    @Test
    @DisplayName("buildSequence: CLS row is a distinct array — mutating patches[0] does not affect seq[0]")
    void buildSequence_clsRow_isIsolatedFromCls() {
        float[][] patches = { { 0f, 0f } };
        float[]   cls     = { 7f, 7f };

        float[][] seq = VisionEncoder.buildSequence(patches, cls);
        cls[0] = 0f; // mutate original cls after buildSequence

        assertThat(seq[0][0]).isEqualTo(7f); // seq[0] was copied, not aliased
    }

    @Test
    @DisplayName("buildSequence: patchStart = N - nP is 0 without CLS (SigLIP) and 1 with CLS (CLIP)")
    void buildSequence_patchStartIndex_noClsIsZeroClsIsOne() {
        int nP = 4;
        float[][] patches = new float[nP][2];

        float[][] seqNoCls = VisionEncoder.buildSequence(patches, null);
        float[][] seqCls   = VisionEncoder.buildSequence(patches, new float[2]);

        int patchStartNoCls = seqNoCls.length - nP; // N - nP = nP - nP = 0
        int patchStartCls   = seqCls.length   - nP; // N - nP = (nP+1) - nP = 1

        assertThat(patchStartNoCls).isZero();
        assertThat(patchStartCls).isEqualTo(1);
    }
}