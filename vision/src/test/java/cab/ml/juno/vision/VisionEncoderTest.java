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
}