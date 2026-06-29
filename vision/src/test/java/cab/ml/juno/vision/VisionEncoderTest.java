package cab.ml.juno.vision;

import static org.assertj.core.api.Assertions.assertThat;
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
}