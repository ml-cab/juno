package cab.ml.juno.vision;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("VisionConfig — metadata parsing and derived geometry")
class VisionConfigTest {

    @Test
    @DisplayName("numPatches = (imageSize / patchSize)^2")
    void num_patches_geometry() {
        VisionConfig cfg = VisionConfig.synthetic(336, 14, 1024, 24, 16, 4096);
        // (336 / 14)^2 = 24^2 = 576
        assertThat(cfg.numPatches()).isEqualTo(576);
    }

    @Test
    @DisplayName("numVisionTokens = numPatches + 1 CLS")
    void num_vision_tokens_includes_cls() {
        VisionConfig cfg = VisionConfig.synthetic(336, 14, 1024, 24, 16, 4096);
        assertThat(cfg.numVisionTokens()).isEqualTo(cfg.numPatches() + 1);
    }

    @Test
    @DisplayName("headDim = hiddenSize / numHeads")
    void head_dim_derived() {
        VisionConfig cfg = VisionConfig.synthetic(224, 16, 768, 12, 12, 512);
        assertThat(cfg.headDim()).isEqualTo(768 / 12);
    }

    @Test
    @DisplayName("toString contains key fields")
    void to_string_contains_fields() {
        VisionConfig cfg = VisionConfig.synthetic(336, 14, 1024, 24, 16, 4096);
        String s = cfg.toString();
        assertThat(s).contains("image=336");
        assertThat(s).contains("patch=14");
        assertThat(s).contains("hidden=1024");
        assertThat(s).contains("layers=24");
        assertThat(s).contains("proj=4096");
    }

    @Test
    @DisplayName("different patch sizes produce correct patch counts")
    void patch_size_variants() {
        // CLIP-B/32: 224 / 32 = 7  → 49 patches
        assertThat(VisionConfig.synthetic(224, 32, 768, 12, 12, 512).numPatches()).isEqualTo(49);
        // CLIP-L/14: 336 / 14 = 24 → 576 patches
        assertThat(VisionConfig.synthetic(336, 14, 1024, 24, 16, 4096).numPatches()).isEqualTo(576);
    }
}