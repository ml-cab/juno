package cab.ml.juno.vision;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Path;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("VisionModelPaths — resolves which GGUF holds the CLIP vision tensors")
class VisionModelPathsTest {

    private static final Path MODEL = Path.of("/models/llava-v1.5-7b-Q4_K_M.gguf");
    private static final Path MMPROJ = Path.of("/models/mmproj-model-f16.gguf");

    @Test
    @DisplayName("with an explicit mmproj path, vision tensors are read from the mmproj file, not the model file")
    void separateMmprojIsUsedForVisionWeights() {
        VisionModelPaths paths = VisionModelPaths.of(MODEL, MMPROJ);

        assertThat(paths.textModelPath()).isEqualTo(MODEL);
        assertThat(paths.visionWeightsPath()).isEqualTo(MMPROJ);
        assertThat(paths.usesSeparateMmproj()).isTrue();
    }

    @Test
    @DisplayName("without an mmproj path, the model file itself is probed (merged-file fallback)")
    void missingMmprojFallsBackToModelFile() {
        VisionModelPaths paths = VisionModelPaths.of(MODEL, null);

        assertThat(paths.textModelPath()).isEqualTo(MODEL);
        assertThat(paths.visionWeightsPath()).isEqualTo(MODEL);
        assertThat(paths.usesSeparateMmproj()).isFalse();
    }

    @Test
    @DisplayName("null model path is rejected")
    void nullModelPathRejected() {
        assertThatThrownBy(() -> VisionModelPaths.of(null, MMPROJ))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    @DisplayName("mmproj path equal to model path is treated as a merged file, not a separate one")
    void identicalPathsAreNotConsideredSeparate() {
        VisionModelPaths paths = VisionModelPaths.of(MODEL, MODEL);

        assertThat(paths.usesSeparateMmproj()).isFalse();
    }
}