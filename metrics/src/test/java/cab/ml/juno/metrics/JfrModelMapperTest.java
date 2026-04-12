// Created by Yevhen Soldatov
// Initial implementation: 2026

package cab.ml.juno.metrics;

import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

class JfrModelMapperTest {

    @Test
    void extractsModelStemFromPath() {
        Assertions.assertThat(JfrModelMapper.modelStemFromPath("models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"))
                .isEqualTo("tinyllama-1.1b-chat-v1.0.Q4_K_M");
        Assertions.assertThat(JfrModelMapper.modelStemFromPath("/abs/phi-3.5-mini-instruct-Q4_K_M.gguf"))
                .isEqualTo("phi-3.5-mini-instruct-Q4_K_M");
    }

    @Test
    void mapsJfrFilesToModelsByStem() {
        ModelsConfig config = new ModelsConfig(List.of(
                new ModelsConfig.ModelEntry("TinyLlama", "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
                new ModelsConfig.ModelEntry("Phi3.5", "models/phi-3.5-mini-instruct-Q4_K_M.gguf")));

        Path jfr1 = Path.of("juno-tinyllama-1.1b-chat-v1.0.Q4_K_M-20260401-005454.jfr");
        Path jfr2 = Path.of("juno-phi-3.5-mini-instruct-Q4_K_M-20260401-010000.jfr");
        Path jfr3 = Path.of("unrelated.jfr");

        Map<Path, ModelsConfig.ModelEntry> mapped = JfrModelMapper.mapByModelStem(List.of(jfr1, jfr2, jfr3), config);

        Assertions.assertThat(mapped).hasSize(2).containsKeys(jfr1, jfr2);
        Assertions.assertThat(mapped.get(jfr1).getName()).isEqualTo("TinyLlama");
        Assertions.assertThat(mapped.get(jfr2).getName()).isEqualTo("Phi3.5");
    }

    @Test
    void mapsJfrToModelWhenManifestStemDiffersOnlyByCase() {
        ModelsConfig config = new ModelsConfig(List.of(
                new ModelsConfig.ModelEntry("TinyLlama", "/home/Models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf")));

        Path jfr = Path.of("juno-tinyllama-1.1b-chat-v1.0.Q4_K_M-20260402-170723.jfr");

        Map<Path, ModelsConfig.ModelEntry> mapped = JfrModelMapper.mapByModelStem(List.of(jfr), config);

        Assertions.assertThat(mapped).hasSize(1).containsKey(jfr);
        Assertions.assertThat(mapped.get(jfr).getName()).isEqualTo("TinyLlama");
    }
}
