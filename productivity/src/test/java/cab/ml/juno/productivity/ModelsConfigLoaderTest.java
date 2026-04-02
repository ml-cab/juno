// Created by Yevhen Soldatov
// Initial implementation: 2026

package cab.ml.juno.productivity;

import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;

class ModelsConfigLoaderTest {

    private final ModelsConfigLoader loader = new ModelsConfigLoader();

    @Test
    void parsesValidModelsJsonFromMainModelsConfig() throws IOException, URISyntaxException {
        Path path = fromResource("models.json");

        ModelsConfig config = loader.load(path);

        Assertions.assertThat(config.getModels())
                .hasSize(1)
                .first()
                .satisfies(entry -> {
                    Assertions.assertThat(entry.getName()).isEqualTo("TinyLlama-1.1B-Chat-v1.0.Q4_K_M");
                    Assertions.assertThat(entry.getPath()).isEqualTo("TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf");
                });
    }

    @Test
    void rejectsMissingModelsArray() throws IOException, URISyntaxException {
        Path path = fromResource("models-missing-array.json");

        Assertions.assertThatThrownBy(() -> loader.load(path))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("non-empty 'models' array");
    }

    @Test
    void rejectsBlankNameOrPath() throws IOException, URISyntaxException {
        Path path = fromResource("models-blank-name.json");

        Assertions.assertThatThrownBy(() -> loader.load(path))
                .isInstanceOf(IllegalArgumentException.class);
    }

    private static Path fromResource(String name) throws URISyntaxException {
        var url = ModelsConfigLoaderTest.class.getClassLoader().getResource(name);
        if (url == null) {
            throw new IllegalStateException("Resource not found: " + name);
        }
        return Paths.get(url.toURI());
    }
}
