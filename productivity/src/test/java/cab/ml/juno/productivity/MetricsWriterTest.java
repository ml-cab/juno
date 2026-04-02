// Created by Yevhen Soldatov
// Initial implementation: 2026

package cab.ml.juno.productivity;

import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

class MetricsWriterTest {

    @Test
    void writesJsonWithModelsArray() throws IOException {
        Map<String, Double> metrics = new LinkedHashMap<>();
        metrics.put("jfr.file.bytes", 1234.0);
        MetricsSnapshot.ModelMetrics mm = new MetricsSnapshot.ModelMetrics(
                "TinyLlama", "models/tinyllama.gguf", "juno-tiny-20260101-120000.jfr", metrics);

        Path out = Files.createTempFile("metrics", ".json");
        MetricsWriter.write(out, List.of(mm));

        String json = Files.readString(out, StandardCharsets.UTF_8);
        Assertions.assertThat(json).contains("\"models\"");
        Assertions.assertThat(json).contains("\"TinyLlama\"");
        Assertions.assertThat(json).contains("\"jfrFile\"");
        Assertions.assertThat(json).contains("juno-tiny-20260101-120000.jfr");
        Assertions.assertThat(json).contains("\"jfr.file.bytes\"");
        Assertions.assertThat(json).contains("\n  \"runId\"");
        Assertions.assertThat(json).startsWith("{\n");
        Assertions.assertThat(json.trim()).endsWith("}");
    }
}
