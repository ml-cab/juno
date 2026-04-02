// Created by Yevhen Soldatov
// Initial implementation: 2026

package cab.ml.juno.productivity;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.Map;

final class MetricsWriter {

    private static final String INDENT_UNIT = "  ";

    private MetricsWriter() {
    }

    static void write(Path output, List<MetricsSnapshot.ModelMetrics> models) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        appendIndent(sb, 1).append("\"runId\": \"").append(escape(Instant.now().toString())).append("\",\n");
        appendIndent(sb, 1).append("\"models\": [\n");
        for (int i = 0; i < models.size(); i++) {
            MetricsSnapshot.ModelMetrics m = models.get(i);
            if (i > 0) {
                sb.append(",\n");
            }
            appendIndent(sb, 2).append("{\n");
            appendIndent(sb, 3).append("\"name\": \"").append(escape(m.getName())).append("\",\n");
            appendIndent(sb, 3).append("\"path\": \"").append(escape(m.getPath())).append("\",\n");
            appendIndent(sb, 3).append("\"jfrFile\": \"").append(escape(m.getJfrFileName())).append("\",\n");
            appendIndent(sb, 3).append("\"metrics\": {\n");
            int j = 0;
            for (Map.Entry<String, Double> e : m.getMetrics().entrySet()) {
                if (j++ > 0) {
                    sb.append(",\n");
                }
                appendIndent(sb, 4).append('"').append(escape(e.getKey())).append("\": ").append(e.getValue());
            }
            sb.append('\n');
            appendIndent(sb, 3).append("}\n");
            appendIndent(sb, 2).append('}');
        }
        if (!models.isEmpty()) {
            sb.append('\n');
        }
        appendIndent(sb, 1).append("]\n");
        sb.append("}\n");

        Files.createDirectories(output.getParent());
        Files.writeString(output, sb.toString(), StandardCharsets.UTF_8);
    }

    private static StringBuilder appendIndent(StringBuilder sb, int level) {
        for (int i = 0; i < level; i++) {
            sb.append(INDENT_UNIT);
        }
        return sb;
    }

    private static String escape(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
