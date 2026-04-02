// Created by Yevhen Soldatov
// Initial implementation: 2026

package cab.ml.juno.productivity;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

final class JfrModelMapper {

    /**
     * {@code juno-&lt;modelFileStem&gt;-YYYYMMDD-HHMMSS.jfr}
     */
    private static final Pattern JFR_WITH_MODEL =
            Pattern.compile("^juno-(.+)-(\\d{8})-(\\d{6})\\.jfr$");

    private JfrModelMapper() {
    }

    static Map<Path, ModelsConfig.ModelEntry> mapByModelStem(List<Path> jfrFiles, ModelsConfig config) {
        Objects.requireNonNull(jfrFiles, "jfrFiles");
        Objects.requireNonNull(config, "config");

        Map<String, ModelsConfig.ModelEntry> byStem = new HashMap<>();
        for (ModelsConfig.ModelEntry entry : config.getModels()) {
            // Keys are lower-case so JFR names (from actual file basename) match models.json
            // even when casing differs (e.g. TinyLlama… in manifest vs tinyllama… on disk).
            String key = canonicalStemKey(modelStemFromPath(entry.getPath()));
            byStem.put(key, entry);
        }

        Map<Path, ModelsConfig.ModelEntry> result = new HashMap<>();
        for (Path jfr : jfrFiles) {
            String fileName = jfr.getFileName().toString();
            String stem = modelStemFromJfr(fileName);
            if (stem == null) {
                continue;
            }
            ModelsConfig.ModelEntry entry = byStem.get(canonicalStemKey(stem));
            if (entry != null) {
                result.put(jfr, entry);
            }
        }
        return result;
    }

    static String modelStemFromPath(String modelPath) {
        String fileName = Path.of(modelPath).getFileName().toString();
        int dot = fileName.lastIndexOf('.');
        return dot > 0 ? fileName.substring(0, dot) : fileName;
    }

    static String modelStemFromJfr(String jfrFileName) {
        Matcher m = JFR_WITH_MODEL.matcher(jfrFileName);
        if (!m.matches()) {
            return null;
        }
        return m.group(1);
    }

    static String canonicalStemKey(String stem) {
        return stem.toLowerCase(Locale.ROOT);
    }
}
