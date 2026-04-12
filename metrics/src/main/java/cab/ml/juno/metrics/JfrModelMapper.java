/*
 * Copyright 2026 Dmytro Soloviov (soulaway)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package cab.ml.juno.metrics;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * @author Yevhen Soldatov
 */

final class JfrModelMapper {

    /**
     * {@code juno-&lt;modelFileStem&gt;-YYYYMMDD-HHMMSS.jfr}
     */
    /**
     * Matches both naming conventions:
     * <ul>
     *   <li>{@code juno-<modelStem>-YYYYMMDD-HHMMSS.jfr} — local {@code run.sh} recordings</li>
     *   <li>{@code <modelStem>-<nodeId>-YYYYMMDD-HHMMSS.jfr} — AWS deploy recordings (no prefix)</li>
     * </ul>
     * The {@code (?:juno-)?} makes the prefix optional so group 1 always captures just the model stem.
     */
    private static final Pattern JFR_WITH_MODEL =
            Pattern.compile("^(?:juno-)?(.+)-(\\d{8})-(\\d{6})\\.jfr$");

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