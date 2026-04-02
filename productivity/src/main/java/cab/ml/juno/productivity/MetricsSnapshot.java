// Created by Yevhen Soldatov
// Initial implementation: 2026

package cab.ml.juno.productivity;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

public final class MetricsSnapshot {

    public static final class ModelMetrics {
        private final String name;
        private final String path;
        private final String jfrFileName;
        private final Map<String, Double> metrics;

        public ModelMetrics(String name, String path, String jfrFileName, Map<String, Double> metrics) {
            this.name = name;
            this.path = path;
            this.jfrFileName = jfrFileName;
            this.metrics = Collections.unmodifiableMap(new LinkedHashMap<>(metrics));
        }

        public String getName() {
            return name;
        }

        public String getPath() {
            return path;
        }

        public String getJfrFileName() {
            return jfrFileName;
        }

        public Map<String, Double> getMetrics() {
            return metrics;
        }
    }
}
