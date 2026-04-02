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

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * @author Yevhen Soldatov
 */

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
