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

import java.util.List;
import java.util.Objects;

/**
 * @author Yevhen Soldatov
 */

public final class ModelsConfig {

    public static final class ModelEntry {
        private final String name;
        private final String path;

        public ModelEntry(String name, String path) {
            this.name = Objects.requireNonNull(name, "name");
            this.path = Objects.requireNonNull(path, "path");
        }

        public String getName() {
            return name;
        }

        public String getPath() {
            return path;
        }
    }

    private final List<ModelEntry> models;

    public ModelsConfig(List<ModelEntry> models) {
        this.models = List.copyOf(Objects.requireNonNull(models, "models"));
    }

    public List<ModelEntry> getModels() {
        return models;
    }
}
