// Created by Yevhen Soldatov
// Initial implementation: 2026

package cab.ml.juno.productivity;

import java.util.List;
import java.util.Objects;

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
