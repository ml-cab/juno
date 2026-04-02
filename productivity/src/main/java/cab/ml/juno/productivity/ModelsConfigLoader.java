// Created by Yevhen Soldatov
// Initial implementation: 2026

package cab.ml.juno.productivity;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class ModelsConfigLoader {

    private static final Pattern MODEL_OBJECT_PATTERN =
            Pattern.compile("\\{[^}]*\"name\"\\s*:\\s*\"([^\"]+)\"[^}]*\"path\"\\s*:\\s*\"([^\"]+)\"[^}]*}");

    public ModelsConfig load(Path path) throws IOException {
        String json = Files.readString(path, StandardCharsets.UTF_8);
        Matcher matcher = MODEL_OBJECT_PATTERN.matcher(json);

        List<ModelsConfig.ModelEntry> entries = new ArrayList<>();
        while (matcher.find()) {
            String name = matcher.group(1).trim();
            String modelPath = matcher.group(2).trim();
            if (name.isEmpty() || modelPath.isEmpty()) {
                throw new IllegalArgumentException("models[name,path] must be non-blank strings");
            }
            entries.add(new ModelsConfig.ModelEntry(name, modelPath));
        }

        if (entries.isEmpty()) {
            throw new IllegalArgumentException("models.json must contain non-empty 'models' array");
        }

        return new ModelsConfig(entries);
    }
}
