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

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * @author Yevhen Soldatov
 */

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
