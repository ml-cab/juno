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
package cab.ml.juno.registry;

import java.util.List;

/**
 * Resolves a client-supplied {@code "model"} field against {@link ModelRegistry},
 * shared by every REST handler (text chat, vision chat, ...) so they fail — or
 * fall back — the same way.
 *
 * <p>Plain OpenAI-style clients, `curl` one-liners copied from other projects'
 * docs, and this project's own example commands routinely send a {@code model}
 * value that does not match the exact loaded model id (which, in {@code
 * --local} mode, is the loaded GGUF's filename — see {@code
 * ConsoleMain.buildLocalModelRegistry}). Previously any mismatch was a hard
 * {@code 503 service_unavailable}, even when exactly one model was loaded and
 * there was therefore only one sane model the request could have meant.
 *
 * <p>Resolution rules, in order:
 * <ol>
 *   <li>no model loaded at all → error, "No model is currently loaded"
 *   <li>{@code requested} blank/null → the (first) loaded model, no warning
 *   <li>{@code requested} matches a loaded model id exactly → that model, no warning
 *   <li>{@code requested} matches nothing, exactly one model is loaded, and the
 *       caller opted into {@link FallbackPolicy#SINGLE_MODEL_FALLBACK} → fall
 *       back to that model, with a warning the caller should log (the fix for
 *       the "wrong model name in curl" failure mode on user-facing REST
 *       endpoints)
 *   <li>{@code requested} matches nothing and the caller opted into
 *       {@link FallbackPolicy#STRICT}, or more than one model is loaded → error,
 *       naming the requested id and listing what is actually loaded (ambiguous,
 *       or the caller asked for exactness — silently guessing would be wrong
 *       either way)
 * </ol>
 *
 * <p>Fallback is opt-in per call site, not universal: the native
 * {@code /v1/inference} API ({@code InferenceApiServer}) is typically driven
 * by generated clients rather than hand-typed `curl`, so an explicit
 * nonexistent model id is far more likely to be a real client bug worth
 * surfacing than a typo worth tolerating — it uses {@link FallbackPolicy#STRICT}.
 * The OpenAI-compatible chat endpoint and the vision chat endpoint use
 * {@link FallbackPolicy#SINGLE_MODEL_FALLBACK}.
 */
public final class ModelIdResolver {

    private ModelIdResolver() {}

    /** Whether {@link #resolve} may fall back to the sole loaded model on a name mismatch. */
    public enum FallbackPolicy {
        /** Any mismatch is an error, even with exactly one model loaded. */
        STRICT,
        /** A mismatch with exactly one model loaded falls back to it (with a warning). */
        SINGLE_MODEL_FALLBACK
    }

    /**
     * @param modelId      the model id to use, or {@code null} if unresolved (see {@link #isError()})
     * @param warning      non-null when a fallback was applied and the caller should log it; null otherwise
     * @param errorMessage non-null when resolution failed; null otherwise
     */
    public record Resolution(String modelId, String warning, String errorMessage) {
        public boolean isError() {
            return modelId == null;
        }
    }

    /** Equivalent to {@code resolve(registry, requested, FallbackPolicy.STRICT)}. */
    public static Resolution resolve(ModelRegistry registry, String requested) {
        return resolve(registry, requested, FallbackPolicy.STRICT);
    }

    public static Resolution resolve(ModelRegistry registry, String requested, FallbackPolicy policy) {
        if (registry == null)
            throw new IllegalArgumentException("registry must not be null");
        if (policy == null)
            throw new IllegalArgumentException("policy must not be null");

        List<String> loadedIds = registry.listModels().stream().map(ModelDescriptor::modelId)
                .filter(registry::isLoaded).toList();

        if (loadedIds.isEmpty())
            return new Resolution(null, null, "No model is currently loaded");

        String req = requested == null ? null : requested.strip();
        if (req == null || req.isEmpty())
            return new Resolution(loadedIds.get(0), null, null);

        if (loadedIds.contains(req))
            return new Resolution(req, null, null);

        if (policy == FallbackPolicy.SINGLE_MODEL_FALLBACK && loadedIds.size() == 1) {
            String only = loadedIds.get(0);
            String warning = "Requested model '" + req + "' is not loaded; falling back to the only loaded model '"
                    + only + "'. Pass \"model\":\"" + only + "\" (or omit \"model\") to silence this warning.";
            return new Resolution(only, warning, null);
        }

        return new Resolution(null, null,
                "Model '" + req + "' is not loaded. Loaded models: " + loadedIds);
    }
}