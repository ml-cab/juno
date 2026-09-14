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
package cab.ml.juno.tokenizer;

import java.util.concurrent.ConcurrentHashMap;

/**
 * Process-local registry of resolved {@link ChatTemplate}s keyed by the same
 * model-type string used as {@code InferenceRequest.modelId()} /
 * {@code ChatTemplate.forModelType(String)}'s argument.
 *
 * <p>{@link ChatTemplateFormatter#forModelType(String)} consults this registry
 * before falling through to the named-template lookup — this is how a
 * GGUF-embedded template (resolved once at model load, via
 * {@link GgufChatTemplateResolver}) reaches every downstream call site
 * ({@code GenerationLoop}, {@code ContinuousBatchEngine}, the embedding
 * facade) without changing their code.
 *
 * <p><b>Scoped opt-in, not automatic:</b> nothing populates this registry
 * implicitly just because a GGUF happens to carry embedded template metadata.
 * Callers decide per product surface whether to register — the base
 * text-inference REPL entry points do; {@code --lora-play} and LoRA train
 * intentionally do not (train-time and inference-time formatting must stay
 * identical for adapter recall — see the feature/surface interaction matrix in
 * {@code docs/infra-plan/PLAN-Infra-Tier7.md}), so
 * for those surfaces this registry simply stays empty and
 * {@code forModelType} behaves exactly as before this feature existed.
 *
 * <p>Each CLI invocation is a fresh JVM, so the registry's lifetime is
 * naturally scoped to one {@code ./juno ...} run — there is no cross-session
 * leakage to guard against in production; {@link #clear()} exists for test
 * isolation.
 */
public final class EmbeddedChatTemplateRegistry {

	private static final ConcurrentHashMap<String, ChatTemplate> OVERRIDES = new ConcurrentHashMap<>();

	private EmbeddedChatTemplateRegistry() {
	}

	/** Register (or replace) the resolved template to use for {@code modelTypeKey}. */
	public static void register(String modelTypeKey, ChatTemplate template) {
		if (modelTypeKey == null || modelTypeKey.isBlank() || template == null)
			return;
		OVERRIDES.put(modelTypeKey, template);
	}

	/** The registered override for {@code modelTypeKey}, or {@code null} if none. */
	public static ChatTemplate lookup(String modelTypeKey) {
		return modelTypeKey == null ? null : OVERRIDES.get(modelTypeKey);
	}

	/** Test-only hygiene — clears all registered overrides. */
	public static void clear() {
		OVERRIDES.clear();
	}
}
