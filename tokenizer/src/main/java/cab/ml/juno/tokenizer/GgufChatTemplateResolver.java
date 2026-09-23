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

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import cab.ml.juno.node.GgufReader;

/**
 * Resolves the effective {@link ChatTemplate} for a GGUF model: prefer the
 * model's own embedded {@code tokenizer.chat_template} metadata (rendered
 * through the restricted {@link MiniJinjaTemplate} engine) and fall back to
 * the named {@link ChatTemplate#forModelType(String)} lookup when the metadata
 * is absent, fails to parse, or fails a smoke-render.
 *
 * <p>Precedence:
 * <ol>
 * <li>metadata present, parses, and renders a non-blank probe conversation →
 * use the embedded template ({@link EmbeddedChatTemplate}, which itself falls
 * back per-call on any later render failure)
 * <li>otherwise → the existing named {@link ChatTemplate#forModelType(String)}
 * path, unchanged
 * </ol>
 *
 * <p>This class never throws for a malformed or missing embedded template —
 * every failure path returns the named fallback instead.
 */
public final class GgufChatTemplateResolver {

	private static final Logger log = Logger.getLogger(GgufChatTemplateResolver.class.getName());

	/** GGUF metadata key holding the model's own Jinja chat template, if any. */
	public static final String METADATA_KEY = "tokenizer.chat_template";

	private static final List<ChatMessage> PROBE_MESSAGES = List.of(
			ChatMessage.system("You are a helpful assistant."), ChatMessage.user("Hello"));

	private GgufChatTemplateResolver() {
	}

	/**
	 * True when {@code reader}'s metadata carries a non-blank
	 * {@value #METADATA_KEY} — used to decide whether an "embedded template
	 * ignored on this surface" notice is warranted (e.g. LoRA train / play, which
	 * intentionally keep using named templates so train-time and inference-time
	 * formatting stay identical).
	 */
	public static boolean hasEmbeddedTemplate(GgufReader reader) {
		if (reader == null)
			return false;
		String t = reader.metaString(METADATA_KEY);
		return t != null && !t.isBlank();
	}

	/**
	 * Resolve the effective template for an already-open {@link GgufReader},
	 * falling back to {@code ChatTemplate.forModelType(fallbackModelType)} when
	 * {@value #METADATA_KEY} is absent, unparseable, or fails a smoke render.
	 */
	public static ChatTemplate resolve(GgufReader reader, String fallbackModelType) {
		ChatTemplate fallback = ChatTemplate.forModelType(fallbackModelType);
		if (reader == null)
			return fallback;
		String src = reader.metaString(METADATA_KEY);
		if (src == null || src.isBlank())
			return fallback;
		try {
			MiniJinjaTemplate mini = new MiniJinjaTemplate(src);
			String bos = resolveTokenString(reader, "tokenizer.ggml.bos_token_id");
			String eos = resolveTokenString(reader, "tokenizer.ggml.eos_token_id");
			// Validate: smoke-render a representative conversation before committing to
			// the embedded template — "metadata -> render -> validate -> fallback".
			String probe = mini.render(buildContext(PROBE_MESSAGES, bos, eos));
			if (probe == null || probe.isBlank())
				throw new MiniJinjaException("embedded chat template rendered a blank probe conversation");
			return new EmbeddedChatTemplate(mini, fallback, bos, eos);
		} catch (RuntimeException e) {
			log.log(Level.FINE, e,
					() -> "GGUF embedded chat template unusable, falling back to named template '"
							+ fallback.modelType() + "'");
			return fallback;
		}
	}

	/**
	 * Convenience overload: open the GGUF at {@code modelPath}, resolve, and
	 * close it. Any I/O failure also falls back to the named template rather than
	 * propagating — a missing/unreadable file is reported elsewhere (model load),
	 * not by chat-template resolution.
	 */
	public static ChatTemplate resolveFromPath(String modelPath, String fallbackModelType) {
		if (modelPath == null)
			return ChatTemplate.forModelType(fallbackModelType);
		try (GgufReader reader = GgufReader.open(Path.of(modelPath))) {
			return resolve(reader, fallbackModelType);
		} catch (IOException | RuntimeException e) {
			log.log(Level.FINE, e, () -> "Could not read GGUF metadata from " + modelPath
					+ " for chat-template resolution; using named template");
			return ChatTemplate.forModelType(fallbackModelType);
		}
	}

	private static String resolveTokenString(GgufReader reader, String idKey) {
		try {
			long id = reader.metaLong(idKey, -1);
			if (id < 0)
				return "";
			Object toksObj = reader.meta("tokenizer.ggml.tokens");
			if (toksObj instanceof Object[] toks && id < toks.length) {
				Object t = toks[(int) id];
				return t == null ? "" : t.toString();
			}
		} catch (RuntimeException ignore) {
			// Token-string resolution is best-effort; bos/eos default to "".
		}
		return "";
	}

	/** Shared context builder used by both the smoke-render probe and {@link EmbeddedChatTemplate}. */
	static Map<String, Object> buildContext(List<ChatMessage> messages, String bosToken, String eosToken) {
		List<Map<String, Object>> msgs = new ArrayList<>(messages.size());
		for (ChatMessage m : messages) {
			Map<String, Object> mm = new HashMap<>();
			mm.put("role", m.role());
			mm.put("content", m.content());
			msgs.add(mm);
		}
		Map<String, Object> ctx = new HashMap<>();
		ctx.put("messages", msgs);
		ctx.put("bos_token", bosToken == null ? "" : bosToken);
		ctx.put("eos_token", eosToken == null ? "" : eosToken);
		ctx.put("add_generation_prompt", Boolean.TRUE);
		return ctx;
	}
}
