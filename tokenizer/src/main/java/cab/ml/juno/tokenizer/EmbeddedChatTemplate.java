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

import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A {@link ChatTemplate} backed by a GGUF-embedded {@code tokenizer.chat_template}
 * Jinja-subset template ({@link MiniJinjaTemplate}), with a named template as a
 * safety-net fallback.
 *
 * <p>{@link #format} never throws: any render-time failure (a construct the
 * restricted engine does not support, an unexpected message shape) is caught
 * and silently delegates to the fallback named template instead — "no crash,
 * no silent corruption" per the chat-template resolver design.
 * Construction-time (parse and
 * smoke-render) validation lives in {@link GgufChatTemplateResolver#resolve};
 * by the time an instance of this class exists, the template has already
 * proven it can render at least one representative conversation.
 */
final class EmbeddedChatTemplate implements ChatTemplate {

	private static final Logger log = Logger.getLogger(EmbeddedChatTemplate.class.getName());

	private final MiniJinjaTemplate mini;
	private final ChatTemplate fallback;
	private final String bosToken;
	private final String eosToken;

	EmbeddedChatTemplate(MiniJinjaTemplate mini, ChatTemplate fallback, String bosToken, String eosToken) {
		this.mini = mini;
		this.fallback = fallback;
		this.bosToken = bosToken == null ? "" : bosToken;
		this.eosToken = eosToken == null ? "" : eosToken;
	}

	@Override
	public String format(List<ChatMessage> messages) {
		try {
			String out = mini.render(GgufChatTemplateResolver.buildContext(messages, bosToken, eosToken));
			if (out == null || out.isBlank())
				throw new MiniJinjaException("GGUF embedded chat template produced empty output");
			return out;
		} catch (RuntimeException e) {
			log.log(Level.FINE, e, () -> "GGUF embedded chat template render failed at runtime, falling back to '"
					+ fallback.modelType() + "'");
			return fallback.format(messages);
		}
	}

	/**
	 * Reports the fallback's family so JFR ({@link TemplateFormatEvent}) and
	 * training-key comparisons keep grouping by model family; the fact that this
	 * particular instance is GGUF-embedded is an implementation detail, not part
	 * of the {@link ChatTemplate} contract.
	 */
	@Override
	public String modelType() {
		return fallback.modelType();
	}

	/** True for any instance of this class — used by trace logging. */
	boolean isEmbedded() {
		return true;
	}
}
