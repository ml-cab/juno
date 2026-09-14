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

/**
 * Signals that a GGUF-embedded chat template could not be parsed or rendered
 * by {@link MiniJinjaTemplate}'s restricted Jinja subset.
 *
 * <p>Callers ({@link GgufChatTemplateResolver}, {@link EmbeddedChatTemplate})
 * catch this (and any other {@link RuntimeException} the engine may throw for
 * malformed input) and fall back to a named {@link ChatTemplate} — this
 * exception must never propagate out of the chat-template resolution path.
 */
public final class MiniJinjaException extends RuntimeException {

	public MiniJinjaException(String message) {
		super(message);
	}

	public MiniJinjaException(String message, Throwable cause) {
		super(message, cause);
	}
}
