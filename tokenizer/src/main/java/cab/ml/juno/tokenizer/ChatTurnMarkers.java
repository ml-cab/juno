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
 * The literal strings that mark a chat turn boundary in every template Juno
 * formats prompts with. Generation ends at the first one that appears in the
 * decoded output, and the marker itself is never streamed.
 *
 * <p>Two groups, both of which terminate the assistant's turn:
 *
 * <ul>
 * <li>{@link #TURN_END} — the marker a template appends <em>after</em> the
 * assistant's content ({@code </s>}, {@code <|im_end|>}, …). The model emitting
 * one means "my answer is finished". Most models close a turn with the real EOS
 * token id instead, which {@code GenerationLoop} catches before decoding; these
 * strings cover the models (notably LoRA-tuned ones) that emit the marker as
 * ordinary vocabulary pieces.
 * <li>{@link #ROLE_OPEN} — the header a template writes <em>in front of</em> a
 * role's content ({@code <|user|>}, {@code <|im_start|>}, …). A model that emits
 * one has run past the end of its own turn and started writing somebody else's:
 * without this group a small model answers "Hello", then invents the user's next
 * question and answers that too, and the whole fabricated dialogue is streamed to
 * the caller and appended to the conversation history as if the assistant had
 * said it. The answer really ended at the header, so generation stops there.
 * </ul>
 *
 * <p>Only unambiguous bracketed template tokens belong here. Mistral's
 * {@code [INST]} turn opener is deliberately absent: plain square brackets are
 * common enough in ordinary prose and code to truncate legitimate answers, and
 * the Mistral template closes the assistant turn with {@code </s>} anyway.
 */
public final class ChatTurnMarkers {

	/** Turn-end markers: LLaMA/Mistral/TinyLlama, GPT-2 style, Phi-3, LLaMA 3, Gemma, ChatML/Qwen. */
	public static final String[] TURN_END = { "</s>", "<|endoftext|>", "<|end|>", "<|eot_id|>", "<end_of_turn>",
			"<|im_end|>" };

	/** Turn-opening role headers: TinyLlama/Zephyr/Phi-3, ChatML/Qwen, LLaMA 3, Gemma. */
	public static final String[] ROLE_OPEN = { "<|user|>", "<|assistant|>", "<|system|>", "<|im_start|>",
			"<|start_header_id|>", "<start_of_turn>" };

	/** {@link #TURN_END} followed by {@link #ROLE_OPEN}, in one array to scan. */
	public static final String[] ALL;

	/** Length of the longest entry in {@link #ALL} — the hold-back window width. */
	public static final int MAX_LENGTH;

	static {
		String[] all = new String[TURN_END.length + ROLE_OPEN.length];
		System.arraycopy(TURN_END, 0, all, 0, TURN_END.length);
		System.arraycopy(ROLE_OPEN, 0, all, TURN_END.length, ROLE_OPEN.length);
		ALL = all;
		int max = 0;
		for (String m : all)
			max = Math.max(max, m.length());
		MAX_LENGTH = max;
	}

	/** True when {@code piece} is a turn-end marker — the model saying it has finished. */
	public static boolean isTurnEnd(String piece) {
		for (String m : TURN_END)
			if (m.equals(piece))
				return true;
		return false;
	}

	/** True when {@code piece} is a turn-end marker or a turn-opening role header. */
	public static boolean isTurnMarker(String piece) {
		if (isTurnEnd(piece))
			return true;
		for (String m : ROLE_OPEN)
			if (m.equals(piece))
				return true;
		return false;
	}

	private ChatTurnMarkers() {
	}
}
