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

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The split a GPT-2 BPE vocabulary applies to text before its merges run, as
 * named by the {@code tokenizer.ggml.pre} metadata key.
 *
 * <p>
 * A BPE vocabulary is trained on text that was first cut into pre-tokens by a
 * fixed pattern, and merges are only ever learned inside one pre-token. Merging
 * a whole run of text in one pass therefore admits pairs the training never
 * produced: {@code "a  b"} becomes {@code "a"} + {@code "  "} + {@code "b"}
 * where the trained split gives {@code "a"} + {@code " "} + {@code " b"}, and a
 * digit run groups by whatever the merge table happens to allow rather than by
 * the rule the vocabulary was built with. Nothing throws when that happens; the
 * model simply receives a token sequence it was not trained on.
 *
 * <p>
 * Applying the declared split restores those boundaries. Each piece this class
 * returns is merged on its own, so no merge can cross a boundary.
 *
 * <p>
 * The set below is an allowlist of the types declared by real model files that
 * were checked against a second engine's tokenization of the same strings, not
 * the full catalogue of types that exist. A file declaring anything else is
 * rejected by name at load rather than tokenized under a split that was never
 * verified for it. A file declaring nothing, and a file declaring
 * {@code default}, keep the whole-run behaviour that predates this class: those
 * two are the same case, since a declared {@code default} is exactly what an
 * absent key means.
 *
 * <p>
 * SentencePiece vocabularies do not take a pre-tokenizer split at all, so
 * {@link GgufTokenizer} does not consult the key for them.
 *
 * <p>
 * Instances are immutable and safe to share across threads.
 */
final class BpePreTokenizer {

	/** Contraction suffixes, matched ahead of the letter run so they split off it. */
	private static final String CONTRACTIONS = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])";

	/**
	 * Shared tail: a punctuation run with at most one leading space and any line
	 * breaks after it, a line-break run, the whitespace before a word (leaving its
	 * last space to the word), and any remaining whitespace.
	 */
	private static final String TAIL = "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

	/** A letter run with at most one non-letter, non-digit character in front. */
	private static final String WORD = "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+";

	/**
	 * Declared {@code qwen2}: one piece per digit. Files on disk: the Qwen2.5,
	 * Qwen3 and Qwen3-Coder GGUFs.
	 */
	private static final BpePreTokenizer QWEN2 = new BpePreTokenizer("qwen2",
			CONTRACTIONS + "|" + WORD + "|\\p{N}" + TAIL);

	/**
	 * Declared {@code llama-bpe}: digits group up to three at a time. Files on
	 * disk: the Llama-3.2 llamafile.
	 */
	private static final BpePreTokenizer LLAMA_BPE = new BpePreTokenizer("llama-bpe",
			CONTRACTIONS + "|" + WORD + "|\\p{N}{1,3}" + TAIL);

	private final String typeName;
	private final Pattern pattern;

	private BpePreTokenizer(String typeName, String regex) {
		this.typeName = typeName;
		// UNICODE_CHARACTER_CLASS so that \s and \S cover non-ASCII whitespace,
		// which is how the trained splits classify it.
		this.pattern = Pattern.compile(regex, Pattern.UNICODE_CHARACTER_CLASS);
	}

	/**
	 * The pre-tokenizer for a declared {@code tokenizer.ggml.pre} value.
	 *
	 * @param declaredType the metadata value, or {@code null} when the file has no
	 *                     such key
	 * @return {@code null} when no split applies and the caller keeps merging each
	 *         segment as one run, otherwise the split to apply
	 * @throws UnsupportedPreTokenizerException naming the type when the file declares
	 *                                          one that has no verified implementation
	 */
	static BpePreTokenizer resolve(String declaredType) {
		if (declaredType == null)
			return null;
		String declared = declaredType.trim().toLowerCase(Locale.ROOT);
		if (declared.isEmpty() || "default".equals(declared))
			return null;
		return switch (declared) {
		case "qwen2" -> QWEN2;
		case "llama-bpe" -> LLAMA_BPE;
		default -> throw new UnsupportedPreTokenizerException("Unsupported pre-tokenizer type '" + declaredType
				+ "' in tokenizer.ggml.pre. Implemented: qwen2, llama-bpe; a file with no such key, or with "
				+ "'default', keeps the whole-run merge. Each type cuts text into different pre-tokens before "
				+ "the BPE merges run, so tokenizing this file under another type's split would feed the model "
				+ "a token sequence it was not trained on.");
		};
	}

	/** The declared value this instance implements. */
	String typeName() {
		return typeName;
	}

	/**
	 * Cuts {@code text} into the pre-tokens this type specifies, in order.
	 *
	 * <p>
	 * Every character of the input appears in exactly one piece. Text that no
	 * alternative matches — which the patterns above are written to make
	 * impossible — is emitted as its own piece rather than dropped, because a
	 * dropped character would silently shorten the prompt.
	 */
	List<String> split(String text) {
		List<String> pieces = new ArrayList<>();
		Matcher m = pattern.matcher(text);
		int consumed = 0;
		while (m.find()) {
			if (m.start() > consumed)
				pieces.add(text.substring(consumed, m.start()));
			if (m.end() > m.start())
				pieces.add(m.group());
			consumed = m.end();
		}
		if (consumed < text.length())
			pieces.add(text.substring(consumed));
		return pieces;
	}
}
