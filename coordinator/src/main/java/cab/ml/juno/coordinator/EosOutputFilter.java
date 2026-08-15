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

package cab.ml.juno.coordinator;

/**
 * Suppresses chat turn-end markers from streamed generation output.
 *
 * <p>
 * After {@code /train-qa}, models often emit the same end-of-turn string used in
 * the training completion ({@code </s>}, {@code <|end|>}, {@code <|im_end|>},
 * …). Those strings must stop generation and must not appear in
 * {@link GenerationResult#text()} or reach {@link TokenConsumer}.
 *
 * <p>
 * GgufTokenizer may surface a marker as one vocab piece, as a non-EOS token id
 * that decodes to the marker, or as several character-level pieces (e.g.
 * {@code "</"} + {@code "s"} + {@code ">"}). This filter:
 * <ul>
 * <li>truncates at the first complete marker (including markers followed by
 * trailing whitespace in the same piece)</li>
 * <li>holds back a trailing suffix that is a proper prefix of any marker so
 * partial pieces are never streamed</li>
 * </ul>
 */
final class EosOutputFilter {

	/**
	 * Markers for every chat template used by LoRA / inference:
	 * {@code </s>} (LLaMA/Mistral/TinyLlama), {@code <|endoftext|>},
	 * {@code <|end|>} (Phi-3), {@code <|eot_id|>} (LLaMA 3),
	 * {@code <end_of_turn>} (Gemma), {@code <|im_end|>} (ChatML/Qwen).
	 */
	static final String[] MARKERS = { "</s>", "<|endoftext|>", "<|end|>", "<|eot_id|>", "<end_of_turn>",
			"<|im_end|>" };

	private static final int MAX_MARKER_LEN;

	static {
		int max = 0;
		for (String m : MARKERS)
			max = Math.max(max, m.length());
		MAX_MARKER_LEN = max;
	}

	record Outcome(boolean stop, String emit) {
	}

	private final StringBuilder text = new StringBuilder();
	private int emittedLen;

	/**
	 * Accept one decoded piece. {@link Outcome#emit()} is safe to stream;
	 * {@link Outcome#stop()} means a turn-end marker was found and stripped.
	 */
	Outcome accept(String piece) {
		if (piece == null || piece.isEmpty())
			return new Outcome(false, "");
		text.append(piece);
		return drain(true);
	}

	/**
	 * End of generation: apply optional decoder flush, then release any held-back
	 * prefix that did not complete into a marker.
	 */
	Outcome finish(String tail) {
		if (tail != null && !tail.isEmpty())
			text.append(tail);
		Outcome o = drain(true);
		if (o.stop())
			return o;
		String emit = text.substring(emittedLen);
		emittedLen = text.length();
		return new Outcome(false, emit);
	}

	/** Accumulated text with any turn-end marker removed. */
	String text() {
		return text.toString();
	}

	/**
	 * Drop a held-back suffix that never completed into a marker (e.g. generation
	 * stopped on the real EOS token id after {@code "</"} was buffered).
	 */
	void discardHeld() {
		text.setLength(emittedLen);
	}

	private Outcome drain(boolean allowHoldback) {
		int markerAt = indexOfMarker(text);
		if (markerAt >= 0) {
			text.setLength(markerAt);
			String emit = text.substring(emittedLen);
			emittedLen = text.length();
			return new Outcome(true, emit);
		}
		int safe = allowHoldback ? safeEmitLength(text) : text.length();
		String emit = text.substring(emittedLen, safe);
		emittedLen = safe;
		return new Outcome(false, emit);
	}

	static int indexOfMarker(CharSequence text) {
		int best = -1;
		for (String marker : MARKERS) {
			int idx = indexOf(text, marker);
			if (idx >= 0 && (best < 0 || idx < best))
				best = idx;
		}
		return best;
	}

	/**
	 * Length of the prefix that cannot be the start of an unfinished marker.
	 * Holds back the longest trailing proper prefix of any {@link #MARKERS} entry.
	 */
	static int safeEmitLength(CharSequence text) {
		int len = text.length();
		if (len == 0)
			return 0;
		int from = Math.max(0, len - (MAX_MARKER_LEN - 1));
		for (int start = from; start < len; start++) {
			if (isProperPrefixOfMarker(text, start, len))
				return start;
		}
		return len;
	}

	private static boolean isProperPrefixOfMarker(CharSequence text, int start, int end) {
		int suffixLen = end - start;
		if (suffixLen <= 0)
			return false;
		for (String marker : MARKERS) {
			if (suffixLen >= marker.length())
				continue;
			if (regionEquals(text, start, end, marker, suffixLen))
				return true;
		}
		return false;
	}

	private static int indexOf(CharSequence haystack, String needle) {
		int n = needle.length();
		int limit = haystack.length() - n;
		outer: for (int i = 0; i <= limit; i++) {
			for (int j = 0; j < n; j++) {
				if (haystack.charAt(i + j) != needle.charAt(j))
					continue outer;
			}
			return i;
		}
		return -1;
	}

	private static boolean regionEquals(CharSequence text, int start, int end, String marker, int len) {
		for (int i = 0; i < len; i++) {
			if (text.charAt(start + i) != marker.charAt(i))
				return false;
		}
		return true;
	}
}
