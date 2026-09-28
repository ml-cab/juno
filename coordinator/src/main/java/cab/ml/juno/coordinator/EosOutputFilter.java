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

import cab.ml.juno.tokenizer.ChatTurnMarkers;

/**
 * Ends generation at the first chat turn boundary in the decoded output and
 * suppresses the marker itself.
 *
 * <p>
 * Two things end an assistant turn, and {@link ChatTurnMarkers} lists both. A
 * turn-end marker ({@code </s>}, {@code <|end|>}, {@code <|im_end|>}, …) is the
 * model saying it is finished — after {@code /train-qa} models often emit the
 * one used in the training completion instead of the real EOS token id. A
 * turn-opening role header ({@code <|user|>}, {@code <|im_start|>}, …) is the
 * model having run past its own turn into the next speaker's, which small chat
 * models do readily; the answer ended where the header began.
 *
 * <p>
 * Either way the marker must stop generation and must not appear in
 * {@link GenerationResult#text()} or reach {@link TokenConsumer} — otherwise the
 * fabricated continuation is streamed to the caller and stored as the
 * assistant's own words in the conversation history.
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
 *
 * <p>
 * Below a request's minimum token count the model may not end its turn, so a
 * marker completed then is ordinary text: {@link #accept(String, boolean)} with
 * {@code mayStop == false} emits it and never looks at it again.
 */
final class EosOutputFilter {

	record Outcome(boolean stop, String emit) {
	}

	private final StringBuilder text = new StringBuilder();
	private int emittedLen;
	/** Text before this index was passed over while ending was not allowed; never rescanned. */
	private int scanFloor;

	/**
	 * Accept one decoded piece. {@link Outcome#emit()} is safe to stream;
	 * {@link Outcome#stop()} means a turn marker was found and stripped.
	 */
	Outcome accept(String piece) {
		return accept(piece, true);
	}

	/**
	 * As {@link #accept(String)}; with {@code mayStop == false} (the request is
	 * still below its minimum) a complete marker does not stop and is emitted as
	 * text. A trailing marker prefix is still held back, so a marker that completes
	 * once stopping is allowed is caught whole.
	 */
	Outcome accept(String piece, boolean mayStop) {
		if (piece == null || piece.isEmpty())
			return new Outcome(false, "");
		text.append(piece);
		if (mayStop)
			return drain(true);
		int safe = safeEmitLength(text);
		String emit = text.substring(emittedLen, safe);
		emittedLen = safe;
		scanFloor = safe;
		return new Outcome(false, emit);
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

	/** Accumulated text with any turn marker removed. */
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
		// Everything below emittedLen was scanned by an earlier drain and holds no
		// marker, and no emitted text ever ends in a marker prefix (that is what the
		// hold-back guarantees). So a marker that is complete now must start within
		// one marker-length of emittedLen — rescanning the whole buffer every token
		// would make the decode loop quadratic in the length of the answer.
		int scanFrom = Math.max(scanFloor, emittedLen - (ChatTurnMarkers.MAX_LENGTH - 1));
		int markerAt = indexOfMarker(text, scanFrom);
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
		return indexOfMarker(text, 0);
	}

	static int indexOfMarker(CharSequence text, int from) {
		int best = -1;
		for (String marker : ChatTurnMarkers.ALL) {
			int idx = indexOf(text, marker, from);
			if (idx >= 0 && (best < 0 || idx < best))
				best = idx;
		}
		return best;
	}

	/**
	 * Length of the prefix that cannot be the start of an unfinished marker.
	 * Holds back the longest trailing proper prefix of any {@link ChatTurnMarkers}
	 * entry.
	 */
	static int safeEmitLength(CharSequence text) {
		int len = text.length();
		if (len == 0)
			return 0;
		int from = Math.max(0, len - (ChatTurnMarkers.MAX_LENGTH - 1));
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
		for (String marker : ChatTurnMarkers.ALL) {
			if (suffixLen >= marker.length())
				continue;
			if (regionEquals(text, start, end, marker, suffixLen))
				return true;
		}
		return false;
	}

	private static int indexOf(CharSequence haystack, String needle, int from) {
		int n = needle.length();
		int limit = haystack.length() - n;
		outer: for (int i = Math.max(0, from); i <= limit; i++) {
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
