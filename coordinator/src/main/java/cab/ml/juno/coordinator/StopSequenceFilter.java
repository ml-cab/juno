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
 * Holds back streamed text that may complete into a configured OpenAI
 * {@code stop} sequence, then truncates at the first match.
 *
 * <p>Same holdback algorithm as {@link EosOutputFilter}, but with
 * request-supplied stop strings.
 */
final class StopSequenceFilter {

	record Outcome(boolean stop, String emit) {
	}

	private final String[] stops;
	private final int maxStopLen;
	private final StringBuilder text = new StringBuilder();
	private int emittedLen;

	StopSequenceFilter(String[] stops) {
		if (stops == null || stops.length == 0) {
			this.stops = new String[0];
			this.maxStopLen = 0;
		} else {
			this.stops = stops.clone();
			int max = 0;
			for (String s : this.stops)
				max = Math.max(max, s.length());
			this.maxStopLen = max;
		}
	}

	Outcome accept(String piece) {
		if (piece == null || piece.isEmpty())
			return new Outcome(false, "");
		if (stops.length == 0) {
			text.append(piece);
			String emit = text.substring(emittedLen);
			emittedLen = text.length();
			return new Outcome(false, emit);
		}
		text.append(piece);
		return drain(true);
	}

	Outcome finish(String tail) {
		if (tail != null && !tail.isEmpty())
			text.append(tail);
		if (stops.length == 0) {
			String emit = text.substring(emittedLen);
			emittedLen = text.length();
			return new Outcome(false, emit);
		}
		Outcome o = drain(true);
		if (o.stop())
			return o;
		String emit = text.substring(emittedLen);
		emittedLen = text.length();
		return new Outcome(false, emit);
	}

	String text() {
		return text.toString();
	}

	void discardHeld() {
		text.setLength(emittedLen);
	}

	private Outcome drain(boolean allowHoldback) {
		int stopAt = indexOfStop(text);
		if (stopAt >= 0) {
			text.setLength(stopAt);
			String emit = text.substring(emittedLen);
			emittedLen = text.length();
			return new Outcome(true, emit);
		}
		int safe = allowHoldback ? safeEmitLength(text) : text.length();
		String emit = text.substring(emittedLen, safe);
		emittedLen = safe;
		return new Outcome(false, emit);
	}

	private int indexOfStop(CharSequence haystack) {
		int best = -1;
		for (String stop : stops) {
			int idx = indexOf(haystack, stop);
			if (idx >= 0 && (best < 0 || idx < best))
				best = idx;
		}
		return best;
	}

	private int safeEmitLength(CharSequence haystack) {
		int len = haystack.length();
		if (len == 0 || maxStopLen <= 1)
			return len;
		int from = Math.max(0, len - (maxStopLen - 1));
		for (int start = from; start < len; start++) {
			if (isProperPrefixOfStop(haystack, start, len))
				return start;
		}
		return len;
	}

	private boolean isProperPrefixOfStop(CharSequence text, int start, int end) {
		int suffixLen = end - start;
		if (suffixLen <= 0)
			return false;
		for (String stop : stops) {
			if (suffixLen >= stop.length())
				continue;
			if (regionEquals(text, start, stop, suffixLen))
				return true;
		}
		return false;
	}

	private static int indexOf(CharSequence haystack, String needle) {
		int n = needle.length();
		if (n == 0)
			return -1;
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

	private static boolean regionEquals(CharSequence text, int start, String stop, int len) {
		for (int i = 0; i < len; i++) {
			if (text.charAt(start + i) != stop.charAt(i))
				return false;
		}
		return true;
	}
}
