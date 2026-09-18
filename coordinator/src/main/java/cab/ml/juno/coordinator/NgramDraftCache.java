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

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Ngram-simple speculative-decoding draft source: indexes every {@code n}-token
 * window seen so far (prompt + generated tokens) against the token that
 * followed it, then proposes future tokens by chaining that lookup.
 *
 * <p>No second model and no static corpus — draft quality comes entirely from
 * repeated substrings within the <em>same</em> request (templated JSON,
 * repeated boilerplate, echoed context). Free-form novel text mostly misses and
 * the caller falls back to plain one-token decoding for that step; see
 * {@code PLAN-Infra-Tier9.md}'s exit gate ("measurable TPS gain on a repetitive
 * workload, or documented neutrality on natural text").
 *
 * <p>Not thread-safe — one instance per in-flight {@code generate()} call,
 * matching {@link GenerationLoop}'s per-request state.
 */
final class NgramDraftCache {

	private static final int MAX_ENTRIES = 4096;

	private final int n;
	private final Map<Key, Integer> nextTokenByNgram = new LinkedHashMap<>(16, 0.75f, true) {
		@Override
		protected boolean removeEldestEntry(Map.Entry<Key, Integer> eldest) {
			return size() > MAX_ENTRIES;
		}
	};
	private int indexedLength = 0;

	NgramDraftCache(int n) {
		if (n < 1)
			throw new IllegalArgumentException("n must be >= 1, got: " + n);
		this.n = n;
	}

	/**
	 * Index every new {@code n}-gram to next-token pair that {@code tokens}
	 * exposes since the last call. Safe to call repeatedly as the sequence grows —
	 * only the newly-appended suffix is indexed each time.
	 */
	void observe(int[] tokens) {
		int start = Math.max(indexedLength, n);
		for (int i = start; i < tokens.length; i++) {
			nextTokenByNgram.put(new Key(tokens, i - n, n), tokens[i]);
		}
		if (tokens.length > indexedLength)
			indexedLength = tokens.length;
	}

	/**
	 * Propose up to {@code maxDraft} future tokens by chaining ngram lookups
	 * starting from the last {@code n} tokens of {@code tokens}.
	 *
	 * @return a draft of length {@code [0, maxDraft]}; empty when {@code tokens}
	 *         is shorter than {@code n} or the very first lookup misses
	 */
	int[] propose(int[] tokens, int maxDraft) {
		if (tokens.length < n || maxDraft <= 0)
			return new int[0];

		int[] window = Arrays.copyOfRange(tokens, tokens.length - n, tokens.length);
		int[] draft = new int[maxDraft];
		int count = 0;
		while (count < maxDraft) {
			Integer next = nextTokenByNgram.get(new Key(window, 0, n));
			if (next == null)
				break;
			draft[count++] = next;
			System.arraycopy(window, 1, window, 0, n - 1);
			window[n - 1] = next;
		}
		return count == maxDraft ? draft : Arrays.copyOf(draft, count);
	}

	/** Number of distinct ngrams currently indexed — for tests / diagnostics. */
	int size() {
		return nextTokenByNgram.size();
	}

	/**
	 * Immutable {@code n}-token window key with content-based equality. A record
	 * over {@code int[]} would use reference equality for the array component, so
	 * this needs a hand-written {@code equals}/{@code hashCode}.
	 */
	private static final class Key {
		private final int[] values;
		private final int hash;

		Key(int[] source, int offset, int length) {
			this.values = Arrays.copyOfRange(source, offset, offset + length);
			this.hash = Arrays.hashCode(values);
		}

		@Override
		public boolean equals(Object o) {
			return o instanceof Key other && Arrays.equals(values, other.values);
		}

		@Override
		public int hashCode() {
			return hash;
		}
	}
}
