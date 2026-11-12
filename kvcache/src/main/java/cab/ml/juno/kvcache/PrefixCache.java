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

package cab.ml.juno.kvcache;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Trie-based prefix cache for shared token sequences.
 *
 * Problem: 16 clients all send the same 500-token system prompt. Without prefix
 * caching: each request recomputes those 500 tokens. With prefix caching: first
 * request computes + caches, rest skip forward to token 501.
 *
 * Usage in the inference loop: 1. findLongestPrefix(tokens) → PrefixMatch
 * (matched length + KV block refs) 2. If match.length > 0: start forward pass
 * at match.length (skip matched prefix) 3. After generation:
 * cachePrefix(tokens, kvBlockRefs)
 *
 * Thread-safe via ReadWriteLock — many concurrent reads, exclusive writes.
 *
 * <p>An entry belongs to the KV key that cached it: the positions it covers were written under that key and nowhere
 * else. A request reuses only its own key's entries ({@link #findLongestPrefix(int[], String)}); several keys may cache
 * the same tokens, and each keeps its entry until it is invalidated. A key holds one entry, its latest: caching again
 * replaces it, because a later prompt that branched off the earlier one has overwritten the earlier one's positions
 * past the branch.
 */
public final class PrefixCache {

	private final TrieNode root = new TrieNode();
	/** Each key's entry: the tokens of the path that ends at the node holding the key. */
	private final Map<String, int[]> entries = new HashMap<>();
	private final ReadWriteLock lock = new ReentrantReadWriteLock();
	private final LongAdder lookups = new LongAdder();
	private final LongAdder hits = new LongAdder();

	/**
	 * Find the longest prefix of the given token sequence that any key cached, with the key that cached it (the
	 * earliest of several). The positions it covers live under that key, not the caller's: a request that resumes
	 * from its own KV looks up with {@link #findLongestPrefix(int[], String)}.
	 *
	 * @param tokens input token IDs
	 * @return PrefixMatch with the matched length (0 if no match) and cache key
	 */
	public PrefixMatch findLongestPrefix(int[] tokens) {
		return find(tokens, null);
	}

	/**
	 * Find the longest prefix of the given token sequence that {@code ownerKey} cached: the positions a request
	 * keyed {@code ownerKey} has already written and may skip. A {@code null} key matches nothing.
	 *
	 * @param tokens   input token IDs
	 * @param ownerKey the KV key the request's positions are written under
	 * @return PrefixMatch with the matched length (0 if no match) and {@code ownerKey}
	 */
	public PrefixMatch findLongestPrefix(int[] tokens, String ownerKey) {
		if (ownerKey == null) {
			lookups.increment();
			return PrefixMatch.empty();
		}
		return find(tokens, ownerKey);
	}

	/**
	 * The longest prefix {@code tokens} has in common with any key's entry, whether or not the entry ends there, and
	 * a key whose entry runs through it: that key's positions {@code [0, matchedTokens)} hold exactly these tokens.
	 * Unlike {@link #findLongestPrefix(int[])}, a request sharing only the opening of a longer cached prompt (its
	 * system prompt, say) matches that opening.
	 */
	public PrefixMatch findLongestCommonPrefix(int[] tokens) {
		lookups.increment();
		if (tokens == null || tokens.length == 0)
			return PrefixMatch.empty();
		lock.readLock().lock();
		try {
			TrieNode current = root;
			int depth = 0;
			for (int token : tokens) {
				TrieNode next = current.children.get(token);
				if (next == null)
					break;
				current = next;
				depth++;
			}
			if (depth == 0)
				return PrefixMatch.empty();
			// A node is kept only while an entry ends at it or below it, so this descent ends at an owner.
			TrieNode below = current;
			while (below.owners == null)
				below = below.children.values().iterator().next();
			hits.increment();
			return new PrefixMatch(depth, below.owners.iterator().next());
		} finally {
			lock.readLock().unlock();
		}
	}

	/** The longest cached prefix of {@code tokens}: cached by {@code ownerKey}, or by any key when it is null. */
	private PrefixMatch find(int[] tokens, String ownerKey) {
		lookups.increment();
		if (tokens == null || tokens.length == 0)
			return PrefixMatch.empty();

		lock.readLock().lock();
		try {
			TrieNode current = root;
			int depth = 0;
			int matchLen = 0;
			String matchKey = null;

			for (int token : tokens) {
				TrieNode next = current.children.get(token);
				if (next == null)
					break;
				current = next;
				depth++;
				if (current.owners == null)
					continue;
				if (ownerKey == null) {
					matchLen = depth;
					matchKey = current.owners.iterator().next();
				} else if (current.owners.contains(ownerKey)) {
					matchLen = depth;
					matchKey = ownerKey;
				}
			}

			PrefixMatch match = matchKey != null ? new PrefixMatch(matchLen, matchKey) : PrefixMatch.empty();
			if (match.isHit())
				hits.increment();
			return match;
		} finally {
			lock.readLock().unlock();
		}
	}

	/** Total {@link #findLongestPrefix} calls, including empty inputs. */
	public long lookupCount() {
		return lookups.sum();
	}

	/** Lookups that returned {@link PrefixMatch#isHit()}. */
	public long hitCount() {
		return hits.sum();
	}

	/** {@code hits / lookups}, or 0 when no lookups. */
	public double hitRate() {
		long n = lookups.sum();
		return n == 0 ? 0.0 : (double) hits.sum() / n;
	}

	/**
	 * Cache a token prefix with a reference to its KV blocks, replacing the entry {@code cacheKey} held before.
	 *
	 * @param tokens    the full token sequence (prefix is extracted internally)
	 * @param prefixLen how many tokens to cache (typically full prompt length)
	 * @param cacheKey  reference key to look up KV blocks in KVCacheManager
	 */
	public void cachePrefix(int[] tokens, int prefixLen, String cacheKey) {
		if (tokens == null || prefixLen < 1 || cacheKey == null)
			return;
		int len = Math.min(prefixLen, tokens.length);

		lock.writeLock().lock();
		try {
			int[] earlier = entries.remove(cacheKey);
			if (earlier != null)
				removeEntry(earlier, cacheKey);
			TrieNode current = root;
			for (int i = 0; i < len; i++) {
				current = current.children.computeIfAbsent(tokens[i], k -> new TrieNode());
			}
			if (current.owners == null)
				current.owners = new LinkedHashSet<>(2);
			current.owners.add(cacheKey);
			entries.put(cacheKey, Arrays.copyOf(tokens, len));
		} finally {
			lock.writeLock().unlock();
		}
	}

	/**
	 * Invalidate the prefix {@code cacheKey} cached. Other keys' entries, on the same tokens included, stay.
	 */
	public void invalidate(String cacheKey) {
		lock.writeLock().lock();
		try {
			int[] path = entries.remove(cacheKey);
			if (path != null)
				removeEntry(path, cacheKey);
		} finally {
			lock.writeLock().unlock();
		}
	}

	/**
	 * Removes {@code cacheKey} from the node at the end of {@code path} and prunes the nodes of that path left with
	 * no key and no child, deepest first. Walks only the path: iterative, since the trie is one node deep per cached
	 * token and a session of a few thousand tokens overflowed a virtual thread's stack when this recursed.
	 */
	private void removeEntry(int[] path, String cacheKey) {
		TrieNode[] nodes = new TrieNode[path.length + 1];
		nodes[0] = root;
		for (int i = 0; i < path.length; i++) {
			nodes[i + 1] = nodes[i].children.get(path[i]);
			if (nodes[i + 1] == null)
				throw new IllegalStateException("prefix entry of " + cacheKey + " is not in the trie");
		}
		TrieNode end = nodes[path.length];
		if (end.owners != null && end.owners.remove(cacheKey) && end.owners.isEmpty())
			end.owners = null;
		for (int i = path.length; i >= 1 && nodes[i].owners == null && nodes[i].children.isEmpty(); i--)
			nodes[i - 1].children.remove(path[i - 1]);
	}

	// ── Inner types ───────────────────────────────────────────────────────────

	private static final class TrieNode {
		final Map<Integer, TrieNode> children = new HashMap<>();
		/** Keys that cached the prefix ending here, in caching order; null when none. */
		Set<String> owners;
	}

	/**
	 * Result of a prefix lookup.
	 */
	public record PrefixMatch(int matchedTokens, String cacheKey) {

		public static PrefixMatch empty() {
			return new PrefixMatch(0, null);
		}

		public boolean isHit() {
			return matchedTokens > 0 && cacheKey != null;
		}
	}
}
