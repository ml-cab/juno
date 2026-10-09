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

import java.util.HashMap;
import java.util.Map;
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
 */
public final class PrefixCache {

	private final TrieNode root = new TrieNode();
	private final ReadWriteLock lock = new ReentrantReadWriteLock();
	private final LongAdder lookups = new LongAdder();
	private final LongAdder hits = new LongAdder();

	/**
	 * Find the longest cached prefix of the given token sequence.
	 *
	 * @param tokens input token IDs
	 * @return PrefixMatch with the matched length (0 if no match) and cache key
	 */
	public PrefixMatch findLongestPrefix(int[] tokens) {
		lookups.increment();
		if (tokens == null || tokens.length == 0)
			return PrefixMatch.empty();

		lock.readLock().lock();
		try {
			TrieNode current = root;
			int matchLen = 0;
			String lastCacheKey = null;

			for (int token : tokens) {
				TrieNode next = current.children.get(token);
				if (next == null)
					break;
				current = next;
				matchLen++;
				if (current.cacheKey != null)
					lastCacheKey = current.cacheKey;
			}

			PrefixMatch match = matchLen > 0 && lastCacheKey != null
					? new PrefixMatch(matchLen, lastCacheKey)
					: PrefixMatch.empty();
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
	 * Cache a token prefix with a reference to its KV blocks.
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
			TrieNode current = root;
			for (int i = 0; i < len; i++) {
				current = current.children.computeIfAbsent(tokens[i], k -> new TrieNode());
			}
			current.cacheKey = cacheKey;
		} finally {
			lock.writeLock().unlock();
		}
	}

	/**
	 * Invalidate a cached prefix by its cache key.
	 */
	public void invalidate(String cacheKey) {
		lock.writeLock().lock();
		try {
			invalidateNode(root, cacheKey);
		} finally {
			lock.writeLock().unlock();
		}
	}

	/**
	 * Clears {@code cacheKey} everywhere below {@code start} and prunes the branches
	 * left empty. Iterative, children before parents: the trie is one node deep per
	 * cached token, and a recursive walk over a session of a few thousand tokens
	 * overflowed a virtual thread's stack.
	 */
	private void invalidateNode(TrieNode start, String cacheKey) {
		java.util.ArrayDeque<TrieNode> pending = new java.util.ArrayDeque<>();
		java.util.ArrayList<TrieNode> order = new java.util.ArrayList<>();
		pending.push(start);
		while (!pending.isEmpty()) {
			TrieNode node = pending.pop();
			order.add(node);
			if (cacheKey.equals(node.cacheKey))
				node.cacheKey = null;
			for (TrieNode child : node.children.values())
				pending.push(child);
		}
		// Every child was visited after its parent, so walking the visit order backwards
		// settles each node's children before the node itself.
		for (int i = order.size() - 1; i >= 0; i--)
			order.get(i).children.values().removeIf(c -> c.cacheKey == null && c.children.isEmpty());
	}

	// ── Inner types ───────────────────────────────────────────────────────────

	private static final class TrieNode {
		final Map<Integer, TrieNode> children = new HashMap<>();
		String cacheKey = null; // set only at leaf / checkpoint nodes
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
