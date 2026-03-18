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

	/**
	 * Find the longest cached prefix of the given token sequence.
	 *
	 * @param tokens input token IDs
	 * @return PrefixMatch with the matched length (0 if no match) and cache key
	 */
	public PrefixMatch findLongestPrefix(int[] tokens) {
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

			return matchLen > 0 && lastCacheKey != null ? new PrefixMatch(matchLen, lastCacheKey) : PrefixMatch.empty();
		} finally {
			lock.readLock().unlock();
		}
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
				current = current.children.computeIfAbsent(tokens[i], _ -> new TrieNode());
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

	private boolean invalidateNode(TrieNode node, String cacheKey) {
		if (cacheKey.equals(node.cacheKey)) {
			node.cacheKey = null;
		}
		node.children.values().removeIf(child -> invalidateNode(child, cacheKey));
		return node.cacheKey == null && node.children.isEmpty();
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
