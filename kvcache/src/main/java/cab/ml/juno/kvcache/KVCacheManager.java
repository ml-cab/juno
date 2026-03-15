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

import java.util.Optional;
import java.util.logging.Logger;

/**
 * Unified KV cache facade — orchestrates GPU and CPU tiers.
 *
 * Write policy: write-through to both tiers. Read policy: GPU first, fall back
 * to CPU, promote on CPU hit. Eviction: per-tier (Caffeine on CPU, LRU slab on
 * GPU). Cleanup: evict(requestId) removes from both tiers on request
 * completion.
 *
 * Distributed KV: Each node creates its own KVCacheManager with a LayerRange
 * matching its ShardAssignment. put() enforces the range — writing KV for a
 * layer this node does not own is a routing bug and throws
 * IllegalArgumentException.
 *
 * The coordinator creates a KVCacheManager with LayerRange.all() (the default)
 * for backward compatibility and prefix cache operations.
 *
 * Thread-safe — each tier implementation is independently thread-safe.
 */
public final class KVCacheManager {

	private static final Logger log = Logger.getLogger(KVCacheManager.class.getName());

	private final GpuKVCache gpuCache;
	private final CpuKVCache cpuCache;
	private final PrefixCache prefixCache;
	private final LayerRange layerRange;

	/**
	 * Backward-compatible constructor — no layer restriction (LayerRange.all()).
	 * Used by coordinator and single-node setups.
	 */
	public KVCacheManager(GpuKVCache gpuCache, CpuKVCache cpuCache) {
		this(gpuCache, cpuCache, LayerRange.all());
	}

	/**
	 * Layer-range-aware constructor — for distributed nodes. put() will reject any
	 * block whose layerIndex is outside this range.
	 *
	 * @param layerRange the layer range this node is responsible for
	 */
	public KVCacheManager(GpuKVCache gpuCache, CpuKVCache cpuCache, LayerRange layerRange) {
		if (gpuCache == null)
			throw new IllegalArgumentException("gpuCache must not be null");
		if (cpuCache == null)
			throw new IllegalArgumentException("cpuCache must not be null");
		if (layerRange == null)
			throw new IllegalArgumentException("layerRange must not be null");
		this.gpuCache = gpuCache;
		this.cpuCache = cpuCache;
		this.prefixCache = new PrefixCache();
		this.layerRange = layerRange;
	}

	/**
	 * Store a KV block. Written to GPU tier first, then CPU tier.
	 *
	 * @throws IllegalArgumentException if key.layerIndex() is outside this
	 *                                  manager's LayerRange. Wrong-range puts are
	 *                                  routing bugs — fail fast.
	 */
	public void put(KVKey key, KVBlock block) {
		if (!layerRange.contains(key.layerIndex())) {
			throw new IllegalArgumentException(String.format(
					"KVCacheManager owns %s but got put for layer %d (key=%s). " + "This is a shard routing bug.",
					layerRange, key.layerIndex(), key));
		}
		gpuCache.put(key, block);
		cpuCache.put(key, block);
	}

	/**
	 * Retrieve a KV block. Checks GPU first, falls back to CPU. On CPU hit,
	 * promotes block back to GPU tier.
	 */
	public Optional<KVBlock> get(KVKey key) {
		// GPU hit — fast path
		Optional<KVBlock> gpuHit = gpuCache.get(key);
		if (gpuHit.isPresent())
			return gpuHit;

		// CPU hit — promote back to GPU
		Optional<KVBlock> cpuHit = cpuCache.get(key);
		if (cpuHit.isPresent()) {
			log.fine("KV cache CPU→GPU promotion: " + key);
			gpuCache.put(key, cpuHit.get());
			return cpuHit;
		}

		return Optional.empty();
	}

	/**
	 * Evict all blocks for a completed request from both tiers.
	 */
	public void evict(String requestId) {
		gpuCache.evict(requestId);
		cpuCache.evict(requestId);
	}

	/**
	 * Store a prefix in the prefix trie.
	 *
	 * @param tokens    full token sequence
	 * @param prefixLen how many tokens to register as cached prefix
	 * @param cacheKey  reference key for KV lookup
	 */
	public void cachePrefix(int[] tokens, int prefixLen, String cacheKey) {
		prefixCache.cachePrefix(tokens, prefixLen, cacheKey);
	}

	/**
	 * Find the longest matching cached prefix for the given token sequence.
	 */
	public PrefixCache.PrefixMatch findLongestPrefix(int[] tokens) {
		return prefixCache.findLongestPrefix(tokens);
	}

	/**
	 * Remove the prefix-trie entry associated with {@code cacheKey}.
	 *
	 * Call this together with {@link #evict(String)} when a conversation session
	 * ends, so a future session that begins with the same tokens does not obtain
	 * a stale prefix-cache hit pointing at evicted KV blocks.
	 *
	 * @param cacheKey the key used in {@link #cachePrefix} — typically the
	 *                 sessionId for multi-turn requests.
	 */
	public void invalidatePrefix(String cacheKey) {
		prefixCache.invalidate(cacheKey);
	}

	// ── Stats + metadata ──────────────────────────────────────────────────────

	public long gpuBlockCount() {
		return gpuCache.size();
	}

	public long cpuBlockCount() {
		return cpuCache.size();
	}

	public long gpuBytesUsed() {
		return gpuCache.estimatedSizeBytes();
	}

	public long cpuBytesUsed() {
		return cpuCache.estimatedSizeBytes();
	}

	public long gpuVramBudgetBytes() {
		return gpuCache.vramBudgetBytes();
	}

	/** The layer range this manager is responsible for. */
	public LayerRange layerRange() {
		return layerRange;
	}

	/** Whether this manager owns the given layer index. */
	public boolean ownsLayer(int layerIndex) {
		return layerRange.contains(layerIndex);
	}
}