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
import java.util.concurrent.atomic.AtomicLong;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;

/**
 * CPU-tier KV cache backed by Caffeine (W-TinyLFU eviction).
 *
 * Bounded by maximum number of blocks (not bytes directly). Use maxBlocks =
 * estimatedHeapBudget / averageBlockSize to size appropriately.
 *
 * Caffeine handles eviction automatically — GC-aware, no off-heap complexity.
 * Thread-safe — Caffeine's internal striped locking handles concurrency.
 */
public final class CpuKVCache implements KVCache {

	private final Cache<KVKey, KVBlock> cache;
	private final AtomicLong totalBytesStored = new AtomicLong(0);

	public CpuKVCache(long maxBlocks) {
		if (maxBlocks < 1)
			throw new IllegalArgumentException("maxBlocks must be >= 1");
		this.cache = Caffeine.newBuilder().maximumSize(maxBlocks).removalListener((_, block, _) -> {
			if (block instanceof KVBlock b)
				totalBytesStored.addAndGet(-b.sizeBytes());
		}).build();
	}

	@Override
	public void put(KVKey key, KVBlock block) {
		KVBlock existing = cache.getIfPresent(key);
		if (existing != null)
			totalBytesStored.addAndGet(-existing.sizeBytes());
		cache.put(key, block);
		totalBytesStored.addAndGet(block.sizeBytes());
	}

	@Override
	public Optional<KVBlock> get(KVKey key) {
		return Optional.ofNullable(cache.getIfPresent(key));
	}

	@Override
	public void evict(String requestId) {
		// Caffeine doesn't support prefix scans — collect keys then invalidate
		cache.asMap().keySet().stream().filter(k -> k.requestId().equals(requestId)).forEach(cache::invalidate);
	}

	@Override
	public boolean contains(KVKey key) {
		return cache.getIfPresent(key) != null;
	}

	@Override
	public long size() {
		return cache.estimatedSize();
	}

	@Override
	public long estimatedSizeBytes() {
		return totalBytesStored.get();
	}

	@Override
	public String tierName() {
		return "cpu";
	}
}
