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

import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * GPU-tier KV cache backed by Cuda CudaBuffer allocations.
 *
 * In production: each KVBlock's data array is a pointer into a pre-allocated
 * VRAM slab. Blocks are managed via a custom LRU within the slab.
 *
 * Current implementation: stores references in a ConcurrentHashMap index (the
 * actual CUDA buffer management is wired in by the node module which has the
 * Cuda dependency). This module stays free of the Cuda dep so it can compile
 * and test without a GPU present.
 *
 * The node module subclasses or wraps this with real CUDA buffer calls. See:
 * node/GpuKVCacheBackend.java
 *
 * Capacity is tracked by bytes — evicts LRU when vramBudgetBytes is exceeded.
 */
public final class GpuKVCache implements KVCache {

	private final long vramBudgetBytes;
	private final Map<KVKey, KVBlock> index = new ConcurrentHashMap<>();
	private final AtomicLong usedBytes = new AtomicLong(0);

	public GpuKVCache(long vramBudgetBytes) {
		if (vramBudgetBytes < 1)
			throw new IllegalArgumentException("vramBudgetBytes must be >= 1");
		this.vramBudgetBytes = vramBudgetBytes;
	}

	@Override
	public void put(KVKey key, KVBlock block) {
		// Evict oldest entries if over budget
		while (usedBytes.get() + block.sizeBytes() > vramBudgetBytes && !index.isEmpty()) {
			evictOldest();
		}
		KVBlock prev = index.put(key, block);
		if (prev != null)
			usedBytes.addAndGet(-prev.sizeBytes());
		usedBytes.addAndGet(block.sizeBytes());
	}

	@Override
	public Optional<KVBlock> get(KVKey key) {
		return Optional.ofNullable(index.get(key));
	}

	@Override
	public void evict(String requestId) {
		index.keySet().stream().filter(k -> k.requestId().equals(requestId)).forEach(k -> {
			KVBlock removed = index.remove(k);
			if (removed != null)
				usedBytes.addAndGet(-removed.sizeBytes());
		});
	}

	@Override
	public boolean contains(KVKey key) {
		return index.containsKey(key);
	}

	@Override
	public long size() {
		return index.size();
	}

	@Override
	public long estimatedSizeBytes() {
		return usedBytes.get();
	}

	@Override
	public String tierName() {
		return "gpu";
	}

	public long vramBudgetBytes() {
		return vramBudgetBytes;
	}

	// ── Helpers ───────────────────────────────────────────────────────────────

	private void evictOldest() {
		index.entrySet().stream().min((a, b) -> a.getValue().lastAccessedAt().compareTo(b.getValue().lastAccessedAt()))
				.ifPresent(entry -> {
					index.remove(entry.getKey());
					usedBytes.addAndGet(-entry.getValue().sizeBytes());
				});
	}
}
