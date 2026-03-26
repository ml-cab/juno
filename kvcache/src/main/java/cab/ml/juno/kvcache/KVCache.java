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

/**
 * Single-tier KV cache contract.
 *
 * Implementations: CpuKVCache — Caffeine, bounded by JVM heap (-Xmx) GpuKVCache
 * — Cuda CudaBuffer, bounded by GPU VRAM
 *
 * All methods are thread-safe.
 */
public interface KVCache {

	/**
	 * Store a KV block. Evicts entries if capacity is exceeded.
	 */
	void put(KVKey key, KVBlock block);

	/**
	 * Retrieve a KV block, or empty if not present.
	 */
	Optional<KVBlock> get(KVKey key);

	/**
	 * Evict all blocks for a given requestId (called when request completes).
	 */
	void evict(String requestId);

	/**
	 * Whether this cache contains a block for the given key.
	 */
	boolean contains(KVKey key);

	/**
	 * Current number of blocks stored.
	 */
	long size();

	/**
	 * Approximate memory used by cached data in bytes.
	 */
	long estimatedSizeBytes();

	/**
	 * Human-readable tier name for logging e.g. "cpu", "gpu".
	 */
	String tierName();
}
