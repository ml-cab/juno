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

/**
 * Shared K/V {@link KvBlockPool}s for the continuous-schedule paged KV path.
 *
 * <p>Owned by {@link KVCacheManager} when schedule selects paged layout. Handlers
 * allocate per-request / per-layer {@link PagedKvTensor}s from these pools.
 */
public final class PagedKvArena {

	private final int pageSize;
	private final int kvDim;
	private final KvBlockPool kPool;
	private final KvBlockPool vPool;

	public PagedKvArena(int pageSize, int kvDim, KvElementType typeK, KvElementType typeV) {
		if (pageSize < 1)
			throw new IllegalArgumentException("pageSize must be >= 1");
		if (kvDim < 1)
			throw new IllegalArgumentException("kvDim must be >= 1");
		if (typeK == null || typeV == null)
			throw new IllegalArgumentException("element types must not be null");
		this.pageSize = pageSize;
		this.kvDim = kvDim;
		this.kPool = new KvBlockPool(pageSize, kvDim, typeK);
		this.vPool = new KvBlockPool(pageSize, kvDim, typeV);
	}

	/** Factory matching CLI page size + cache-type policy. */
	public static PagedKvArena fromOptions(KvPageSizeOptions page, int kvDim, CacheTypeOptions types) {
		if (page == null || types == null)
			throw new IllegalArgumentException("page and types must not be null");
		return new PagedKvArena(page.pageSize(), kvDim, types.typeK(), types.typeV());
	}

	public int pageSize() {
		return pageSize;
	}

	public int kvDim() {
		return kvDim;
	}

	public KvBlockPool kPool() {
		return kPool;
	}

	public KvBlockPool vPool() {
		return vPool;
	}

	public PagedKvTensor newK() {
		return new PagedKvTensor(kPool);
	}

	public PagedKvTensor newV() {
		return new PagedKvTensor(vPool);
	}

	public PagedKvTensor[] newKLayers(int layerCount) {
		return PagedKvTensor.layers(layerCount, kPool);
	}

	public PagedKvTensor[] newVLayers(int layerCount) {
		return PagedKvTensor.layers(layerCount, vPool);
	}

	public String policySummary() {
		return "paged-arena page=" + pageSize
				+ " k=" + kPool.type().cliName()
				+ " v=" + vPool.type().cliName();
	}
}
