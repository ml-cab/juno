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
 * Dual KV path selector for transformer handlers.
 *
 * <p>{@code --schedule static} (default): dense {@link DenseKvTensor} layers;
 * {@code --kv-page-size} is unused. {@code --schedule continuous}: shared
 * {@link PagedKvArena} + gather-to-workspace.
 */
public final class SessionKvLayout {

	private final ServeScheduleOptions schedule;
	private final KvPageSizeOptions pageSize;
	private final CacheTypeOptions cacheTypes;
	private final int kvDim;
	private volatile PagedKvArena arena;

	private SessionKvLayout(ServeScheduleOptions schedule, KvPageSizeOptions pageSize,
			CacheTypeOptions cacheTypes, int kvDim, PagedKvArena arena) {
		if (schedule == null || pageSize == null || cacheTypes == null)
			throw new IllegalArgumentException("schedule/pageSize/cacheTypes must not be null");
		if (kvDim < 1)
			throw new IllegalArgumentException("kvDim must be >= 1");
		this.schedule = schedule;
		this.pageSize = pageSize;
		this.cacheTypes = cacheTypes;
		this.kvDim = kvDim;
		this.arena = arena;
	}

	/** Build from env / system properties for the given model {@code kvDim}. */
	public static SessionKvLayout fromEnv(int kvDim) {
		ServeScheduleOptions schedule = ServeScheduleOptions.fromEnv();
		KvPageSizeOptions page = KvPageSizeOptions.fromEnv();
		CacheTypeOptions types = CacheTypeOptions.fromEnv();
		PagedKvArena arena = schedule.usesPagedKv()
				? PagedKvArena.fromOptions(page, kvDim, types)
				: null;
		return new SessionKvLayout(schedule, page, types, kvDim, arena);
	}

	/**
	 * Explicit factory (tests). Continuous schedule requires a non-null arena
	 * unless {@link #bindSharedArena} is called before allocating layers.
	 */
	public static SessionKvLayout of(ServeScheduleOptions schedule, KvPageSizeOptions pageSize,
			CacheTypeOptions cacheTypes, int kvDim, PagedKvArena arena) {
		return new SessionKvLayout(schedule, pageSize, cacheTypes, kvDim, arena);
	}

	/** Prefer the manager's shared arena when the adapter is wired. */
	public void bindSharedArena(PagedKvArena shared) {
		if (shared == null)
			throw new IllegalArgumentException("shared arena must not be null");
		if (!schedule.usesPagedKv())
			return;
		if (shared.kvDim() != kvDim)
			throw new IllegalArgumentException(
					"shared arena kvDim=" + shared.kvDim() + " != layout kvDim=" + kvDim);
		this.arena = shared;
	}

	public ServeScheduleOptions schedule() {
		return schedule;
	}

	public KvPageSizeOptions pageSize() {
		return pageSize;
	}

	public CacheTypeOptions cacheTypes() {
		return cacheTypes;
	}

	public int kvDim() {
		return kvDim;
	}

	public boolean usesPaged() {
		return schedule.usesPagedKv();
	}

	public Optional<PagedKvArena> arena() {
		return Optional.ofNullable(arena);
	}

	/** Scratch required for q8_0 dequant and/or paged gather. */
	public boolean needsAttentionScratch() {
		return usesPaged() || cacheTypes.usesQuantized();
	}

	public SessionKvTensor[] newKLayers(int layerCount) {
		if (usesPaged())
			return arenaOrThrow().newKLayers(layerCount);
		return DenseKvTensor.layers(layerCount, cacheTypes.typeK(), kvDim);
	}

	public SessionKvTensor[] newVLayers(int layerCount) {
		if (usesPaged())
			return arenaOrThrow().newVLayers(layerCount);
		return DenseKvTensor.layers(layerCount, cacheTypes.typeV(), kvDim);
	}

	/** Release all tensors in a request's layer array (paged returns pages). */
	public static void releaseLayers(SessionKvTensor[] layers) {
		if (layers == null)
			return;
		for (SessionKvTensor t : layers)
			t.release();
	}

	public String policySummary() {
		StringBuilder sb = new StringBuilder(schedule.policySummary());
		sb.append(' ').append(cacheTypes.policySummary());
		if (usesPaged()) {
			sb.append(' ').append(pageSize.policySummary());
			if (arena != null)
				sb.append(' ').append(arena.policySummary());
		} else {
			sb.append(" kv-page-size ignored (dense)");
		}
		return sb.toString();
	}

	private PagedKvArena arenaOrThrow() {
		PagedKvArena a = arena;
		if (a == null)
			throw new IllegalStateException(
					"continuous schedule requires a PagedKvArena (bindSharedArena or fromEnv)");
		return a;
	}
}
