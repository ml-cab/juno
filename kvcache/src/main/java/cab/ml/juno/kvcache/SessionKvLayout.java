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

/**
 * Dual KV path selector for transformer handlers.
 *
 * <p>{@code --schedule static} (default): dense {@link DenseKvTensor} layers;
 * {@code --kv-page-size} is unused. {@code --schedule continuous}: shared
 * {@link PagedKvArena}, read page by page in place.
 */
public final class SessionKvLayout {

	private final ServeScheduleOptions schedule;
	private final KvPageSizeOptions pageSize;
	private final CacheTypeOptions cacheTypes;
	private final int kvDim;
	private volatile PagedKvArena arena;
	/** Released dense layers kept for the next request (static schedule); unused when paged. */
	private final DenseKvRecycler retiredK = new DenseKvRecycler();
	private final DenseKvRecycler retiredV = new DenseKvRecycler();
	/** {@link #reserve}d positions of requests whose layers are not made yet, taken by {@link #newVLayers(String, int)}. */
	private final Map<String, Integer> reserved = new ConcurrentHashMap<>();

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
		KvKeyPrecisionNotice.announce(types);
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

	/**
	 * A request's K layers: paged from the arena, or dense, the layers a released request
	 * left ({@link #retireKLayers}) when there are some, else new ones.
	 */
	public SessionKvTensor[] newKLayers(int layerCount) {
		if (usesPaged())
			return arenaOrThrow().newKLayers(layerCount);
		SessionKvTensor[] reused = retiredK.take(layerCount, cacheTypes.typeK());
		return reused != null ? reused : DenseKvTensor.layers(layerCount, cacheTypes.typeK(), kvDim);
	}

	/**
	 * {@link #newKLayers(int)} for {@code requestId}: layers that grow toward the positions {@link #reserve} gave it,
	 * if it gave any. Make the K layers first, then the V layers, which take the reservation.
	 */
	public SessionKvTensor[] newKLayers(String requestId, int layerCount) {
		Integer positions = requestId == null ? null : reserved.get(requestId);
		return limited(layers(layerCount, retiredK, cacheTypes.typeK(), positions), positions);
	}

	/** As {@link #newKLayers(String, int)}, for V; the reservation is used up. */
	public SessionKvTensor[] newVLayers(String requestId, int layerCount) {
		Integer positions = requestId == null ? null : reserved.remove(requestId);
		return limited(layers(layerCount, retiredV, cacheTypes.typeV(), positions), positions);
	}

	/** Paged layers, a released request's dense layers, or new ones, starting no larger than {@code positions}. */
	private SessionKvTensor[] layers(int layerCount, DenseKvRecycler recycler, KvElementType type, Integer positions) {
		if (usesPaged())
			return recycler == retiredK ? arenaOrThrow().newKLayers(layerCount) : arenaOrThrow().newVLayers(layerCount);
		SessionKvTensor[] reused = recycler.take(layerCount, type);
		if (reused != null)
			return reused;
		int initial = positions == null ? DenseKvTensor.INITIAL_SEQ_CAPACITY
				: Math.min(DenseKvTensor.INITIAL_SEQ_CAPACITY, positions);
		return DenseKvTensor.layers(layerCount, type, kvDim, initial);
	}

	/**
	 * The most positions {@code requestId} can write ({@code positions}: its prompt plus the tokens it may generate),
	 * so that its dense layers grow toward that count instead of doubling past it ({@link SessionKvTensor#limitGrowth}).
	 * Layers it already has ({@code k}, {@code v}; either may be null) take it now; layers made for it later, through
	 * {@link #newKLayers(String, int)} and {@link #newVLayers(String, int)}, take it then. Paged layers ignore it.
	 */
	public void reserve(String requestId, int positions, SessionKvTensor[] k, SessionKvTensor[] v) {
		if (requestId == null)
			throw new IllegalArgumentException("requestId must not be null");
		if (positions < 1)
			throw new IllegalArgumentException("positions must be >= 1 (got " + positions + ")");
		limited(k, positions);
		limited(v, positions);
		if (k == null || v == null)
			reserved.put(requestId, positions);
		else
			reserved.remove(requestId);
	}

	/** Drops a reservation no layers took (the request ended first). */
	public void forget(String requestId) {
		if (requestId != null)
			reserved.remove(requestId);
	}

	private static SessionKvTensor[] limited(SessionKvTensor[] layers, Integer positions) {
		if (layers != null && positions != null)
			for (SessionKvTensor t : layers)
				t.limitGrowth(positions);
		return layers;
	}

	/** As {@link #newKLayers}, for V. */
	public SessionKvTensor[] newVLayers(int layerCount) {
		if (usesPaged())
			return arenaOrThrow().newVLayers(layerCount);
		SessionKvTensor[] reused = retiredV.take(layerCount, cacheTypes.typeV());
		return reused != null ? reused : DenseKvTensor.layers(layerCount, cacheTypes.typeV(), kvDim);
	}

	/**
	 * Ends a request's K layers from {@link #newKLayers}: paged layers return their pages,
	 * dense layers are kept for the next request (one set at most, the larger). The caller
	 * holds no reference to them afterwards.
	 */
	public void retireKLayers(SessionKvTensor[] layers) {
		retire(layers, retiredK);
	}

	/** As {@link #retireKLayers}, for V. */
	public void retireVLayers(SessionKvTensor[] layers) {
		retire(layers, retiredV);
	}

	private void retire(SessionKvTensor[] layers, DenseKvRecycler recycler) {
		if (layers == null)
			return;
		if (usesPaged())
			releaseLayers(layers);
		else
			recycler.offer(layers);
	}

	/** Release all tensors in a request's layer array (paged returns pages). */
	/**
	 * K layers for a new request whose positions {@code [0, len)} hold {@code src}'s rows, so it continues from
	 * them without computing them: on the paged layout the same pages, shared copy-on-write; on the dense layout a
	 * copy of the stored rows. Returns null, holding nothing, when a source layer has fewer than {@code len} rows.
	 * The source must not be written meanwhile.
	 */
	public SessionKvTensor[] sharePrefixK(SessionKvTensor[] src, int len) {
		return sharePrefix(src, len, retiredK, cacheTypes.typeK());
	}

	/** V layers, as {@link #sharePrefixK}. */
	public SessionKvTensor[] sharePrefixV(SessionKvTensor[] src, int len) {
		return sharePrefix(src, len, retiredV, cacheTypes.typeV());
	}

	private SessionKvTensor[] sharePrefix(SessionKvTensor[] src, int len, DenseKvRecycler recycler,
			KvElementType type) {
		if (src == null || src.length == 0 || len < 1)
			return null;
		for (SessionKvTensor t : src)
			if (rowsHeld(t) < len)
				return null;
		if (usesPaged()) {
			SessionKvTensor[] out = new SessionKvTensor[src.length];
			for (int i = 0; i < src.length; i++)
				out[i] = ((PagedKvTensor) src[i]).sharePrefix(len);
			return out;
		}
		SessionKvTensor[] out = recycler.take(src.length, type);
		if (out == null)
			out = DenseKvTensor.layers(src.length, type, kvDim);
		for (int i = 0; i < src.length; i++)
			((DenseKvTensor) out[i]).copyPrefixFrom((DenseKvTensor) src[i], len);
		return out;
	}

	private static int rowsHeld(SessionKvTensor t) {
		if (t instanceof PagedKvTensor p)
			return p.seqLen();
		if (t instanceof DenseKvTensor d)
			return d.writtenTokens();
		return 0;
	}

	/** Heap bytes of a request's layers ({@link SessionKvTensor#heapBytes}); 0 for null. */
	public static long heapBytes(SessionKvTensor[] layers) {
		long sum = 0;
		if (layers != null)
			for (SessionKvTensor t : layers)
				sum += t.heapBytes();
		return sum;
	}

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
