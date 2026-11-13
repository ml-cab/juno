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

import java.util.Collection;

/**
 * Paged KV pages across a set of arenas, K and V pools together.
 *
 * @param livePages      pages a request, a session or a kept prompt holds
 * @param idlePages      free pages that still hold storage, reused before new ones are made
 * @param reclaimedPages idle pages whose storage was released, since the arenas were made
 * @param storedBytes    heap bytes the live and idle pages hold now
 */
public record KvPageCounts(long livePages, long idlePages, long reclaimedPages, long storedBytes) {

	/** No paged KV. */
	public static final KvPageCounts NONE = new KvPageCounts(0, 0, 0, 0);

	/** Summed over {@code arenas}, each counted as given (the caller passes each arena once). */
	public static KvPageCounts sum(Collection<PagedKvArena> arenas) {
		long live = 0;
		long idle = 0;
		long reclaimed = 0;
		long bytes = 0;
		for (PagedKvArena a : arenas)
			for (KvBlockPool p : new KvBlockPool[] { a.kPool(), a.vPool() }) {
				live += p.liveBlocks();
				idle += p.idlePages();
				reclaimed += p.reclaimedPages();
				bytes += p.storedBytes();
			}
		return arenas.isEmpty() ? NONE : new KvPageCounts(live, idle, reclaimed, bytes);
	}
}
