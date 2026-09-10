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

import java.util.ArrayList;
import java.util.List;

/**
 * Per-request page table over a shared {@link KvBlockPool}.
 *
 * <p>Maps logical token positions to (blockId, slot). {@link #gather} copies
 * the active prefix into a contiguous float workspace for BLAS attention.
 */
public final class KvPageTable {

	private final KvBlockPool pool;
	private final List<Integer> blockIds = new ArrayList<>();
	private int seqLen;

	public KvPageTable(KvBlockPool pool) {
		if (pool == null)
			throw new IllegalArgumentException("pool must not be null");
		this.pool = pool;
	}

	public int seqLen() {
		return seqLen;
	}

	public int pageCount() {
		return blockIds.size();
	}

	public int pageSize() {
		return pool.pageSize();
	}

	/** Append one token vector; allocates a new page when the current page is full. */
	public void appendToken(float[] src) {
		appendToken(src, 0);
	}

	public void appendToken(float[] src, int srcOff) {
		int pageSize = pool.pageSize();
		int slot = seqLen % pageSize;
		if (slot == 0) {
			blockIds.add(pool.allocate());
		}
		int blockId = blockIds.get(blockIds.size() - 1);
		pool.writeToken(blockId, slot, src, srcOff);
		seqLen++;
	}

	/**
	 * Copy positions {@code [0, seqLen)} into {@code workspace} (length ≥
	 * {@code seqLen * kvDim}).
	 */
	public void gather(float[] workspace) {
		int need = seqLen * pool.kvDim();
		if (workspace == null || workspace.length < need)
			throw new IllegalArgumentException("workspace must hold seqLen*kvDim floats");
		int pageSize = pool.pageSize();
		int kvDim = pool.kvDim();
		for (int p = 0; p < seqLen; p++) {
			int page = p / pageSize;
			int slot = p % pageSize;
			pool.readToken(blockIds.get(page), slot, workspace, p * kvDim);
		}
	}

	/** Free all pages back to the pool and reset. */
	public void release() {
		for (int id : blockIds)
			pool.free(id);
		blockIds.clear();
		seqLen = 0;
	}
}
