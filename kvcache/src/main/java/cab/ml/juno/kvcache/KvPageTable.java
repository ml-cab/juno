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

	public int kvDim() {
		return pool.kvDim();
	}

	public KvElementType type() {
		return pool.type();
	}

	public KvBlockPool pool() {
		return pool;
	}

	/** Append one token vector; allocates a new page when the current page is full. */
	public void appendToken(float[] src) {
		appendToken(src, 0);
	}

	public void appendToken(float[] src, int srcOff) {
		writeToken(seqLen, src, srcOff);
	}

	/**
	 * Write at logical position {@code pos}. Append-only or in-place overwrite.
	 * Positions ahead of {@code seqLen} are zero-filled (matches dense
	 * {@link DenseKvTensor} capacity growth semantics).
	 */
	public void writeToken(int pos, float[] src) {
		writeToken(pos, src, 0);
	}

	public void writeToken(int pos, float[] src, int srcOff) {
		if (pos < 0)
			throw new IllegalArgumentException("pos must be >= 0");
		if (pos >= DenseKvTensor.MAX_SEQ_LEN)
			throw new IllegalStateException(
					"KV cache position " + pos + " exceeds MAX_SEQ_LEN=" + DenseKvTensor.MAX_SEQ_LEN);
		// Dense path leaves earlier slots as zeros when first write is mid-sequence;
		// match that so single-token forward at startPosition>0 stays compatible.
		if (pos > seqLen) {
			float[] zeros = new float[pool.kvDim()];
			while (seqLen < pos)
				appendToken(zeros);
		}
		int pageSize = pool.pageSize();
		if (pos == seqLen) {
			int slot = seqLen % pageSize;
			if (slot == 0)
				blockIds.add(pool.allocate());
			pool.writeToken(blockIds.get(blockIds.size() - 1), slot, src, srcOff);
			seqLen++;
			return;
		}
		int page = pos / pageSize;
		int slot = pos % pageSize;
		pool.writeToken(blockIds.get(page), slot, src, srcOff);
	}

	/**
	 * Copy positions {@code [0, seqLen)} into {@code workspace} (length ≥
	 * {@code seqLen * kvDim}).
	 */
	public void gather(float[] workspace) {
		gather(workspace, seqLen);
	}

	/** Gather a prefix of length {@code len} ({@code 0 < len <= seqLen}). */
	public void gather(float[] workspace, int len) {
		if (len < 0 || len > seqLen)
			throw new IllegalArgumentException("len out of range: " + len);
		int need = len * pool.kvDim();
		if (workspace == null || workspace.length < need)
			throw new IllegalArgumentException("workspace must hold len*kvDim floats");
		int pageSize = pool.pageSize();
		int kvDim = pool.kvDim();
		for (int p = 0; p < len; p++) {
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
