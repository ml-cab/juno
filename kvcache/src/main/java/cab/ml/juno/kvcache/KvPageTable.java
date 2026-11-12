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
 * <p>Maps logical token positions to (blockId, slot). Attention reads the pages
 * in place through {@link #rows()}, which this table keeps current as its pages
 * change; {@link #gather} copies the active prefix into a contiguous float array
 * (serialization, tests).
 *
 * <p>{@link #sharePrefix} starts a table from this one's first rows without
 * copying them: both hold the same pages, counted by the pool. A page held by
 * more than one table is never written; whichever table writes into it first
 * (an append into a partly filled shared page, an overwrite, a context shift)
 * takes a copy and drops its hold on the shared page, so neither table ever sees
 * the other's later rows. A table is used by one thread at a time, and sharing
 * reads the source table, so it must not overlap a write to that table; tables
 * that share pages may run on different threads.
 */
public final class KvPageTable {

	private final KvBlockPool pool;
	private final List<Integer> blockIds = new ArrayList<>();
	/** This table's pages, in table order, as attention reads them. */
	private final KvRows rows;
	private int seqLen;

	public KvPageTable(KvBlockPool pool) {
		if (pool == null)
			throw new IllegalArgumentException("pool must not be null");
		this.pool = pool;
		this.rows = new KvRows(pool.type(), pool.kvDim(), pool.pageSize());
	}

	/**
	 * A new table over the same pool whose positions {@code [0, len)} are this
	 * table's, holding the same pages rather than copies ({@code 0 <= len <= seqLen}).
	 * Rows either table writes afterwards are its own.
	 */
	public KvPageTable sharePrefix(int len) {
		if (len < 0 || len > seqLen)
			throw new IllegalArgumentException("len " + len + " outside [0, " + seqLen + "]");
		KvPageTable shared = new KvPageTable(pool);
		int pages = (len + pool.pageSize() - 1) / pool.pageSize();
		for (int i = 0; i < pages; i++) {
			int id = blockIds.get(i);
			pool.retain(id);
			shared.blockIds.add(id);
			shared.track(i, id);
		}
		shared.seqLen = len;
		return shared;
	}

	/** The block behind page {@code page}, made this table's alone first if another table holds it. */
	private int writablePage(int page) {
		int id = blockIds.get(page);
		int own = pool.exclusive(id);
		if (own != id) {
			blockIds.set(page, own);
			track(page, own);
		}
		return own;
	}

	/** Points the attention view's page {@code page} at block {@code blockId}'s storage. */
	private void track(int page, int blockId) {
		if (pool.type() == KvElementType.F16)
			rows.setPage(page, pool.floatPage(blockId));
		else
			rows.setPage(page, pool.bytePage(blockId));
	}

	/**
	 * Positions {@code [0, seqLen)} as attention reads them: this table's pages in the pool, in place. Valid until
	 * the table is next written, truncated or released.
	 */
	public KvRows rows() {
		return rows;
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
			int block;
			if (slot == 0) {
				block = pool.allocate();
				blockIds.add(block);
				track(blockIds.size() - 1, block);
			} else {
				block = writablePage(blockIds.size() - 1);
			}
			pool.writeToken(block, slot, src, srcOff);
			seqLen++;
			return;
		}
		pool.writeToken(writablePage(pos / pageSize), pos % pageSize, src, srcOff);
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
		if (len == 0)
			return;
		int pageSize = pool.pageSize();
		int kvDim = pool.kvDim();
		// Through the block ids, not the attention view, so that a copy checks what the view reads.
		if (pool.type() == KvElementType.F16) {
			for (int p = 0, page = 0; p < len; p += pageSize, page++)
				System.arraycopy(pool.floatPage(blockIds.get(page)), 0, workspace, p * kvDim,
						Math.min(pageSize, len - p) * kvDim);
			return;
		}
		for (int p = 0; p < len; p++)
			pool.readToken(blockIds.get(p / pageSize), p % pageSize, workspace, p * kvDim);
	}

	/** Copy the float row at {@code pos} (must be written) into {@code dst[0, kvDim)}. */
	public void readToken(int pos, float[] dst) {
		if (pos < 0 || pos >= seqLen)
			throw new IllegalArgumentException("pos " + pos + " outside [0, " + seqLen + ")");
		int pageSize = pool.pageSize();
		pool.readToken(blockIds.get(pos / pageSize), pos % pageSize, dst);
	}

	/**
	 * Context shift over the written positions: drop {@code [keep, keep + discard)},
	 * move the rest down byte for byte, and return the pages past the new length.
	 */
	public void compact(int keep, int discard) {
		KvContextShift.checkRange(keep, discard, seqLen);
		int pageSize = pool.pageSize();
		for (int dst = keep; dst + discard < seqLen; dst++) {
			int src = dst + discard;
			int dstBlock = writablePage(dst / pageSize);
			pool.copyToken(blockIds.get(src / pageSize), src % pageSize, dstBlock, dst % pageSize);
		}
		truncate(seqLen - discard);
	}

	/** Keep positions {@code [0, len)} and drop this table's hold on every page past them. */
	public void truncate(int len) {
		if (len < 0 || len > seqLen)
			throw new IllegalArgumentException("len " + len + " outside [0, " + seqLen + "]");
		int pages = (len + pool.pageSize() - 1) / pool.pageSize();
		while (blockIds.size() > pages)
			pool.free(blockIds.remove(blockIds.size() - 1));
		rows.clearFrom(pages);
		seqLen = len;
	}

	/** Drop this table's hold on every page (a page returns to the pool with its last holder) and reset. */
	public void release() {
		for (int id : blockIds)
			pool.free(id);
		blockIds.clear();
		rows.clearFrom(0);
		seqLen = 0;
	}
}
