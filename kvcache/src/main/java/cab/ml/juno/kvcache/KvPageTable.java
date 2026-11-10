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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

/**
 * Per-request page table over a shared {@link KvBlockPool}.
 *
 * <p>Maps logical token positions to (blockId, slot). {@link #gather} copies
 * the active prefix into a contiguous float workspace for BLAS attention.
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
	private int seqLen;

	public KvPageTable(KvBlockPool pool) {
		if (pool == null)
			throw new IllegalArgumentException("pool must not be null");
		this.pool = pool;
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
		}
		shared.seqLen = len;
		return shared;
	}

	/** The block behind page {@code page}, made this table's alone first if another table holds it. */
	private int writablePage(int page) {
		int id = blockIds.get(page);
		int own = pool.exclusive(id);
		if (own != id)
			blockIds.set(page, own);
		return own;
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
		// F16 float path: copy whole pages via FloatBuffer (gather-tax mitigation).
		if (pool.type() == KvElementType.F16) {
			gatherF16Bulk(workspace, len);
			return;
		}
		int pageSize = pool.pageSize();
		int kvDim = pool.kvDim();
		for (int p = 0; p < len; p++) {
			int page = p / pageSize;
			int slot = p % pageSize;
			pool.readToken(blockIds.get(page), slot, workspace, p * kvDim);
		}
	}

	private void gatherF16Bulk(float[] workspace, int len) {
		int pageSize = pool.pageSize();
		int kvDim = pool.kvDim();
		int fullPages = len / pageSize;
		int rem = len % pageSize;
		int dst = 0;
		for (int pi = 0; pi < fullPages; pi++) {
			byte[] page = pool.pageBytes(blockIds.get(pi));
			FloatBuffer fb = ByteBuffer.wrap(page).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
			fb.get(workspace, dst, pageSize * kvDim);
			dst += pageSize * kvDim;
		}
		if (rem > 0) {
			byte[] page = pool.pageBytes(blockIds.get(fullPages));
			FloatBuffer fb = ByteBuffer.wrap(page).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
			fb.get(workspace, dst, rem * kvDim);
		}
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
		seqLen = len;
	}

	/** Drop this table's hold on every page (a page returns to the pool with its last holder) and reset. */
	public void release() {
		for (int id : blockIds)
			pool.free(id);
		blockIds.clear();
		seqLen = 0;
	}
}
