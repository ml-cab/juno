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
 * Continuous-schedule KV tensor: fixed-size pages from a shared
 * {@link KvBlockPool} via a {@link KvPageTable}.
 *
 * <p>Attention reads the pages in place, walking the block table
 * ({@link #attentionRows}); nothing is gathered into a contiguous workspace.
 * API mirrors {@link DenseKvTensor} write / view / restore helpers so handlers
 * run one path over either.
 */
public final class PagedKvTensor implements SessionKvTensor {

	private final KvPageTable table;

	public PagedKvTensor(KvBlockPool pool) {
		this(new KvPageTable(pool));
	}

	public PagedKvTensor(KvPageTable table) {
		if (table == null)
			throw new IllegalArgumentException("table must not be null");
		this.table = table;
	}

	public static PagedKvTensor[] layers(int layerCount, KvBlockPool pool) {
		if (layerCount < 1)
			throw new IllegalArgumentException("layerCount must be >= 1");
		PagedKvTensor[] out = new PagedKvTensor[layerCount];
		for (int i = 0; i < layerCount; i++)
			out[i] = new PagedKvTensor(pool);
		return out;
	}

	public KvElementType type() {
		return table.type();
	}

	public int kvDim() {
		return table.kvDim();
	}

	public int seqLen() {
		return table.seqLen();
	}

	public int pageCount() {
		return table.pageCount();
	}

	public int pageSize() {
		return table.pageSize();
	}

	public KvPageTable pageTable() {
		return table;
	}

	@Override
	public void ensureCapacity(int pos) {
		if (pos < 0)
			throw new IllegalArgumentException("pos must be >= 0");
		if (pos >= DenseKvTensor.MAX_SEQ_LEN)
			throw new IllegalStateException(
					"KV cache position " + pos + " exceeds MAX_SEQ_LEN=" + DenseKvTensor.MAX_SEQ_LEN);
		// Pages are allocated on writeToken; this only enforces the seq cap.
	}

	@Override
	public void writeToken(int pos, float[] src) {
		writeToken(pos, src, 0);
	}

	@Override
	public void writeToken(int pos, float[] src, int srcOff) {
		table.writeToken(pos, src, srcOff);
	}

	/** The block table's pages, read in place ({@link KvPageTable#rows}). */
	@Override
	public KvRows attentionRows(int seqLen) {
		if (seqLen < 1)
			throw new IllegalArgumentException("seqLen must be >= 1");
		if (seqLen > table.seqLen())
			throw new IllegalArgumentException("seqLen " + seqLen + " > stored " + table.seqLen());
		return table.rows();
	}

	@Override
	public void loadFloatPrefix(float[] src, int seqLen) {
		if (seqLen < 1)
			throw new IllegalArgumentException("seqLen must be >= 1");
		if (table.seqLen() != 0)
			throw new IllegalStateException("loadFloatPrefix requires an empty tensor");
		int n = seqLen * table.kvDim();
		if (src.length < n)
			throw new IllegalArgumentException("src shorter than seqLen*kvDim");
		for (int p = 0; p < seqLen; p++)
			table.appendToken(src, p * table.kvDim());
	}

	/** Gathers positions {@code [0, seqLen)} from the pages into a new array. */
	@Override
	public float[] toFloatArray(int seqLen) {
		if (seqLen < 1)
			throw new IllegalArgumentException("seqLen must be >= 1");
		if (seqLen > table.seqLen())
			throw new IllegalArgumentException("seqLen " + seqLen + " > stored " + table.seqLen());
		float[] out = new float[seqLen * table.kvDim()];
		table.gather(out, seqLen);
		return out;
	}

	@Override
	public int capacityTokens() {
		return table.pageCount() * table.pageSize();
	}

	public long allocatedBytes() {
		return (long) table.pageCount() * table.pageSize() * table.pool().bytesPerToken();
	}

	@Override
	public void readToken(int pos, float[] dst) {
		table.readToken(pos, dst);
	}

	/**
	 * Rows written past {@code seqLen} (a rejected draft window) are dropped first,
	 * so the table ends at {@code seqLen - discard} like the dense tensor.
	 */
	@Override
	public void compact(int keep, int discard, int seqLen) {
		KvContextShift.checkRange(keep, discard, seqLen);
		if (seqLen > table.seqLen())
			throw new IllegalArgumentException("seqLen " + seqLen + " > stored " + table.seqLen());
		table.truncate(seqLen);
		table.compact(keep, discard);
	}

	/**
	 * A new tensor whose positions {@code [0, len)} are this one's, holding the same pages (copy-on-write, see
	 * {@link KvPageTable#sharePrefix}). Must not overlap a write to this tensor.
	 */
	public PagedKvTensor sharePrefix(int len) {
		return new PagedKvTensor(table.sharePrefix(len));
	}

	@Override
	public void release() {
		table.release();
	}
}
