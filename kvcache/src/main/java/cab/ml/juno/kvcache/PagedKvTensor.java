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
 * <p>Attention always gathers into a contiguous float workspace (gather tax).
 * API mirrors {@link DenseKvTensor} write / view / restore helpers so handlers
 * can dual-path later.
 */
public final class PagedKvTensor {

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

	public void writeToken(int pos, float[] src) {
		writeToken(pos, src, 0);
	}

	public void writeToken(int pos, float[] src, int srcOff) {
		table.writeToken(pos, src, srcOff);
	}

	/**
	 * Gather positions {@code [0, seqLen)} into {@code scratch} for attention.
	 * Always copies (paged storage is not a contiguous float array).
	 */
	public float[] viewForAttention(int seqLen, float[] scratch) {
		if (seqLen < 1)
			throw new IllegalArgumentException("seqLen must be >= 1");
		if (seqLen > table.seqLen())
			throw new IllegalArgumentException("seqLen " + seqLen + " > stored " + table.seqLen());
		int need = seqLen * table.kvDim();
		if (scratch == null || scratch.length < need)
			throw new IllegalArgumentException("scratch must hold seqLen*kvDim floats");
		table.gather(scratch, seqLen);
		return scratch;
	}

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

	public float[] toFloatArray(int seqLen) {
		float[] out = new float[seqLen * table.kvDim()];
		return viewForAttention(seqLen, out);
	}

	public long allocatedBytes() {
		return (long) table.pageCount() * table.pageSize() * table.pool().bytesPerToken();
	}

	public void release() {
		table.release();
	}
}
