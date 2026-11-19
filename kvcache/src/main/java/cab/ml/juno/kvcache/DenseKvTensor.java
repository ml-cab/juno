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

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;

/**
 * Dense per-layer KV tensor for one request: float32 rows ({@link KvElementType#F32}) or encoded rows
 * ({@link KvElementType#F16} halves, {@link KvElementType#Q8_0} / {@link KvElementType#Q4_0} blocks; {@link KvRowCodec}).
 *
 * <p>Attention reads the backing array in place as one page ({@link #attentionRows}):
 * {@code F32} rows as stored floats, encoded rows decoded a head slice at a time inside the attention
 * loop. Nothing is decoded into a scratch buffer.
 *
 * <p>Capacity doubles as positions are written, from {@link #INITIAL_SEQ_CAPACITY}. A tensor told how many positions
 * its request can reach ({@link #limitGrowth}) doubles only up to that count and then grows to it, so a request never
 * holds more positions than it can write; past the count it doubles again.
 *
 * <p>A tensor can serve several requests one after another ({@link SessionKvLayout#retireKLayers}).
 * Positions below {@link #writtenTokens} hold the current request's values; everything at
 * or above it reads as zero, as on a new tensor: a write or read past it first zeroes the
 * gap, so a request never sees an earlier request's values.
 */
public final class DenseKvTensor implements SessionKvTensor {

	public static final int INITIAL_SEQ_CAPACITY = 64;
	/**
	 * Hard cap on positions per request. Raised for continuous/paged long-context
	 * measurement (8k / 32k cells). Dense still grows on demand from
	 * {@link #INITIAL_SEQ_CAPACITY}; this is only the ceiling.
	 */
	public static final int MAX_SEQ_LEN = 32768;

	private final KvElementType type;
	private final int kvDim;
	private float[] f32;
	/** Encoded rows of an {@code F16}, {@code Q8_0} or {@code Q4_0} tensor, {@link #bytesPerToken()} bytes each. */
	private byte[] packed;
	/** The backing array as attention reads it; repointed when the array grows. */
	private final KvRows rows;
	private int capacityTokens;
	/** Positions {@code [0, writtenTokens)} belong to the current request; see the class comment. */
	private int writtenTokens;
	/** The most positions the current request can write ({@link #limitGrowth}); {@link #MAX_SEQ_LEN} when not told. */
	private int growthLimit = MAX_SEQ_LEN;

	public DenseKvTensor(KvElementType type, int kvDim) {
		this(type, kvDim, INITIAL_SEQ_CAPACITY);
	}

	public DenseKvTensor(KvElementType type, int kvDim, int initialTokens) {
		if (type == null)
			throw new IllegalArgumentException("type must not be null");
		if (kvDim < 1)
			throw new IllegalArgumentException("kvDim must be >= 1");
		if (initialTokens < 1)
			throw new IllegalArgumentException("initialTokens must be >= 1");
		this.type = type;
		this.kvDim = kvDim;
		this.capacityTokens = initialTokens;
		this.rows = new KvRows(type, kvDim, KvRows.ONE_PAGE);
		allocate(initialTokens);
	}

	/** Allocate one tensor per layer. */
	public static DenseKvTensor[] layers(int layerCount, KvElementType type, int kvDim) {
		return layers(layerCount, type, kvDim, INITIAL_SEQ_CAPACITY);
	}

	/** Allocate one tensor per layer, each holding {@code initialTokens} positions to start with. */
	public static DenseKvTensor[] layers(int layerCount, KvElementType type, int kvDim, int initialTokens) {
		if (layerCount < 1)
			throw new IllegalArgumentException("layerCount must be >= 1");
		DenseKvTensor[] out = new DenseKvTensor[layerCount];
		for (int i = 0; i < layerCount; i++)
			out[i] = new DenseKvTensor(type, kvDim, initialTokens);
		return out;
	}

	public KvElementType type() {
		return type;
	}

	public int kvDim() {
		return kvDim;
	}

	public int capacityTokens() {
		return capacityTokens;
	}

	/** Grow so position {@code pos} (0-based) fits. */
	public void ensureCapacity(int pos) {
		if (pos < 0)
			throw new IllegalArgumentException("pos must be >= 0");
		if (pos >= MAX_SEQ_LEN)
			throw new IllegalStateException("KV cache position " + pos + " exceeds MAX_SEQ_LEN=" + MAX_SEQ_LEN);
		int need = pos + 1;
		if (need <= capacityTokens)
			return;
		int newCap = capacityTokens;
		while (newCap < need)
			newCap = Math.min(newCap * 2, MAX_SEQ_LEN);
		if (need <= growthLimit)
			newCap = Math.min(newCap, growthLimit);
		if (newCap < need)
			throw new IllegalStateException("KV cache position " + pos + " exceeds MAX_SEQ_LEN=" + MAX_SEQ_LEN);
		grow(newCap);
	}

	public void writeToken(int pos, float[] src) {
		writeToken(pos, src, 0);
	}

	public void writeToken(int pos, float[] src, int srcOff) {
		ensureCapacity(pos);
		if (srcOff < 0 || srcOff + kvDim > src.length)
			throw new IllegalArgumentException("src too small for kvDim=" + kvDim);
		exposeThrough(pos);
		writtenTokens = Math.max(writtenTokens, pos + 1);
		switch (type) {
		case F32 -> System.arraycopy(src, srcOff, f32, pos * kvDim, kvDim);
		case F16, Q8_0, Q4_0 -> KvRowCodec.encode(type, src, srcOff, kvDim, packed, pos * bytesPerToken());
		}
	}

	/** F32 is one bulk copy into the backing array; the encoded types encode a row at a time. */
	@Override
	public void writeRows(int startPos, int rows, MemorySegment src, long srcOffset) {
		if (rows < 1)
			return;
		if (!type.storesFloats()) {
			SessionKvTensor.super.writeRows(startPos, rows, src, srcOffset);
			return;
		}
		ensureCapacity(startPos + rows - 1);
		exposeThrough(startPos);
		MemorySegment.copy(src, ValueLayout.JAVA_FLOAT, srcOffset, f32, startPos * kvDim, rows * kvDim);
		writtenTokens = Math.max(writtenTokens, startPos + rows);
	}

	/** One bulk copy into the packed rows. */
	@Override
	public void writeEncodedRows(int startPos, int rows, MemorySegment src, long srcOffset) {
		if (type.storesFloats())
			throw new IllegalArgumentException(type.cliName() + " rows are stored as floats, not encoded");
		if (rows < 1)
			return;
		ensureCapacity(startPos + rows - 1);
		exposeThrough(startPos);
		int bpt = bytesPerToken();
		MemorySegment.copy(src, ValueLayout.JAVA_BYTE, srcOffset, packed, startPos * bpt, rows * bpt);
		writtenTokens = Math.max(writtenTokens, startPos + rows);
	}

	/**
	 * Copy restored float data for positions {@code [0, seqLen)}. Used by
	 * {@code NodeKVCacheAdapter} restore into an empty tensor.
	 */
	public void loadFloatPrefix(float[] src, int seqLen) {
		if (seqLen < 1)
			throw new IllegalArgumentException("seqLen must be >= 1");
		ensureCapacity(seqLen - 1);
		int n = seqLen * kvDim;
		if (src.length < n)
			throw new IllegalArgumentException("src shorter than seqLen*kvDim");
		switch (type) {
		case F32 -> System.arraycopy(src, 0, f32, 0, n);
		case F16, Q8_0, Q4_0 -> {
			for (int p = 0; p < seqLen; p++)
				KvRowCodec.encode(type, src, p * kvDim, kvDim, packed, p * bytesPerToken());
		}
		}
		writtenTokens = Math.max(writtenTokens, seqLen);
	}

	/**
	 * The backing array as one page, read in place. Positions {@code [writtenTokens, seqLen)} are zeroed first (see
	 * the class comment), so attention never reads an earlier request's rows.
	 */
	@Override
	public KvRows attentionRows(int seqLen) {
		if (seqLen < 1)
			throw new IllegalArgumentException("seqLen must be >= 1");
		ensureCapacity(seqLen - 1);
		exposeThrough(seqLen);
		return rows;
	}

	/** Stored bytes for one token (q8_0 and q4_0 pad {@code kvDim} to 32-value blocks). */
	public int bytesPerToken() {
		return KvRowCodec.rowBytes(type, kvDim);
	}

	/** Copy positions {@code [0, seqLen)} into a new float array (flush / tests). */
	public float[] toFloatArray(int seqLen) {
		if (seqLen < 1)
			throw new IllegalArgumentException("seqLen must be >= 1");
		ensureCapacity(seqLen - 1);
		exposeThrough(seqLen);
		float[] out = new float[seqLen * kvDim];
		switch (type) {
		case F32 -> System.arraycopy(f32, 0, out, 0, out.length);
		case F16, Q8_0, Q4_0 -> {
			int bpt = bytesPerToken();
			for (int p = 0; p < seqLen; p++)
				KvRowCodec.decode(type, packed, p * bpt, out, p * kvDim, kvDim);
		}
		}
		return out;
	}

	/**
	 * The most positions this tensor's current request can write: capacity grows toward it instead of doubling past
	 * it. Clamped to {@link #MAX_SEQ_LEN}; never shrinks the capacity already held. A reused tensor starts without one.
	 */
	@Override
	public void limitGrowth(int positions) {
		if (positions < 1)
			throw new IllegalArgumentException("positions must be >= 1 (got " + positions + ")");
		growthLimit = Math.min(positions, MAX_SEQ_LEN);
	}

	/** Heap bytes of the backing array: {@link #allocatedBytes}. */
	@Override
	public long heapBytes() {
		return allocatedBytes();
	}

	public int allocatedBytes() {
		return switch (type) {
		case F32 -> f32.length * Float.BYTES;
		case F16, Q8_0, Q4_0 -> packed.length;
		};
	}

	@Override
	public void readToken(int pos, float[] dst) {
		if (pos < 0 || pos >= capacityTokens)
			throw new IllegalArgumentException("pos " + pos + " outside [0, " + capacityTokens + ")");
		exposeThrough(pos + 1);
		switch (type) {
		case F32 -> System.arraycopy(f32, pos * kvDim, dst, 0, kvDim);
		case F16, Q8_0, Q4_0 -> KvRowCodec.decode(type, packed, pos * bytesPerToken(), dst, 0, kvDim);
		}
	}

	@Override
	public void compact(int keep, int discard, int seqLen) {
		KvContextShift.checkRange(keep, discard, seqLen);
		if (seqLen > capacityTokens)
			throw new IllegalArgumentException("seqLen " + seqLen + " > capacity " + capacityTokens);
		int moved = seqLen - keep - discard;
		switch (type) {
		case F32 -> System.arraycopy(f32, (keep + discard) * kvDim, f32, keep * kvDim, moved * kvDim);
		case F16, Q8_0, Q4_0 -> {
			int bpt = bytesPerToken();
			System.arraycopy(packed, (keep + discard) * bpt, packed, keep * bpt, moved * bpt);
		}
		}
	}

	/**
	 * Makes positions {@code [0, len)} a copy of {@code src}'s, stored bytes copied rather than re-encoded, so
	 * attention over them reads exactly what it reads in {@code src}; nothing past {@code len} is readable until
	 * written. Lets a request start from another request's prefix without computing it.
	 *
	 * @throws IllegalArgumentException unless {@code src} has this tensor's type and width and has written at
	 *                                  least {@code len >= 1} positions
	 */
	public void copyPrefixFrom(DenseKvTensor src, int len) {
		if (src.type != type || src.kvDim != kvDim)
			throw new IllegalArgumentException("source is " + src.type + " x " + src.kvDim + ", this is " + type
					+ " x " + kvDim);
		if (len < 1 || len > src.writtenTokens)
			throw new IllegalArgumentException("len " + len + " outside [1, " + src.writtenTokens + "]");
		ensureCapacity(len - 1);
		switch (type) {
		case F32 -> System.arraycopy(src.f32, 0, f32, 0, len * kvDim);
		case F16, Q8_0, Q4_0 -> System.arraycopy(src.packed, 0, packed, 0, len * bytesPerToken());
		}
		writtenTokens = len;
	}

	/** Positions the current request has written or exposed: {@code [0, writtenTokens())}. */
	int writtenTokens() {
		return writtenTokens;
	}

	@Override
	public void release() {
		// dense storage is GC'd with the tensor
	}

	/** Hands the tensor to a new request: its earlier values become unreadable (see the class comment). */
	void reuse() {
		writtenTokens = 0;
		growthLimit = MAX_SEQ_LEN;
	}

	/**
	 * Zeroes positions {@code [writtenTokens, tokens)} and makes them part of the current
	 * request, so nothing an earlier request wrote there can be read. A no-op in the
	 * usual order, where every position is written before it is read.
	 */
	private void exposeThrough(int tokens) {
		if (tokens <= writtenTokens)
			return;
		switch (type) {
		case F32 -> Arrays.fill(f32, writtenTokens * kvDim, tokens * kvDim, 0f);
		case F16, Q8_0, Q4_0 -> Arrays.fill(packed, writtenTokens * bytesPerToken(), tokens * bytesPerToken(), (byte) 0);
		}
		writtenTokens = tokens;
	}

	private void allocate(int tokens) {
		capacityTokens = tokens;
		switch (type) {
		case F32 -> rows.setPage(0, f32 = new float[tokens * kvDim]);
		case F16, Q8_0, Q4_0 -> rows.setPage(0, packed = new byte[tokens * bytesPerToken()]);
		}
	}

	private void grow(int newTokens) {
		switch (type) {
		case F32 -> rows.setPage(0, f32 = Arrays.copyOf(f32, newTokens * kvDim));
		case F16, Q8_0, Q4_0 -> rows.setPage(0, packed = Arrays.copyOf(packed, newTokens * bytesPerToken()));
		}
		capacityTokens = newTokens;
	}
}
