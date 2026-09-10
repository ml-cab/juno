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

import java.util.Arrays;

/**
 * Dense per-layer KV tensor for one request: float32 ({@link KvElementType#F16})
 * or packed {@link KvElementType#Q8_0}.
 *
 * <p>Attention always consumes float views. For {@code F16}, {@link #viewForAttention}
 * returns the backing array (bit-compatible). For {@code Q8_0}, it dequants into
 * the provided scratch buffer.
 */
public final class DenseKvTensor {

	public static final int INITIAL_SEQ_CAPACITY = 64;
	/** Matches transformer handler decode window cap. */
	public static final int MAX_SEQ_LEN = 2048;

	private final KvElementType type;
	private final int kvDim;
	private float[] f32;
	private byte[] q8;
	private int capacityTokens;

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
		allocate(initialTokens);
	}

	/** Allocate one tensor per layer. */
	public static DenseKvTensor[] layers(int layerCount, KvElementType type, int kvDim) {
		if (layerCount < 1)
			throw new IllegalArgumentException("layerCount must be >= 1");
		DenseKvTensor[] out = new DenseKvTensor[layerCount];
		for (int i = 0; i < layerCount; i++)
			out[i] = new DenseKvTensor(type, kvDim);
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
		switch (type) {
		case F16 -> System.arraycopy(src, srcOff, f32, pos * kvDim, kvDim);
		case Q8_0 -> Q8_0KvCodec.encode(src, srcOff, kvDim, q8, pos * bytesPerToken());
		}
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
		case F16 -> System.arraycopy(src, 0, f32, 0, n);
		case Q8_0 -> {
			for (int p = 0; p < seqLen; p++)
				Q8_0KvCodec.encode(src, p * kvDim, kvDim, q8, p * bytesPerToken());
		}
		}
	}

	/**
	 * Float view of positions {@code [0, seqLen)} for attention.
	 *
	 * @param scratch required when type is {@link KvElementType#Q8_0}; ignored for F16
	 * @return backing F16 array or {@code scratch} filled with dequantized values
	 */
	public float[] viewForAttention(int seqLen, float[] scratch) {
		if (seqLen < 1)
			throw new IllegalArgumentException("seqLen must be >= 1");
		ensureCapacity(seqLen - 1);
		int n = seqLen * kvDim;
		return switch (type) {
		case F16 -> f32;
		case Q8_0 -> {
			if (scratch == null || scratch.length < n)
				throw new IllegalArgumentException("scratch must hold seqLen*kvDim floats");
			int bpt = bytesPerToken();
			for (int p = 0; p < seqLen; p++)
				Q8_0KvCodec.decode(q8, p * bpt, scratch, p * kvDim, kvDim);
			yield scratch;
		}
		};
	}

	/** Packed bytes for one token (q8_0 pads {@code kvDim} to QK blocks). */
	public int bytesPerToken() {
		return switch (type) {
		case F16 -> kvDim * Float.BYTES;
		case Q8_0 -> Q8_0KvCodec.encodedBytes(kvDim);
		};
	}

	/** Copy positions {@code [0, seqLen)} into a new float array (flush / tests). */
	public float[] toFloatArray(int seqLen) {
		float[] out = new float[seqLen * kvDim];
		float[] view = viewForAttention(seqLen, out);
		if (view != out)
			System.arraycopy(view, 0, out, 0, out.length);
		return out;
	}

	public int allocatedBytes() {
		return switch (type) {
		case F16 -> f32.length * Float.BYTES;
		case Q8_0 -> q8.length;
		};
	}

	private void allocate(int tokens) {
		capacityTokens = tokens;
		switch (type) {
		case F16 -> f32 = new float[tokens * kvDim];
		case Q8_0 -> q8 = new byte[tokens * Q8_0KvCodec.encodedBytes(kvDim)];
		}
	}

	private void grow(int newTokens) {
		switch (type) {
		case F16 -> f32 = Arrays.copyOf(f32, newTokens * kvDim);
		case Q8_0 -> q8 = Arrays.copyOf(q8, newTokens * Q8_0KvCodec.encodedBytes(kvDim));
		}
		capacityTokens = newTokens;
	}
}
