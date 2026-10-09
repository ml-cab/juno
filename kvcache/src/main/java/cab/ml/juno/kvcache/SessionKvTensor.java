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
 * Handler-facing per-layer KV tensor for one request.
 *
 * <p>Implemented by {@link DenseKvTensor} (static schedule) and
 * {@link PagedKvTensor} (continuous schedule + gather).
 */
public interface SessionKvTensor {

	KvElementType type();

	int kvDim();

	/** Ensure position {@code pos} (0-based) is writable. */
	void ensureCapacity(int pos);

	void writeToken(int pos, float[] src);

	void writeToken(int pos, float[] src, int srcOff);

	/**
	 * Float view of positions {@code [0, seqLen)} for attention.
	 *
	 * @param scratch required when {@link #needsAttentionScratch()} is true
	 */
	float[] viewForAttention(int seqLen, float[] scratch);

	void loadFloatPrefix(float[] src, int seqLen);

	float[] toFloatArray(int seqLen);

	/**
	 * Dense: allocated token capacity. Paged: {@code pageCount * pageSize}
	 * (block granularity).
	 */
	int capacityTokens();

	/**
	 * True when {@link #viewForAttention} must copy into {@code scratch}
	 * (q8_0 dequant or paged gather).
	 */
	boolean needsAttentionScratch();

	/** Return pages to the pool (paged); no-op for dense. */
	void release();

	/** Copy the float row stored at {@code pos} into {@code dst[0, kvDim)}. */
	void readToken(int pos, float[] dst);

	/**
	 * Context shift: drop positions {@code [keep, keep + discard)} of the
	 * {@code seqLen} written ones and move {@code [keep + discard, seqLen)} down to
	 * start at {@code keep}, unchanged (stored bytes are copied, not re-encoded).
	 * Afterwards {@code seqLen - discard} positions are written. The caller
	 * re-rotates moved keys; see {@link KvContextShift}.
	 *
	 * @throws IllegalArgumentException unless {@code keep >= 0}, {@code discard >= 1}
	 *                                  and {@code keep + discard <= seqLen}
	 */
	void compact(int keep, int discard, int seqLen);
}
