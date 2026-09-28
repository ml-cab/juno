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

package cab.ml.juno.node;

import java.util.concurrent.atomic.AtomicReferenceArray;

/**
 * RoPE cosines and sines per position, computed once instead of once per head,
 * per layer, per call.
 *
 * <p>The angle of pair {@code i} at position {@code pos} depends only on
 * {@code pos}, {@code i}, the head size and the base, yet the scalar rotation
 * evaluated {@code Math.pow}, {@code Math.cos} and {@code Math.sin} for every
 * pair of every head of every layer: 792 times per angle per token on a
 * 22-layer, 36-head model. This table holds each value once. Every entry is
 * computed with exactly the expressions the rotation used,
 * {@code (float) Math.cos(pos * (1.0 / Math.pow(theta, (2.0 * i) / headDim)))}
 * and likewise for the sine, so a rotation that reads the table is bit-identical
 * to one that computes in place.
 *
 * <p>Positions are filled lazily in blocks of {@link #BLOCK} positions up to
 * {@link #MAX_POSITIONS}; a block is computed off to the side and published
 * whole, so readers take no lock and never see a partly filled block. Two
 * threads that miss the same block both compute it and one publication wins;
 * the values are identical, so either is right. A position at or beyond
 * {@link #MAX_POSITIONS} is computed on the spot instead. Tables are shared by
 * every caller with the same head size and base (at most about 16 MB for head
 * size 128 when every position has been touched).
 */
final class RopeTable {

	/** Positions per lazily computed block. */
	static final int BLOCK = 256;

	/** Positions the table covers; matches the KV cache's maximum sequence length. */
	static final int MAX_POSITIONS = 32768;

	/** Copy-on-write list of every table built so far; scanned without locking. */
	private static volatile RopeTable[] tables = new RopeTable[0];

	private final int headDim;
	private final float theta;
	private final int pairs;
	/** Per block: {@code [pos-in-block][pair] -> cos, sin} interleaved. */
	private final AtomicReferenceArray<float[]> blocks = new AtomicReferenceArray<>(MAX_POSITIONS / BLOCK);

	private RopeTable(int headDim, float theta) {
		this.headDim = headDim;
		this.theta = theta;
		this.pairs = headDim / 2;
	}

	/** The shared table for {@code (headDim, theta)}; allocation-free once it exists. */
	static RopeTable of(int headDim, float theta) {
		RopeTable[] snapshot = tables;
		for (RopeTable t : snapshot) {
			if (t.headDim == headDim && Float.floatToRawIntBits(t.theta) == Float.floatToRawIntBits(theta))
				return t;
		}
		return register(headDim, theta);
	}

	private static synchronized RopeTable register(int headDim, float theta) {
		for (RopeTable t : tables) {
			if (t.headDim == headDim && Float.floatToRawIntBits(t.theta) == Float.floatToRawIntBits(theta))
				return t;
		}
		RopeTable created = new RopeTable(headDim, theta);
		RopeTable[] grown = java.util.Arrays.copyOf(tables, tables.length + 1);
		grown[tables.length] = created;
		tables = grown;
		return created;
	}

	/**
	 * The block holding {@code pos}, laid out as {@code 2 * headDim/2} floats per
	 * position (cos, sin per pair); the row for {@code pos} starts at
	 * {@link #rowOffset(int)}. Null when {@code pos} is outside the table.
	 */
	float[] block(int pos) {
		if (pos < 0 || pos >= MAX_POSITIONS)
			return null;
		int b = pos / BLOCK;
		float[] blk = blocks.get(b);
		if (blk == null) {
			blk = compute(b);
			if (!blocks.compareAndSet(b, null, blk))
				blk = blocks.get(b);
		}
		return blk;
	}

	/** Offset of {@code pos}'s row within {@link #block(int)}. */
	int rowOffset(int pos) {
		return (pos % BLOCK) * 2 * pairs;
	}

	private float[] compute(int b) {
		float[] blk = new float[BLOCK * 2 * pairs];
		double[] freq = new double[pairs];
		for (int i = 0; i < pairs; i++)
			freq[i] = 1.0 / Math.pow(theta, (2.0 * i) / headDim);
		int base = b * BLOCK;
		for (int p = 0; p < BLOCK; p++) {
			int pos = base + p;
			int row = p * 2 * pairs;
			for (int i = 0; i < pairs; i++) {
				double angle = pos * freq[i];
				blk[row + 2 * i] = (float) Math.cos(angle);
				blk[row + 2 * i + 1] = (float) Math.sin(angle);
			}
		}
		return blk;
	}
}
