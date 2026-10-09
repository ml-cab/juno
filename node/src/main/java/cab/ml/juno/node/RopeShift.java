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

import cab.ml.juno.kvcache.KvContextShift;

/**
 * Turns cached keys back by a context shift's distance. Every rotary variant
 * Juno runs rotates pair {@code i} of a head by {@code pos * freq[i]}, scaled by a
 * factor that does not depend on the position (Phi-3's and YaRN's attention
 * factor). Moving a key from {@code p} to {@code p - d} is therefore a pure
 * rotation by {@code -d * freq[i]}, with no scale: the scale the key already
 * carries is the one its new position would give it.
 *
 * <p>Frequencies are computed in double precision from the same parameters the
 * forward rotation uses; each handler builds its instance from its own config
 * ({@link #standard}, {@link #partialSplitHalf}, {@code Phi3Rope.shift},
 * {@code Qwen3Rope.shift}).
 */
final class RopeShift {

	private final int headDim;
	private final boolean splitHalf;
	/** Angle per position for pair {@code i}; {@code freq.length} pairs per head are rotated. */
	private final double[] freq;

	private RopeShift(int headDim, RopePairing pairing, double[] freq) {
		if (2 * freq.length > headDim)
			throw new IllegalArgumentException(freq.length + " pairs do not fit a head of " + headDim);
		this.headDim = headDim;
		this.splitHalf = pairing == RopePairing.SPLIT_HALF;
		this.freq = freq;
	}

	/** Pairs {@code (2i, 2i+1)} or {@code (i, i + rotated/2)} with the given per-pair frequencies. */
	static RopeShift of(int headDim, RopePairing pairing, double[] freq) {
		return new RopeShift(headDim, pairing, freq.clone());
	}

	/** Standard rotation over the whole head: {@code freq[i] = theta^(-2i/headDim)}. */
	static RopeShift standard(int headDim, float theta, RopePairing pairing) {
		return new RopeShift(headDim, pairing, baseFrequencies(headDim / 2, headDim, theta));
	}

	/** Split-half rotation of the first {@code ropeDim} dims of each head (Phi-2); the rest stay put. */
	static RopeShift partialSplitHalf(int headDim, int ropeDim, float theta) {
		return new RopeShift(headDim, RopePairing.SPLIT_HALF, baseFrequencies(ropeDim / 2, ropeDim, theta));
	}

	/** {@code theta^(-2i/dims)} for {@code i < pairs}. */
	static double[] baseFrequencies(int pairs, int dims, double theta) {
		double[] f = new double[pairs];
		for (int i = 0; i < pairs; i++)
			f[i] = 1.0 / Math.pow(theta, (2.0 * i) / dims);
		return f;
	}

	/**
	 * Rotation taking a K row of {@code nKvHeads} heads from position {@code p} to
	 * {@code p - delta}. Its tables are built once here; the returned rotation only
	 * reads them, so the layers of one shift may share it across threads.
	 */
	KvContextShift.KeyRotation back(int delta, int nKvHeads) {
		int pairs = freq.length;
		float[] cos = new float[pairs];
		float[] sin = new float[pairs];
		for (int i = 0; i < pairs; i++) {
			double a = -(double) delta * freq[i];
			cos[i] = (float) Math.cos(a);
			sin[i] = (float) Math.sin(a);
		}
		int half = pairs;
		return row -> {
			for (int h = 0; h < nKvHeads; h++) {
				int base = h * headDim;
				for (int i = 0; i < pairs; i++) {
					int i0 = splitHalf ? base + i : base + 2 * i;
					int i1 = splitHalf ? base + i + half : base + 2 * i + 1;
					float x0 = row[i0];
					float x1 = row[i1];
					row[i0] = x0 * cos[i] - x1 * sin[i];
					row[i1] = x0 * sin[i] + x1 * cos[i];
				}
			}
		};
	}
}
