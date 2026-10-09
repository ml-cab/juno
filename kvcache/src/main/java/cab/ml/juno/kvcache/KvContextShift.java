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

import java.util.stream.IntStream;

/**
 * Context shift over a request's host KV: positions {@code [keep, keep + discard)}
 * are dropped and the later ones move down by {@code discard}. V rows move
 * unchanged. K rows were rotated for their old positions, so each moved K row is
 * passed through the caller's {@link KeyRotation}, which turns it back by
 * {@code discard} positions; rotating the cached keys is exact for rotary
 * embeddings and avoids recomputing the kept tokens.
 *
 * <p>Works on any {@link SessionKvTensor} (dense or paged, f16 or q8_0). A q8_0
 * key row is re-encoded after rotation, which adds one rounding step per shift.
 */
public final class KvContextShift {

	/** Rotates one K row ({@code kvDim} floats) in place. Must be safe to call from several threads. */
	@FunctionalInterface
	public interface KeyRotation {
		void rotate(float[] row);
	}

	private KvContextShift() {
	}

	/**
	 * Shift every layer of one request. Layers are independent and run in
	 * parallel on the common pool.
	 *
	 * @param seqLen positions written before the shift; {@code seqLen - discard} after
	 */
	public static void shift(SessionKvTensor[] kLayers, SessionKvTensor[] vLayers, int seqLen, int keep, int discard,
			KeyRotation rotation) {
		checkRange(keep, discard, seqLen);
		if (kLayers.length != vLayers.length)
			throw new IllegalArgumentException("K has " + kLayers.length + " layers, V " + vLayers.length);
		int newLen = seqLen - discard;
		IntStream.range(0, kLayers.length).parallel().forEach(li -> {
			SessionKvTensor k = kLayers[li];
			k.compact(keep, discard, seqLen);
			vLayers[li].compact(keep, discard, seqLen);
			float[] row = new float[k.kvDim()];
			for (int p = keep; p < newLen; p++) {
				k.readToken(p, row);
				rotation.rotate(row);
				k.writeToken(p, row);
			}
		});
	}

	static void checkRange(int keep, int discard, int seqLen) {
		if (keep < 0 || discard < 1 || keep + discard > seqLen)
			throw new IllegalArgumentException(
					"context shift needs keep >= 0, discard >= 1 and keep + discard <= seqLen; got keep=" + keep
							+ " discard=" + discard + " seqLen=" + seqLen);
	}
}
