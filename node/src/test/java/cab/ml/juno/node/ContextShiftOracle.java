package cab.ml.juno.node;

import cab.ml.juno.kvcache.SessionKvTensor;

/**
 * Independent expectation for a context shift, for handler tests. Takes a
 * request's host KV before the shift and builds what it must hold after: rows
 * {@code [keep, keep + discard)} gone, later rows moved down, and each moved key
 * taken back to its unrotated form at its old position and rotated afresh at its
 * new one with the handler's own forward rotation (not with {@link RopeShift}).
 * Writing that KV into a second request gives the oracle the shifted request's
 * next logits are compared with.
 *
 * <p>A fresh request prefilled with only the kept tokens is not an oracle: in
 * every layer after the first, the kept tokens' keys and values were computed
 * while they attended to the discarded tokens, and a shift keeps them.
 */
final class ContextShiftOracle {

	@FunctionalInterface
	interface RowRotation {
		void apply(float[] row, int pos);
	}

	private ContextShiftOracle() {
	}

	/** Positions {@code [0, len)} of every layer, as float rows. */
	static float[][] snapshot(SessionKvTensor[] layers, int len) {
		float[][] out = new float[layers.length][];
		for (int li = 0; li < layers.length; li++)
			out[li] = layers[li].toFloatArray(len).clone();
		return out;
	}

	/** Expected KV after the shift ({@code seqLen - discard} rows per layer). */
	static float[][] expected(float[][] before, int kvDim, int seqLen, int keep, int discard, boolean keys,
			RowRotation unrotate, RowRotation rotate) {
		int newLen = seqLen - discard;
		float[][] out = new float[before.length][newLen * kvDim];
		float[] row = new float[kvDim];
		for (int li = 0; li < before.length; li++) {
			for (int p = 0; p < newLen; p++) {
				int src = p < keep ? p : p + discard;
				System.arraycopy(before[li], src * kvDim, row, 0, kvDim);
				if (keys && src != p) {
					unrotate.apply(row, src);
					rotate.apply(row, p);
				}
				System.arraycopy(row, 0, out[li], p * kvDim, kvDim);
			}
		}
		return out;
	}

	/** Overwrites positions {@code [0, len)} of a request that already holds at least that many. */
	static void inject(SessionKvTensor[] layers, float[][] rows, int kvDim, int len) {
		float[] row = new float[kvDim];
		for (int li = 0; li < layers.length; li++)
			for (int p = 0; p < len; p++) {
				System.arraycopy(rows[li], p * kvDim, row, 0, kvDim);
				layers[li].writeToken(p, row);
			}
	}
}
