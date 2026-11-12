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

import java.util.Arrays;

import cab.ml.juno.kvcache.KvElementType;
import cab.ml.juno.kvcache.KvRows;
import cab.ml.juno.kvcache.Q8_0KvCodec;

/**
 * CPU attention for a prefill window of one request: query rows {@code q[0, rows)} at positions
 * {@code startPos + b}, row {@code b} attending the keys up to its own position ({@link GqaMath#firstKey} under a
 * window). Every head of every row gets the bits {@link GqaMath#attendPooled} gives that row on its own, in the head
 * form chosen the same way.
 *
 * <p>The worker pool is entered once per window with (row, KV head) pairs as its units, instead of once per row with
 * query heads as units. A unit runs the {@code gqaRatio} query heads that share one KV head. Over
 * {@link KvElementType#Q8_0} rows it decodes each key slice and each value slice once for all of those heads
 * ({@link Q8_0KvCodec#decodeRange}); a unit of one head would decode it once per head. Each head then scores and sums
 * exactly as its own unit would: the decoded values are the same, and so are the dot product and the key-order sums.
 * Over {@link KvElementType#F16} rows a unit runs its heads one after another through the per-head forms.
 */
final class GqaWindowAttention {

	/** Per-thread scores: one array per query head of a unit, each grown to the longest context seen and reused. */
	private static final ThreadLocal<float[][]> SCORES = ThreadLocal.withInitial(() -> new float[0][]);

	private GqaWindowAttention() {
	}

	/**
	 * Attention for rows {@code [0, rows)} of the window: writes every head's slice of {@code out[b]}. K and V hold at
	 * least {@code startPos + rows} rows, read in place.
	 */
	static void attend(float[][] q, KvRows k, KvRows v, int startPos, int rows, float[][] out, int numHeads,
			int headDim, int gqaRatio, int kvDim, int window) {
		attend(q, k, v, startPos, rows, out, numHeads, headDim, gqaRatio, kvDim, window, GqaMath.VECTOR_HEAD);
	}

	/**
	 * {@link #attend(float[][], KvRows, KvRows, int, int, float[][], int, int, int, int, int)} with the head form
	 * chosen by the caller, as {@link GqaMath#attendPooled(float[], KvRows, KvRows, int, float[], int, int, int, int, int, boolean)}
	 * takes it: {@code vectorHead} needs the Vector API module. For tests that compare the two.
	 */
	static void attend(float[][] q, KvRows k, KvRows v, int startPos, int rows, float[][] out, int numHeads,
			int headDim, int gqaRatio, int kvDim, int window, boolean vectorHead) {
		float scale = (float) (1.0 / Math.sqrt(headDim));
		int kvHeads = numHeads / gqaRatio;
		CpuThreads.forEach(rows * kvHeads, u -> {
			int b = u / kvHeads;
			int seqLen = startPos + b + 1;
			unit(q[b], k, v, seqLen, GqaMath.firstKey(seqLen, window), out[b], u - b * kvHeads, headDim, gqaRatio,
					kvDim, scale, vectorHead);
		});
	}

	/** Query heads {@code [g * gqaRatio, (g + 1) * gqaRatio)} of one row, over keys {@code [lo, seqLen)}. */
	private static void unit(float[] q, KvRows k, KvRows v, int seqLen, int lo, float[] out, int g, int headDim,
			int gqaRatio, int kvDim, float scale, boolean vectorHead) {
		int n = seqLen - lo;
		float[][] scores = scores(gqaRatio, n);
		int kBase = g * headDim;
		int h0 = g * gqaRatio;
		if (k.type() == KvElementType.F16) {
			for (int i = 0; i < gqaRatio; i++) {
				int qBase = (h0 + i) * headDim;
				if (vectorHead)
					GqaVectorHead.scoreKeys(q, qBase, k, kBase, lo, seqLen, headDim, kvDim, scale, scores[i]);
				else
					GqaMath.scoreKeys(q, qBase, k, kBase, lo, seqLen, headDim, kvDim, scale, scores[i]);
			}
		} else {
			scoreDecodedKeys(q, k, kBase, h0, lo, seqLen, headDim, gqaRatio, scale, scores, vectorHead);
		}
		for (int i = 0; i < gqaRatio; i++)
			LlamaTransformerHandler.softmax(scores[i], n);
		if (v.type() == KvElementType.F16) {
			for (int i = 0; i < gqaRatio; i++) {
				int outBase = (h0 + i) * headDim;
				if (vectorHead) {
					GqaVectorHead.addValues(v, kBase, lo, seqLen, out, outBase, headDim, kvDim, scores[i]);
				} else {
					Arrays.fill(out, outBase, outBase + headDim, 0f);
					GqaMath.addValues(v, kBase, lo, seqLen, out, outBase, headDim, kvDim, scores[i]);
				}
			}
		} else {
			addDecodedValues(v, kBase, h0, lo, seqLen, out, headDim, gqaRatio, scores);
		}
	}

	/** Each {@code Q8_0} key slice decoded once, then scored against every query head of the unit. */
	private static void scoreDecodedKeys(float[] q, KvRows k, int kBase, int h0, int lo, int seqLen, int headDim,
			int gqaRatio, float scale, float[][] scores, boolean vectorHead) {
		float[] row = GqaMath.row(headDim);
		int per = k.rowsPerPage();
		int bytesPerRow = k.bytesPerRow();
		for (int t = lo; t < seqLen;) {
			int page = t / per;
			int slot = t - page * per;
			int count = Math.min(seqLen - t, per - slot);
			byte[] rows = k.bytePage(page);
			for (int r = 0, ro = slot * bytesPerRow; r < count; r++, ro += bytesPerRow) {
				Q8_0KvCodec.decodeRange(rows, ro, kBase, headDim, row, 0);
				int s = t - lo + r;
				for (int i = 0; i < gqaRatio; i++) {
					if (vectorHead)
						GqaVectorHead.scoreRun(q, (h0 + i) * headDim, row, 0, 0, 1, headDim, scale, scores[i], s);
					else
						GqaMath.scoreRun(q, (h0 + i) * headDim, row, 0, 0, 1, headDim, scale, scores[i], s);
				}
			}
			t += count;
		}
	}

	/**
	 * Each {@code Q8_0} value slice decoded once, then added into every query head's output with that head's weight:
	 * each output element sums its products in key order from zero, the per-head loop's sums.
	 */
	private static void addDecodedValues(KvRows v, int kBase, int h0, int lo, int seqLen, float[] out, int headDim,
			int gqaRatio, float[][] scores) {
		Arrays.fill(out, h0 * headDim, (h0 + gqaRatio) * headDim, 0f);
		float[] row = GqaMath.row(headDim);
		int per = v.rowsPerPage();
		int bytesPerRow = v.bytesPerRow();
		for (int t = lo; t < seqLen;) {
			int page = t / per;
			int slot = t - page * per;
			int count = Math.min(seqLen - t, per - slot);
			byte[] rows = v.bytePage(page);
			for (int r = 0, ro = slot * bytesPerRow; r < count; r++, ro += bytesPerRow) {
				Q8_0KvCodec.decodeRange(rows, ro, kBase, headDim, row, 0);
				int s = t - lo + r;
				for (int i = 0; i < gqaRatio; i++) {
					float w = scores[i][s];
					int ob = (h0 + i) * headDim;
					for (int d = 0; d < headDim; d++)
						out[ob + d] += w * row[d];
				}
			}
			t += count;
		}
	}

	/** The calling thread's score arrays: {@code heads} of them, each at least {@code n} long. */
	private static float[][] scores(int heads, int n) {
		float[][] held = SCORES.get();
		if (held.length < heads) {
			held = Arrays.copyOf(held, heads);
			SCORES.set(held);
		}
		for (int i = 0; i < heads; i++)
			if (held[i] == null || held[i].length < n)
				held[i] = new float[n];
		return held;
	}
}
