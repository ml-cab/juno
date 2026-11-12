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
 * Grouped-query attention math (QK^T + softmax + weighted-V-sum), extracted
 * from {@link LlamaTransformerHandler#gqaInto}/{@code gqa} so it can also serve
 * as the CPU oracle for the GPU-resident attention path ({@link CudaGqaAttention})
 * without duplicating the algorithm.
 *
 * <p>Causal masking is implicit via {@code seqLen} — the KV cache view passed
 * in only ever holds valid, already-written positions.
 *
 * <p>K and V are read in place ({@link KvRows}): a request's paged KV page by page through its block table, dense KV
 * as one page, never gathered into a contiguous copy. A page's run of rows goes through the same loop the whole
 * cache did when it was one array, so the sums, and the output bits, do not depend on the page size. A
 * {@link KvElementType#Q8_0} row is decoded one head's slice at a time into a per-thread row inside the loop
 * ({@link Q8_0KvCodec#decodeRange}), to the values a whole-row decode gives. The {@code float[]} forms (training's
 * own caches, the GPU kernels' oracle) read their arrays as one page.
 */
final class GqaMath {

	private GqaMath() {
	}

	/**
	 * Zero-allocation grouped-query attention. Writes output into the
	 * pre-allocated {@code out} buffer; reuses the shared {@code scores}
	 * scratch array (both must be at least {@code numHeads * headDim} and
	 * {@code seqLen} long, respectively).
	 *
	 * <p>With {@code window > 0} the row attends over its last {@code window} keys
	 * only, {@code [max(0, seqLen - window), seqLen)}, the GPU kernel's contract
	 * ({@link GqaAttentionKernel}); 0 means every key. Rows before the window are
	 * not read. A window no shorter than {@code seqLen} runs exactly the
	 * arithmetic of no window.
	 *
	 * @param k       at least {@code seqLen} rows, read in place
	 * @param v       at least {@code seqLen} rows, read in place
	 * @param window  keys a row attends over, counted back from its own; 0 for all
	 */
	static void attend(float[] q, KvRows k, KvRows v, int seqLen,
			float[] out, float[] scores, int numHeads, int headDim, int gqaRatio, int kvDim, int window) {
		float scale = (float) (1.0 / Math.sqrt(headDim));
		Arrays.fill(out, 0f);
		int lo = firstKey(seqLen, window);
		for (int h = 0; h < numHeads; h++)
			attendHead(q, k, v, seqLen, lo, out, h, headDim, gqaRatio, kvDim, scale, scores);
	}

	/**
	 * {@link #attend(float[], KvRows, KvRows, int, float[], float[], int, int, int, int, int)} over rows held in one
	 * array each, row-major with stride {@code kvDim}.
	 */
	static void attend(float[] q, float[] kCache, float[] vCache, int seqLen,
			float[] out, float[] scores, int numHeads, int headDim, int gqaRatio, int kvDim, int window) {
		attend(q, KvRows.of(kCache, kvDim), KvRows.of(vCache, kvDim), seqLen, out, scores, numHeads, headDim,
				gqaRatio, kvDim, window);
	}

	/** As {@link #attend(float[], float[], float[], int, float[], float[], int, int, int, int, int)} with no window. */
	static void attend(float[] q, float[] kCache, float[] vCache, int seqLen,
			float[] out, float[] scores, int numHeads, int headDim, int gqaRatio, int kvDim) {
		attend(q, kCache, vCache, seqLen, out, scores, numHeads, headDim, gqaRatio, kvDim, 0);
	}

	/** Per-thread score scratch for {@link #attendPooled}; grows to the longest context seen and is reused. */
	private static final ThreadLocal<float[][]> SCORES = ThreadLocal.withInitial(() -> new float[][] { new float[0] });

	/** Per-thread row a {@code Q8_0} head slice is decoded into; grows to the widest head seen and is reused. */
	private static final ThreadLocal<float[][]> ROW = ThreadLocal.withInitial(() -> new float[][] { new float[0] });

	/**
	 * True when {@link #attendPooled} runs each head through {@link GqaVectorHead}: the Vector API module is
	 * present. Without it every head takes the serial loop's body, which is bit-identical to {@link #attend}; the
	 * vector head differs from it by float rounding only, within its oracle bound.
	 */
	static final boolean VECTOR_HEAD = VectorQuantKernels.AVAILABLE;

	/**
	 * {@link #attend(float[], KvRows, KvRows, int, float[], float[], int, int, int, int, int)} with the query heads
	 * spread over the CPU worker pool ({@link CpuThreads}). Each head writes its own slice of {@code out} with a
	 * score scratch of its own thread, through {@link GqaVectorHead} when {@link #VECTOR_HEAD} holds, and the output
	 * is the same bits at every thread count. Called from inside a pool task it runs on the calling thread.
	 */
	static void attendPooled(float[] q, KvRows k, KvRows v, int seqLen, float[] out, int numHeads,
			int headDim, int gqaRatio, int kvDim, int window) {
		attendPooled(q, k, v, seqLen, out, numHeads, headDim, gqaRatio, kvDim, window, VECTOR_HEAD);
	}

	/** {@link #attendPooled(float[], KvRows, KvRows, int, float[], int, int, int, int, int)} over one array each. */
	static void attendPooled(float[] q, float[] kCache, float[] vCache, int seqLen, float[] out, int numHeads,
			int headDim, int gqaRatio, int kvDim, int window) {
		attendPooled(q, KvRows.of(kCache, kvDim), KvRows.of(vCache, kvDim), seqLen, out, numHeads, headDim, gqaRatio,
				kvDim, window, VECTOR_HEAD);
	}

	/**
	 * {@link #attendPooled(float[], KvRows, KvRows, int, float[], int, int, int, int, int)} with every head through
	 * the serial loop's body, so the output is {@link #attend}'s bit for bit at any thread count, with or without the
	 * Vector API module. A decode batch attends this way: a request decoded alone attends through {@link #attend}, and
	 * its output must not depend on whether it shared a decode step with other requests.
	 */
	static void attendPooledAsSerial(float[] q, KvRows k, KvRows v, int seqLen, float[] out, int numHeads,
			int headDim, int gqaRatio, int kvDim, int window) {
		attendPooled(q, k, v, seqLen, out, numHeads, headDim, gqaRatio, kvDim, window, false);
	}

	/** {@link #attendPooledAsSerial(float[], KvRows, KvRows, int, float[], int, int, int, int, int)} over one array each. */
	static void attendPooledAsSerial(float[] q, float[] kCache, float[] vCache, int seqLen, float[] out, int numHeads,
			int headDim, int gqaRatio, int kvDim, int window) {
		attendPooled(q, KvRows.of(kCache, kvDim), KvRows.of(vCache, kvDim), seqLen, out, numHeads, headDim, gqaRatio,
				kvDim, window, false);
	}

	/**
	 * {@link #attendPooled(float[], KvRows, KvRows, int, float[], int, int, int, int, int)} with the head form chosen
	 * by the caller: {@code vectorHead} needs the Vector API module. For tests that compare the two.
	 */
	static void attendPooled(float[] q, KvRows k, KvRows v, int seqLen, float[] out, int numHeads,
			int headDim, int gqaRatio, int kvDim, int window, boolean vectorHead) {
		float scale = (float) (1.0 / Math.sqrt(headDim));
		int lo = firstKey(seqLen, window);
		if (vectorHead)
			CpuThreads.forEach(numHeads, h -> GqaVectorHead.attend(q, k, v, seqLen, lo, out, h, headDim,
					gqaRatio, kvDim, scale, scores(seqLen)));
		else
			CpuThreads.forEach(numHeads, h -> attendHead(q, k, v, seqLen, lo, out, h, headDim, gqaRatio,
					kvDim, scale, scores(seqLen)));
	}

	/** {@link #attendPooled(float[], KvRows, KvRows, int, float[], int, int, int, int, int, boolean)} over one array each. */
	static void attendPooled(float[] q, float[] kCache, float[] vCache, int seqLen, float[] out, int numHeads,
			int headDim, int gqaRatio, int kvDim, int window, boolean vectorHead) {
		attendPooled(q, KvRows.of(kCache, kvDim), KvRows.of(vCache, kvDim), seqLen, out, numHeads, headDim, gqaRatio,
				kvDim, window, vectorHead);
	}

	/** The calling thread's score scratch, at least {@code seqLen} long. */
	private static float[] scores(int seqLen) {
		float[][] holder = SCORES.get();
		if (holder[0].length < seqLen)
			holder[0] = new float[seqLen];
		return holder[0];
	}

	/** The calling thread's {@code Q8_0} decode row, at least {@code width} long. */
	static float[] row(int width) {
		float[][] holder = ROW.get();
		if (holder[0].length < width)
			holder[0] = new float[width];
		return holder[0];
	}

	/**
	 * One head of the serial loop, writing only {@code out[h * headDim, (h + 1) * headDim)} from keys
	 * {@code [lo, seqLen)}, with {@code scores} (at least {@code seqLen - lo} long) as scratch.
	 */
	static void attendHead(float[] q, KvRows k, KvRows v, int seqLen, int lo, float[] out, int h,
			int headDim, int gqaRatio, int kvDim, float scale, float[] scores) {
		int kBase = (h / gqaRatio) * headDim;
		int qBase = h * headDim;
		scoreKeys(q, qBase, k, kBase, lo, seqLen, headDim, kvDim, scale, scores);
		LlamaTransformerHandler.softmax(scores, seqLen - lo);
		int outBase = h * headDim;
		Arrays.fill(out, outBase, outBase + headDim, 0f);
		addValues(v, kBase, lo, seqLen, out, outBase, headDim, kvDim, scores);
	}

	/** {@code scores[t - lo]} for keys {@code [lo, seqLen)}: the serial dot product of the head's query and key slice. */
	static void scoreKeys(float[] q, int qBase, KvRows k, int kBase, int lo, int seqLen, int headDim,
			int kvDim, float scale, float[] scores) {
		int per = k.rowsPerPage();
		if (k.type() == KvElementType.F16) {
			for (int t = lo; t < seqLen;) {
				int page = t / per;
				int slot = t - page * per;
				int count = Math.min(seqLen - t, per - slot);
				scoreRun(q, qBase, k.floatPage(page), slot * kvDim + kBase, kvDim, count, headDim, scale, scores, t - lo);
				t += count;
			}
			return;
		}
		float[] row = row(headDim);
		int bytesPerRow = k.bytesPerRow();
		for (int t = lo; t < seqLen;) {
			int page = t / per;
			int slot = t - page * per;
			int count = Math.min(seqLen - t, per - slot);
			byte[] rows = k.bytePage(page);
			for (int r = 0, ro = slot * bytesPerRow; r < count; r++, ro += bytesPerRow) {
				Q8_0KvCodec.decodeRange(rows, ro, kBase, headDim, row, 0);
				scoreRun(q, qBase, row, 0, 0, 1, headDim, scale, scores, t - lo + r);
			}
			t += count;
		}
	}

	/** Scores of {@code count} keys whose slices start at {@code rows[off]}, {@code stride} apart, into {@code scores[s0..)}. */
	static void scoreRun(float[] q, int qBase, float[] rows, int off, int stride, int count, int headDim,
			float scale, float[] scores, int s0) {
		for (int r = 0; r < count; r++, off += stride) {
			float dot = 0f;
			for (int d = 0; d < headDim; d++) dot += q[qBase + d] * rows[off + d];
			scores[s0 + r] = dot * scale;
		}
	}

	/**
	 * {@code out[outBase, outBase + headDim) += scores[t - lo] * V[t][kBase, kBase + headDim)} for {@code t} from
	 * {@code lo} up: each output element sums its products in key order, the serial loop's sums.
	 */
	static void addValues(KvRows v, int kBase, int lo, int seqLen, float[] out, int outBase, int headDim, int kvDim,
			float[] scores) {
		int per = v.rowsPerPage();
		if (v.type() == KvElementType.F16) {
			for (int t = lo; t < seqLen;) {
				int page = t / per;
				int slot = t - page * per;
				int count = Math.min(seqLen - t, per - slot);
				addRun(v.floatPage(page), slot * kvDim + kBase, kvDim, count, out, outBase, headDim, scores, t - lo);
				t += count;
			}
			return;
		}
		float[] row = row(headDim);
		int bytesPerRow = v.bytesPerRow();
		for (int t = lo; t < seqLen;) {
			int page = t / per;
			int slot = t - page * per;
			int count = Math.min(seqLen - t, per - slot);
			byte[] rows = v.bytePage(page);
			for (int r = 0, ro = slot * bytesPerRow; r < count; r++, ro += bytesPerRow) {
				Q8_0KvCodec.decodeRange(rows, ro, kBase, headDim, row, 0);
				addRun(row, 0, 0, 1, out, outBase, headDim, scores, t - lo + r);
			}
			t += count;
		}
	}

	/** Adds {@code count} weighted value slices starting at {@code rows[off]}, {@code stride} apart. */
	private static void addRun(float[] rows, int off, int stride, int count, float[] out, int outBase, int headDim,
			float[] scores, int s0) {
		for (int r = 0; r < count; r++, off += stride) {
			float w = scores[s0 + r];
			for (int d = 0; d < headDim; d++) out[outBase + d] += w * rows[off + d];
		}
	}

	/** The first key a row at context length {@code seqLen} attends to under {@code window} (0: none). */
	static int firstKey(int seqLen, int window) {
		return window > 0 && seqLen > window ? seqLen - window : 0;
	}
}
