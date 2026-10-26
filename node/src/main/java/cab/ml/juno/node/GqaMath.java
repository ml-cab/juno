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

/**
 * Grouped-query attention math (QK^T + softmax + weighted-V-sum), extracted
 * from {@link LlamaTransformerHandler#gqaInto}/{@code gqa} so it can also serve
 * as the CPU oracle for the GPU-resident attention path ({@link CudaGqaAttention})
 * without duplicating the algorithm.
 *
 * <p>Causal masking is implicit via {@code seqLen} — the KV cache view passed
 * in only ever holds valid, already-written positions.
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
	 * @param kCache  row-major, stride {@code kvDim}, at least {@code seqLen} rows
	 * @param vCache  row-major, stride {@code kvDim}, at least {@code seqLen} rows
	 * @param window  keys a row attends over, counted back from its own; 0 for all
	 */
	static void attend(float[] q, float[] kCache, float[] vCache, int seqLen,
			float[] out, float[] scores, int numHeads, int headDim, int gqaRatio, int kvDim, int window) {
		float scale = (float) (1.0 / Math.sqrt(headDim));
		Arrays.fill(out, 0f);
		int lo = firstKey(seqLen, window);
		int n = seqLen - lo;

		for (int h = 0; h < numHeads; h++) {
			int kvHead = h / gqaRatio;
			int qBase = h * headDim;
			int kBase = kvHead * headDim;

			for (int t = lo; t < seqLen; t++) {
				float dot = 0f;
				int kOffset = t * kvDim + kBase;
				for (int d = 0; d < headDim; d++) dot += q[qBase + d] * kCache[kOffset + d];
				scores[t - lo] = dot * scale;
			}
			LlamaTransformerHandler.softmax(scores, n);

			int outBase = h * headDim;
			for (int t = lo; t < seqLen; t++) {
				int vOffset = t * kvDim + kBase;
				float w = scores[t - lo];
				for (int d = 0; d < headDim; d++) out[outBase + d] += w * vCache[vOffset + d];
			}
		}
	}

	/** As {@link #attend(float[], float[], float[], int, float[], float[], int, int, int, int, int)} with no window. */
	static void attend(float[] q, float[] kCache, float[] vCache, int seqLen,
			float[] out, float[] scores, int numHeads, int headDim, int gqaRatio, int kvDim) {
		attend(q, kCache, vCache, seqLen, out, scores, numHeads, headDim, gqaRatio, kvDim, 0);
	}

	/** Per-thread score scratch for {@link #attendPooled}; grows to the longest context seen and is reused. */
	private static final ThreadLocal<float[][]> SCORES = ThreadLocal.withInitial(() -> new float[][] { new float[0] });

	/**
	 * True when {@link #attendPooled} runs each head through {@link GqaVectorHead}: the Vector API module is
	 * present. Without it every head takes the serial loop's body, which is bit-identical to {@link #attend}; the
	 * vector head differs from it by float rounding only, within its oracle bound.
	 */
	static final boolean VECTOR_HEAD = VectorQuantKernels.AVAILABLE;

	/**
	 * {@link #attend(float[], float[], float[], int, float[], float[], int, int, int, int, int)} with the query heads
	 * spread over the CPU worker pool ({@link CpuThreads}). Each head writes its own slice of {@code out} with a
	 * score scratch of its own thread, through {@link GqaVectorHead} when {@link #VECTOR_HEAD} holds, and the output
	 * is the same bits at every thread count. Called from inside a pool task it runs on the calling thread.
	 */
	static void attendPooled(float[] q, float[] kCache, float[] vCache, int seqLen, float[] out, int numHeads,
			int headDim, int gqaRatio, int kvDim, int window) {
		attendPooled(q, kCache, vCache, seqLen, out, numHeads, headDim, gqaRatio, kvDim, window, VECTOR_HEAD);
	}

	/**
	 * {@link #attendPooled(float[], float[], float[], int, float[], int, int, int, int, int)} with the head form
	 * chosen by the caller: {@code vectorHead} needs the Vector API module. For tests that compare the two.
	 */
	static void attendPooled(float[] q, float[] kCache, float[] vCache, int seqLen, float[] out, int numHeads,
			int headDim, int gqaRatio, int kvDim, int window, boolean vectorHead) {
		float scale = (float) (1.0 / Math.sqrt(headDim));
		int lo = firstKey(seqLen, window);
		if (vectorHead)
			CpuThreads.forEach(numHeads, h -> GqaVectorHead.attend(q, kCache, vCache, seqLen, lo, out, h, headDim,
					gqaRatio, kvDim, scale, scores(seqLen)));
		else
			CpuThreads.forEach(numHeads, h -> attendHead(q, kCache, vCache, seqLen, lo, out, h, headDim, gqaRatio,
					kvDim, scale));
	}

	/** The calling thread's score scratch, at least {@code seqLen} long. */
	private static float[] scores(int seqLen) {
		float[][] holder = SCORES.get();
		if (holder[0].length < seqLen)
			holder[0] = new float[seqLen];
		return holder[0];
	}

	/** One head of {@link #attendPooled}: the serial loop's body for head {@code h}, writing only its own slice. */
	private static void attendHead(float[] q, float[] kCache, float[] vCache, int seqLen, int lo, float[] out, int h,
			int headDim, int gqaRatio, int kvDim, float scale) {
		float[] scores = scores(seqLen);
		int n = seqLen - lo;
		int kBase = (h / gqaRatio) * headDim;
		int qBase = h * headDim;
		for (int t = lo; t < seqLen; t++) {
			float dot = 0f;
			int kOffset = t * kvDim + kBase;
			for (int d = 0; d < headDim; d++) dot += q[qBase + d] * kCache[kOffset + d];
			scores[t - lo] = dot * scale;
		}
		LlamaTransformerHandler.softmax(scores, n);
		int outBase = h * headDim;
		java.util.Arrays.fill(out, outBase, outBase + headDim, 0f);
		for (int t = lo; t < seqLen; t++) {
			int vOffset = t * kvDim + kBase;
			float w = scores[t - lo];
			for (int d = 0; d < headDim; d++) out[outBase + d] += w * vCache[vOffset + d];
		}
	}

	/** The first key a row at context length {@code seqLen} attends to under {@code window} (0: none). */
	static int firstKey(int seqLen, int window) {
		return window > 0 && seqLen > window ? seqLen - window : 0;
	}
}
