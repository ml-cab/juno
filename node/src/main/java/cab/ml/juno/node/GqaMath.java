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
	 * @param kCache  row-major, stride {@code kvDim}, at least {@code seqLen} rows
	 * @param vCache  row-major, stride {@code kvDim}, at least {@code seqLen} rows
	 */
	static void attend(float[] q, float[] kCache, float[] vCache, int seqLen,
			float[] out, float[] scores, int numHeads, int headDim, int gqaRatio, int kvDim) {
		float scale = (float) (1.0 / Math.sqrt(headDim));
		Arrays.fill(out, 0f);

		for (int h = 0; h < numHeads; h++) {
			int kvHead = h / gqaRatio;
			int qBase = h * headDim;
			int kBase = kvHead * headDim;

			for (int t = 0; t < seqLen; t++) {
				float dot = 0f;
				int kOffset = t * kvDim + kBase;
				for (int d = 0; d < headDim; d++) dot += q[qBase + d] * kCache[kOffset + d];
				scores[t] = dot * scale;
			}
			LlamaTransformerHandler.softmax(scores, seqLen);

			int outBase = h * headDim;
			for (int t = 0; t < seqLen; t++) {
				int vOffset = t * kvDim + kBase;
				float w = scores[t];
				for (int d = 0; d < headDim; d++) out[outBase + d] += w * vCache[vOffset + d];
			}
		}
	}
}
