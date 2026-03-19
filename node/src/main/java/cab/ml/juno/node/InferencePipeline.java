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

import java.util.List;

/**
 * Contract for executing a forward pass through a pipeline of transformer
 * layers.
 *
 * Single-request: forward() — one request, one logit array. Batched:
 * forwardBatch() — N requests, N logit arrays, ONE GPU call.
 *
 * The default forwardBatch() implementation calls forward() N times serially —
 * all existing implementations get batching support for free. The real
 * GpuForwardPassHandler overrides forwardBatch() to use cuBLAS batched matrix
 * ops, turning N serial GPU launches into one, dramatically increasing
 * utilisation.
 */
public interface InferencePipeline {

	/**
	 * Run a single-request forward pass through all pipeline stages.
	 *
	 * @param requestId unique request identifier (for KV cache routing)
	 * @param tokens    full token sequence (prompt + generated so far)
	 * @param startPos  KV cache offset — tokens before this index are already
	 *                  cached
	 * @return logit array of size vocabSize() for the next token
	 */
	float[] forward(String requestId, int[] tokens, int startPos);

	/**
	 * Run a batched forward pass — N requests in, N logit arrays out.
	 *
	 * Default implementation calls forward() serially — correct but not fast.
	 * Override in GpuForwardPassHandler for true CUDA batching.
	 *
	 * Contract: - requestIds.size() == allTokens.size() == startPositions.size() -
	 * result[i] corresponds to requestIds.get(i) - All result arrays have length
	 * vocabSize()
	 *
	 * @param requestIds     one ID per request in the batch
	 * @param allTokens      one token array per request (different lengths OK)
	 * @param startPositions KV cache offset per request
	 * @return array of logit vectors, one per request
	 */
	default float[][] forwardBatch(List<String> requestIds, List<int[]> allTokens, List<Integer> startPositions) {
		int n = requestIds.size();
		float[][] results = new float[n][];
		for (int i = 0; i < n; i++) {
			results[i] = forward(requestIds.get(i), allTokens.get(i), startPositions.get(i));
		}
		return results;
	}

	/**
	 * Vocabulary size — length of every logit array returned by
	 * forward/forwardBatch.
	 */
	int vocabSize();
}