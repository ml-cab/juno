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
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Contract for executing a forward pass through a pipeline of transformer
 * layers.
 *
 * Single-request: forward() — one request, one logit array. Batched:
 * forwardBatch() — N requests, N logit arrays, ONE GPU call.
 *
 * The default forwardBatch() implementation calls forward() N times serially —
 * all existing implementations get batching support for free. The real
 * LlamaTransformerHandler overrides forwardBatch() to use CUDA batched matrix
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
	 * Default implementation fans out each forward() call onto its own virtual
	 * thread so all N gRPC round-trips are in-flight simultaneously. Batch
	 * latency becomes max(per-node latency) instead of Σ(per-node latency).
	 *
	 * Implementations that can do true CUDA batching (e.g. LlamaTransformerHandler)
	 * should override this to use a single batched kernel call instead.
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
		if (n == 1) {
			// Fast path — skip task overhead for a single request
			return new float[][] { forward(requestIds.get(0), allTokens.get(0), startPositions.get(0)) };
		}

		// One virtual thread per request — all gRPC round-trips in flight simultaneously.
		// A shared executor is created here for the lifetime of this batch call; virtual
		// threads are cheap enough that this is fine. Callers that override this method
		// for CUDA batching never reach this code.
		ExecutorService vte = Executors.newVirtualThreadPerTaskExecutor();
		try {
			@SuppressWarnings("unchecked")
			CompletableFuture<float[]>[] futures = new CompletableFuture[n];
			for (int i = 0; i < n; i++) {
				final int idx = i;
				futures[idx] = CompletableFuture.supplyAsync(
						() -> forward(requestIds.get(idx), allTokens.get(idx), startPositions.get(idx)),
						vte);
			}
			float[][] results = new float[n][];
			for (int i = 0; i < n; i++) {
				results[i] = futures[i].join();
			}
			return results;
		} finally {
			vte.close(); // virtual-thread executors: close() is shutdown + awaitTermination
		}
	}

	/**
	 * Vocabulary size — length of every logit array returned by
	 * forward/forwardBatch.
	 */
	int vocabSize();
}