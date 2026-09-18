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
 * all existing implementations get batching support for free.
 * {@link LocalInferencePipeline} overrides forwardBatch to route N decode
 * streams through {@link ForwardPassHandler#forwardMultiDecode}; Llama
 * handlers batch linear projections in one GEMM per layer.
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
	 * Override in LlamaTransformerHandler for true CUDA batching.
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

	/**
	 * Prefill the KV cache for a contiguous window of new prompt tokens in a
	 * single batched call. Logits are discarded — only the KV state written by
	 * each position's forward pass matters.
	 *
	 * <p><b>Correctness-preserving default</b>: loops {@code newTokens.length}
	 * times, calling {@link #forward} once per token with a minimal single-element
	 * token array — byte-for-byte equivalent to the old per-position prefill loop
	 * in {@link cab.ml.juno.coordinator.GenerationLoop}, minus the growing
	 * {@code copyOfRange} prefix. Implementations that do not override this are
	 * correct but gain no speed benefit; {@link LocalInferencePipeline} overrides
	 * to call the batched handler chain end-to-end.
	 *
	 * @param requestId    KV cache key (session or request id)
	 * @param newTokens    the new tokens to prefill, {@code promptIds[startPosition..end-1]}
	 * @param startPosition KV cache offset of {@code newTokens[0]}
	 */
	default void prefillBatch(String requestId, int[] newTokens, int startPosition) {
		PrefillBatchJfr.run(requestId, newTokens, startPosition, () -> {
			for (int p = 0; p < newTokens.length; p++)
				forward(requestId, new int[] { newTokens[p] }, startPosition + p);
		});
	}

	/**
	 * Verify a window of speculatively-drafted tokens against this pipeline's own
	 * model in one pass, returning the target model's logits at <em>every</em>
	 * position in the window — unlike {@link #prefillBatch}, which discards all
	 * but the last position's logits since prefill has nothing to compare against.
	 *
	 * <p><b>Window shape — same off-by-one rule as {@link #forward}</b>: element
	 * {@code i} of {@code draftTokens} is the token this pipeline treats as already
	 * sitting at position {@code startPosition + i}, and {@code result[i]} is the
	 * prediction for position {@code startPosition + i + 1}. To verify {@code M}
	 * newly-drafted tokens the caller must therefore pass the <em>already-confirmed</em>
	 * last token as {@code draftTokens[0]} (echoing it — this is a no-op re-write of
	 * that position's KV entry) followed by the first {@code M - 1} drafted tokens,
	 * with {@code startPosition} equal to that confirmed token's own position
	 * (exactly the value already passed to {@link #forward} for a plain decode step
	 * at this point in the sequence). {@code result[i]} is then compared against
	 * the {@code i}-th <em>drafted</em> token. Passing the drafted tokens directly
	 * as {@code draftTokens[0..M-1]} instead would silently overwrite the
	 * already-confirmed token's KV entry with an unverified one and corrupt every
	 * later step — KV storage here is indexed by absolute position, not
	 * append-only, so a wrong token at an already-real position is not
	 * self-correcting.
	 *
	 * <p>On a mismatch, the target model's own prediction at that position (not the
	 * draft) is what must be emitted, and every later position in this window is
	 * moot (its KV entries get silently overwritten by the next real write at that
	 * position).
	 *
	 * <p><b>Correctness-preserving default</b>: loops {@code draftTokens.length}
	 * times through {@link #forward}, one window position per call — token-identity
	 * correct, but no speed benefit over plain sequential decoding.
	 * {@link LocalInferencePipeline} overrides this to route the whole window
	 * through {@link ForwardPassHandler#forwardVerify} in one batched GEMM pass per
	 * layer instead — the actual speculative-decoding speed win.
	 *
	 * @param requestId     KV cache key (session or request id)
	 * @param draftTokens   window of {@code M} token ids: the already-confirmed
	 *                      token at {@code startPosition}, then {@code M - 1}
	 *                      speculatively-drafted continuations
	 * @param startPosition KV cache position of {@code draftTokens[0]} — the
	 *                      already-confirmed token's own (existing) position
	 * @return one logits row per window position, {@code result[i].length == vocabSize()}
	 */
	default float[][] verifyDraft(String requestId, int[] draftTokens, int startPosition) {
		int M = draftTokens.length;
		float[][] logits = new float[M][];
		for (int i = 0; i < M; i++) {
			logits[i] = forward(requestId, new int[] { draftTokens[i] }, startPosition + i);
		}
		return logits;
	}

	/**
	 * Release {@code requestId}'s per-request KV state from every stage in
	 * this pipeline — see {@link ForwardPassHandler#evict} for what that
	 * state is and why it must be released explicitly for stateless (no
	 * session) requests.
	 *
	 * <p><b>Correctness-preserving default</b>: no-op, matching
	 * {@link ForwardPassHandler#evict}'s default. {@link LocalInferencePipeline}
	 * overrides this to call {@code evict} on every stage's handler.
	 */
	default void evict(String requestId) {
	}

	/**
	 * Causal prefill over {@code tokens}, returning the RMS/LayerNorm-normalized
	 * hidden vector at every position (before the LM head) for
	 * {@code POST /v1/embeddings} pooling.
	 *
	 * <p><b>Fail closed by default.</b> Embeddings extraction is only implemented
	 * on {@link LocalInferencePipeline} (single JVM / single shard); distributed
	 * pipelines (gRPC node clients, tensor/pipeline-parallel cluster launchers)
	 * throw here rather than silently returning a wrong or zeroed vector, per
	 * ROADMAP Execution rule 6 (no silent flag ignore).
	 *
	 * @param requestId unique key for the per-request KV state used during prefill
	 * @param tokens    prompt token ids, must not be empty
	 * @return {@code hidden[pos][hiddenDim]}, one row per position in {@code tokens}
	 * @throws UnsupportedOperationException when this pipeline does not support
	 *                                        embeddings extraction
	 */
	default float[][] embedTokens(String requestId, int[] tokens) {
		throw new UnsupportedOperationException(
				"Embeddings extraction is not supported by " + getClass().getSimpleName()
						+ "; /v1/embeddings requires a single-process or single-shard pipeline (local mode).");
	}
}