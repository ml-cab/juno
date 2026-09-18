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

/**
 * Output from a batched forward pass over a window of speculatively-drafted
 * tokens, verified against the target model.
 *
 * <p>Sibling to {@link BatchForwardResult}: same two-mode shape (intermediate
 * node carries flattened activations; final node carries logits), but where
 * {@link BatchForwardResult} discards every position's logits except the last
 * (prefill only needs the final token), this type keeps <b>every</b> window
 * position's logits — speculative-decode verification must compare the target
 * model's own prediction against the drafted token at each position, not just
 * the last one.
 *
 * @param requestId    unique request identifier
 * @param activations  flattened {@code windowSize * hiddenDim}; non-null for
 *                     intermediate nodes, null for final node
 * @param logits       flattened {@code windowSize * vocabSize}, row-major (one
 *                     row per window position); non-null for final node, null
 *                     for intermediate nodes
 * @param windowSize   number of positions that were processed
 * @param computeNanos wall time for this node's computation
 */
public record VerifyBatchResult(
		String requestId,
		float[] activations,
		float[] logits,
		int windowSize,
		long computeNanos) {

	/** True when this result comes from the last node (carries logits, not activations). */
	public boolean isFinalNode() {
		return logits != null;
	}

	/** Unflatten {@link #logits} into one {@code float[vocabSize]} row per window position. */
	public float[][] logitsPerPosition(int vocabSize) {
		float[][] rows = new float[windowSize][];
		for (int i = 0; i < windowSize; i++) {
			rows[i] = new float[vocabSize];
			System.arraycopy(logits, i * vocabSize, rows[i], 0, vocabSize);
		}
		return rows;
	}
}
