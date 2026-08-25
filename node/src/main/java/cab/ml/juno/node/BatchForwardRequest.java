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
 * Input to a batched forward pass covering a window of new prompt tokens.
 *
 * <p>Unlike {@link ForwardRequest}, which carries the entire token sequence
 * from position 0 and processes a single position per call, this record
 * carries <em>only the new window</em> — the tokens that have not yet been
 * prefilled — together with {@code startPosition}, the KV cache offset of
 * {@code tokenIds[0]}. Eliminating the O(N&sup2;) growing-prefix
 * {@code copyOfRange} pattern in the calling loop is a correctness-preserving
 * side-effect of this representation.
 *
 * <h3>Two modes</h3>
 * <ul>
 *   <li><b>First node</b> ({@link #isFirstNode()} true): {@code tokenIds} is
 *       set; each element is looked up in the token embedding table
 *       independently.</li>
 *   <li><b>Subsequent nodes</b>: {@code activations} carries a flattened
 *       {@code float[windowSize * hiddenDim]} matrix — all window-row
 *       activations from the previous node in row-major order.</li>
 * </ul>
 *
 * @param requestId     unique request identifier (matches pipeline KV key)
 * @param tokenIds      new token IDs only, in order; non-null for first node
 * @param activations   flattened {@code windowSize * hiddenDim}; non-null for
 *                      subsequent nodes
 * @param startPosition KV cache position of {@code tokenIds[0]}
 * @param windowSize    number of positions in this batch
 *                      ({@code tokenIds.length} or
 *                      {@code activations.length / hiddenDim})
 */
public record BatchForwardRequest(
		String requestId,
		int[] tokenIds,
		float[] activations,
		int startPosition,
		int windowSize) {

	/** Build a first-node request from a contiguous range of new token IDs. */
	public static BatchForwardRequest withTokens(String requestId, int[] tokenIds, int startPosition) {
		if (tokenIds == null || tokenIds.length == 0)
			throw new IllegalArgumentException("tokenIds must not be empty");
		return new BatchForwardRequest(requestId, tokenIds, null, startPosition, tokenIds.length);
	}

	/**
	 * Build a subsequent-node request from flattened activations
	 * ({@code windowSize * hiddenDim} elements).
	 */
	public static BatchForwardRequest withActivations(String requestId, float[] activations,
			int windowSize, int startPosition) {
		if (activations == null || activations.length == 0)
			throw new IllegalArgumentException("activations must not be empty");
		if (windowSize < 1)
			throw new IllegalArgumentException("windowSize must be >= 1");
		return new BatchForwardRequest(requestId, null, activations, startPosition, windowSize);
	}

	/** True when this is the first node in the pipeline (token embedding lookup needed). */
	public boolean isFirstNode() {
		return tokenIds != null;
	}
}
