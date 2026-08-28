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
 * Output from a batched forward pass covering a window of new prompt tokens.
 *
 * <h3>Two modes</h3>
 * <ul>
 *   <li><b>Intermediate node</b> ({@link #isFinalNode()} false): {@code activations}
 *       is a flattened {@code float[windowSize * hiddenDim]} matrix in row-major
 *       order; {@code lastLogits} is null. The next node receives this as a
 *       {@link BatchForwardRequest#withActivations} call.</li>
 *   <li><b>Final node</b> ({@link #isFinalNode()} true): {@code lastLogits} holds
 *       the logit vector for the <em>last</em> position in the window only —
 *       {@code float[vocabSize]}. {@code activations} is null. Only the final
 *       token's logits are needed after prefill; returning all {@code windowSize}
 *       logit vectors would waste allocation for a 32 K–150 K-wide vocabulary.</li>
 * </ul>
 *
 * @param requestId    unique request identifier
 * @param activations  flattened {@code windowSize * hiddenDim}; non-null for
 *                     intermediate nodes, null for final node
 * @param lastLogits   logits for the last window position; non-null for final
 *                     node, null for intermediate nodes
 * @param windowSize   number of positions that were processed
 * @param computeNanos wall time for this node's computation
 */
public record BatchForwardResult(
		String requestId,
		float[] activations,
		float[] lastLogits,
		int windowSize,
		long computeNanos) {

	/** True when this result comes from the last node (carries logits, not activations). */
	public boolean isFinalNode() {
		return lastLogits != null;
	}
}
