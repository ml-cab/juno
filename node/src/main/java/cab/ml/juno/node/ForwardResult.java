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
 * Output from a single node's forward pass.
 *
 * For intermediate nodes: activations carries the hidden state to the next
 * node. For the last node (hasOutputProjection=true): logits is set instead.
 */
public record ForwardResult(String requestId, float[] activations, // non-null for intermediate nodes
		float[] logits, // non-null for last node only (float[vocabSize])
		long computeNanos // wall time for this node's computation
) {

	/** Intermediate node result. */
	public static ForwardResult activations(String requestId, float[] activations, long nanos) {
		return new ForwardResult(requestId, activations, null, nanos);
	}

	/** Last node result — carries final logits. */
	public static ForwardResult logits(String requestId, float[] logits, long nanos) {
		return new ForwardResult(requestId, null, logits, nanos);
	}

	public boolean isFinalNode() {
		return logits != null;
	}
}
