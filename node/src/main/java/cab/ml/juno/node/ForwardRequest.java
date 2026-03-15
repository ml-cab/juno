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
 * Input to a single node's forward pass computation.
 *
 * For the first node (hasEmbeddings=true): activations is null, tokenIds is
 * set. For subsequent nodes: activations carries the float[] from the previous
 * node.
 */
public record ForwardRequest(String requestId, int[] tokenIds, // non-null for first node only
		float[] activations, // non-null for subsequent nodes
		int startPosition // KV cache position (for incremental decode)
) {

	/** First node in the pipeline — takes raw token IDs. */
	public static ForwardRequest withTokens(String requestId, int[] tokenIds, int startPosition) {
		if (tokenIds == null || tokenIds.length == 0)
			throw new IllegalArgumentException("tokenIds must not be empty");
		return new ForwardRequest(requestId, tokenIds, null, startPosition);
	}

	/** Subsequent nodes — take activations from previous node. */
	public static ForwardRequest withActivations(String requestId, float[] activations, int startPosition) {
		if (activations == null || activations.length == 0)
			throw new IllegalArgumentException("activations must not be empty");
		return new ForwardRequest(requestId, null, activations, startPosition);
	}

	public boolean isFirstNode() {
		return tokenIds != null;
	}
}
