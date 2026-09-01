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
 * Input to a batched <em>decode</em> forward pass across N independent requests.
 *
 * <p>Unlike {@link BatchForwardRequest}, which batches {@code W} contiguous
 * prompt positions for <em>one</em> request (prefill window), this record carries
 * one token per request at that request's own KV position — the shape used by
 * static multi-session serving ({@code batch dim = N}, {@code seq len = 1}).
 *
 * <h3>Two modes</h3>
 * <ul>
 *   <li><b>First node</b>: {@code tokenIds[b]} is the current decode token for
 *       {@code requestIds.get(b)} at {@code startPositions[b]}.</li>
 *   <li><b>Subsequent nodes</b>: {@code activations} is flattened
 *       {@code batchSize × hiddenDim} from the previous shard.</li>
 * </ul>
 */
public record MultiDecodeForwardRequest(
		List<String> requestIds,
		int[] tokenIds,
		float[] activations,
		int[] startPositions,
		int batchSize) {

	public static MultiDecodeForwardRequest withTokens(List<String> requestIds, int[] lastTokenIds,
			int[] startPositions) {
		if (requestIds == null || requestIds.isEmpty())
			throw new IllegalArgumentException("requestIds must not be empty");
		if (lastTokenIds == null || lastTokenIds.length != requestIds.size())
			throw new IllegalArgumentException("lastTokenIds length must match requestIds");
		if (startPositions == null || startPositions.length != requestIds.size())
			throw new IllegalArgumentException("startPositions length must match requestIds");
		return new MultiDecodeForwardRequest(List.copyOf(requestIds), lastTokenIds, null,
				startPositions.clone(), requestIds.size());
	}

	public static MultiDecodeForwardRequest withActivations(List<String> requestIds, float[] activations,
			int batchSize, int[] startPositions) {
		if (requestIds == null || requestIds.isEmpty())
			throw new IllegalArgumentException("requestIds must not be empty");
		if (activations == null || activations.length == 0)
			throw new IllegalArgumentException("activations must not be empty");
		if (batchSize < 1)
			throw new IllegalArgumentException("batchSize must be >= 1");
		if (startPositions == null || startPositions.length != batchSize)
			throw new IllegalArgumentException("startPositions length must match batchSize");
		return new MultiDecodeForwardRequest(List.copyOf(requestIds), null, activations,
				startPositions.clone(), batchSize);
	}

	public boolean isFirstNode() {
		return tokenIds != null;
	}
}
