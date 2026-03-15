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

import cab.ml.juno.registry.ShardAssignment;

/**
 * Runtime context for this node's shard assignment. Derived from the ShardMap
 * computed by the registry.
 *
 * Tells the ForwardPassHandler: - which layers to compute - whether to run
 * embedding lookup first - whether to run output projection last
 */
public record ShardContext(String nodeId, int startLayer, int endLayer, boolean hasEmbeddings,
		boolean hasOutputProjection, int vocabSize, int hiddenDim, int numHeads) {

	public ShardContext {
		if (startLayer < 0)
			throw new IllegalArgumentException("startLayer must be >= 0");
		if (endLayer <= startLayer)
			throw new IllegalArgumentException("endLayer must be > startLayer");
		if (vocabSize < 1)
			throw new IllegalArgumentException("vocabSize must be >= 1");
		if (hiddenDim < 1)
			throw new IllegalArgumentException("hiddenDim must be >= 1");
		if (numHeads < 1)
			throw new IllegalArgumentException("numHeads must be >= 1");
	}

	/** Number of transformer layers this node owns. */
	public int layerCount() {
		return endLayer - startLayer;
	}

	/** Build from a ShardAssignment + model metadata. */
	public static ShardContext from(ShardAssignment assignment, int vocabSize, int hiddenDim, int numHeads) {
		return new ShardContext(assignment.nodeId(), assignment.startLayer(), assignment.endLayer(),
				assignment.hasEmbeddings(), assignment.hasOutputProjection(), vocabSize, hiddenDim, numHeads);
	}
}
