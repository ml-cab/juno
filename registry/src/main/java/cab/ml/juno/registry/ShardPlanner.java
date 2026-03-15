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
package cab.ml.juno.registry;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Greedy shard assignment algorithm.
 *
 * Given a list of healthy nodes and model metadata, computes the optimal
 * ShardMap — contiguous layer blocks, VRAM-aware, highest-scoring nodes get
 * first pick.
 *
 * Algorithm: 1. Filter nodes: only IDLE or READY 2. Sort by seedScore
 * descending 3. For each node: assign as many layers as fit in usableVram 4. If
 * all layers are not assigned: throw InsufficientClusterVramException
 *
 * Contiguous block assignment is critical for pipeline parallelism —
 * activations flow node[0] → node[1] → ... → node[N] in strict order.
 */
public final class ShardPlanner {

	private final SeedScorer scorer;

	public ShardPlanner(SeedScorer scorer) {
		if (scorer == null)
			throw new IllegalArgumentException("scorer must not be null");
		this.scorer = scorer;
	}

	public static ShardPlanner create() {
		return new ShardPlanner(SeedScorer.defaults());
	}

	/**
	 * Compute a ShardMap for the given model across the available nodes.
	 *
	 * @param modelId           model identifier
	 * @param totalLayers       total transformer layers in the model
	 * @param vramPerLayerBytes estimated VRAM required per layer
	 * @param nodes             all registered nodes (any status — filtered
	 *                          internally)
	 * @return a validated ShardMap ready for use
	 * @throws InsufficientClusterVramException if cluster has insufficient VRAM
	 */
	public ShardMap plan(String modelId, int totalLayers, long vramPerLayerBytes, List<NodeDescriptor> nodes) {

		if (totalLayers < 1)
			throw new IllegalArgumentException("totalLayers must be >= 1");
		if (vramPerLayerBytes < 1)
			throw new IllegalArgumentException("vramPerLayerBytes must be >= 1");

		// 1. Score and sort eligible nodes
		List<NodeDescriptor> eligible = nodes.stream().filter(NodeDescriptor::isAvailable)
				.sorted(Comparator.comparingDouble(n -> -scorer.score(n, 1.0))).toList();

		if (eligible.isEmpty())
			throw new InsufficientClusterVramException("No eligible nodes available for model: " + modelId);

		// 2. Greedy assignment
		List<ShardAssignment> assignments = new ArrayList<>();
		int currentLayer = 0;

		for (NodeDescriptor node : eligible) {
			if (currentLayer >= totalLayers)
				break;

			long layersFit = node.usableVramBytes() / vramPerLayerBytes;
			if (layersFit < 1)
				continue; // node can't hold even one layer

			int remainingLayers = totalLayers - currentLayer;
			int remainingNodes = eligible.size() - assignments.size();

			long maxLayers = Math.min(layersFit, remainingLayers - (remainingNodes - 1) // leave ≥1 layer per remaining
																						// node
			);

			int endLayer = currentLayer + (int) maxLayers;

			assignments.add(new ShardAssignment(node.nodeId(), node.host(), node.grpcPort(), currentLayer, endLayer,
					currentLayer == 0, // hasEmbeddings — first node only
					endLayer == totalLayers // hasOutputProjection — last node only
			));

			currentLayer = endLayer;
		}

		// 3. Validate all layers were assigned
		if (currentLayer < totalLayers) {
			throw new InsufficientClusterVramException(
					String.format(
							"Cluster could only assign %d of %d layers for model '%s'. "
									+ "Need at least %.2f GB more VRAM.",
							currentLayer, totalLayers, modelId,
							((totalLayers - currentLayer) * vramPerLayerBytes) / (1024.0 * 1024 * 1024)));
		}

		ShardMap map = new ShardMap(modelId, totalLayers, assignments, Instant.now());
		map.validateCoverage(); // sanity check — should never fail
		return map;
	}
}
