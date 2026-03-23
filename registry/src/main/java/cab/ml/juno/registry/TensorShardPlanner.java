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

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Tensor-parallel shard planner.
 *
 * All eligible nodes are assigned the full transformer layer range [0,
 * totalLayers). Each node receives a unique tensorRank in [0, worldSize) that
 * determines which horizontal slice of the attention heads and FFN width it
 * owns.
 *
 * Constraint: numHeads must be even (divisible by 2). Transformer attention
 * heads always pair up for RoPE sin/cos rotation, so an odd head count is
 * architecturally invalid regardless of node count. Uneven distribution across
 * nodes is fine — heads are assigned by ceiling-division so ranks differ by at
 * most 1 (e.g. 32 heads × 3 nodes → 10 / 11 / 11).
 *
 * Node ordering: sorted by seedScore descending (highest-quality nodes get
 * lower ranks so they are dispatched first by the coordinator).
 *
 * Example — 3 nodes, 32 heads, 22 layers: Rank 0: heads [0, 10), layers [0, 22)
 * Rank 1: heads [10, 21), layers [0, 22) (rounding if not divisible) Rank 2:
 * heads [21, 32), layers [0, 22)
 *
 * Contrast with ShardPlanner (pipeline parallel) which gives each node a
 * distinct, non-overlapping layer range.
 */
public final class TensorShardPlanner {

	private final SeedScorer scorer;

	public TensorShardPlanner(SeedScorer scorer) {
		if (scorer == null)
			throw new IllegalArgumentException("scorer must not be null");
		this.scorer = scorer;
	}

	public static TensorShardPlanner create() {
		return new TensorShardPlanner(SeedScorer.defaults());
	}

	/**
	 * Compute tensor-parallel assignments for all eligible nodes.
	 *
	 * @param modelId     model identifier (for error messages)
	 * @param totalLayers total number of transformer layers in the model
	 * @param numHeads    total attention heads — must be even (divisible by 2)
	 * @param nodes       all registered nodes (filtered internally to IDLE | READY)
	 * @return immutable list of TensorShardAssignment, one per eligible node,
	 *         ordered by rank
	 * @throws InsufficientClusterVramException if no eligible nodes are available
	 * @throws IllegalArgumentException         if numHeads is not divisible by 2
	 */
	public List<TensorShardAssignment> plan(String modelId, int totalLayers, int numHeads, List<NodeDescriptor> nodes) {

		if (totalLayers < 1)
			throw new IllegalArgumentException("totalLayers must be >= 1");
		if (numHeads < 1)
			throw new IllegalArgumentException("numHeads must be >= 1");

		List<NodeDescriptor> eligible = nodes.stream().filter(NodeDescriptor::isAvailable)
				.sorted(Comparator.comparingDouble(n -> -scorer.score(n, 1.0))).toList();

		if (eligible.isEmpty())
			throw new InsufficientClusterVramException(
					"No eligible nodes available for tensor-parallel model: " + modelId);

		int worldSize = eligible.size();
		if (numHeads % 2 != 0)
			throw new IllegalArgumentException(
					String.format("numHeads (%d) must be divisible by 2 for tensor-parallel mode. "
							+ "Attention heads pair up for RoPE rotation and cannot be split across "
							+ "nodes when the total count is odd.", numHeads));

		List<TensorShardAssignment> assignments = new ArrayList<>(worldSize);
		for (int rank = 0; rank < worldSize; rank++) {
			NodeDescriptor node = eligible.get(rank);
			assignments.add(new TensorShardAssignment(node.nodeId(), node.host(), node.grpcPort(), 0, // startLayer —
																										// all nodes own
																										// all layers
					totalLayers, // endLayer
					true, // hasEmbeddings — all nodes embed independently
					true, // hasOutputProjection — all nodes produce partial logits
					rank, worldSize));
		}
		return List.copyOf(assignments);
	}
}