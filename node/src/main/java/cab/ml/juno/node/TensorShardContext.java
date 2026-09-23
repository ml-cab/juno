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

import cab.ml.juno.registry.TensorShardAssignment;

/**
 * Shard geometry for one node of a tensor-parallel group: the attention heads
 * and feed-forward slice width that node would own.
 *
 * Unlike {@link ShardContext} (pipeline-parallel — each node owns a distinct
 * contiguous layer range), a tensor-parallel node owns ALL transformer layers
 * and, by design, only a horizontal slice of the weight matrices:
 *
 * Attention: heads [headStart(), headEnd()) — column-parallel Q/K/V projection,
 * row-parallel output projection FFN: width slice [0, sliceDim()) —
 * column-parallel first linear, row-parallel second linear
 *
 * <p><b>Current state: geometry only.</b> No transformer handler reads this
 * record, and the rank a node receives when its shard is loaded is not used to
 * slice weights, so every node loads and runs the full model. The coordinator
 * broadcasts the activation, each node returns a complete logit vector, and the
 * coordinator sums them (star AllReduce): N times the memory and compute of one
 * node, and summed logits that are N times a single node's logits. Per-layer
 * weight slicing with a per-layer AllReduce is not implemented yet.
 *
 * Example of the geometry slicing would use — 3 nodes, 32 heads,
 * hiddenDim=2048: Rank 0: heads [0, 10), headDim=64, sliceDim=682 Rank 1: heads
 * [10, 21), headDim=64, sliceDim=682 Rank 2: heads [21, 32), headDim=64,
 * sliceDim=682
 */
public record TensorShardContext(String nodeId, int startLayer, // inclusive — always 0 in pure tensor-parallel mode
		int endLayer, // exclusive — always totalLayers in pure tensor-parallel mode
		int vocabSize, int hiddenDim, int numHeads, int tensorRank, // this node's rank in [0, tensorWorldSize)
		int tensorWorldSize // total nodes in the tensor-parallel group
) {

	public TensorShardContext {
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
		if (tensorRank < 0)
			throw new IllegalArgumentException("tensorRank must be >= 0");
		if (tensorWorldSize < 1)
			throw new IllegalArgumentException("tensorWorldSize must be >= 1");
		if (tensorRank >= tensorWorldSize)
			throw new IllegalArgumentException(
					"tensorRank (" + tensorRank + ") must be < tensorWorldSize (" + tensorWorldSize + ")");
		if (numHeads % 2 != 0)
			throw new IllegalArgumentException(
					"numHeads (" + numHeads + ") must be divisible by 2 for tensor-parallel mode. "
							+ "Attention heads pair up for RoPE rotation and cannot be odd.");
	}

	// ── Derived geometry ──────────────────────────────────────────────────────

	/** Number of attention heads owned by this node. */
	public int headsPerNode() {
		return numHeads / tensorWorldSize;
	}

	/** First attention head index (inclusive) owned by this node. */
	public int headStart() {
		return tensorRank * headsPerNode();
	}

	/** Last attention head index (exclusive) owned by this node. */
	public int headEnd() {
		return headStart() + headsPerNode();
	}

	/** Size of each attention head vector (hiddenDim / numHeads). */
	public int headDim() {
		return hiddenDim / numHeads;
	}

	/**
	 * Width of this node's output slice before AllReduce. Each node produces
	 * hiddenDim / tensorWorldSize elements per step. Summed across all nodes these
	 * reconstruct the full hiddenDim output.
	 */
	public int sliceDim() {
		return hiddenDim / tensorWorldSize;
	}

	/**
	 * Total number of transformer layers owned by this node (always all layers).
	 */
	public int layerCount() {
		return endLayer - startLayer;
	}

	// ── Factory ───────────────────────────────────────────────────────────────

	/** Build from a TensorShardAssignment and model-level shape metadata. */
	public static TensorShardContext from(TensorShardAssignment assignment, int vocabSize, int hiddenDim,
			int numHeads) {
		return new TensorShardContext(assignment.nodeId(), assignment.startLayer(), assignment.endLayer(), vocabSize,
				hiddenDim, numHeads, assignment.tensorRank(), assignment.tensorWorldSize());
	}
}