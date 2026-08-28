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

import java.io.Serializable;
import java.time.Instant;
import java.util.List;

/**
 * Complete layer-to-node assignment for a loaded model. Stored in Hazelcast
 * IMap("shard-maps") keyed by modelId.
 *
 * Assignments are ordered by startLayer ascending — this is the pipeline
 * execution order: coordinator sends activations to assignments.get(0), then
 * .get(1), etc.
 *
 * Recomputed by the registry whenever cluster membership changes.
 */
public record ShardMap(String modelId, int totalLayers, List<ShardAssignment> assignments, Instant computedAt)
		implements Serializable {

	public ShardMap {
		if (modelId == null || modelId.isBlank())
			throw new IllegalArgumentException("modelId must not be blank");
		if (totalLayers < 1)
			throw new IllegalArgumentException("totalLayers must be >= 1");
		if (assignments == null || assignments.isEmpty())
			throw new IllegalArgumentException("assignments must not be empty");

		// Defensive copy — shard maps are immutable once computed
		assignments = List.copyOf(assignments);
	}

	/** First node in the pipeline — holds embeddings. */
	public ShardAssignment firstNode() {
		return assignments.get(0);
	}

	/** Last node in the pipeline — holds output projection. */
	public ShardAssignment lastNode() {
		return assignments.get(assignments.size() - 1);
	}

	/** Number of nodes this model is split across. */
	public int nodeCount() {
		return assignments.size();
	}

	/**
	 * Verify all layers are covered exactly once with no gaps or overlaps.
	 * 
	 * @throws IllegalStateException if coverage is invalid
	 */
	public void validateCoverage() {
		int expected = 0;
		for (ShardAssignment a : assignments) {
			if (a.startLayer() != expected) {
				throw new IllegalStateException(
						"Layer gap or overlap at layer " + expected + " — assignment starts at " + a.startLayer());
			}
			expected = a.endLayer();
		}
		if (expected != totalLayers) {
			throw new IllegalStateException("ShardMap covers " + expected + " layers but model has " + totalLayers);
		}
	}

	/**
	 * Build an evenly-split ShardMap across {@code nodeCount} local pipeline
	 * stages — each node gets {@code totalLayers / nodeCount} layers, with any
	 * remainder distributed one-per-node starting from the first.
	 *
	 * <p>For in-process pipeline-parallelism simulation (e.g. {@code ./juno
	 * local --nodes N}), where every "node" is really just a stage running in
	 * the same JVM with no real VRAM constraint between stages. Use
	 * {@link ShardPlanner#plan} instead for real multi-machine clusters, where
	 * nodes genuinely differ in available VRAM and the greedy, VRAM-aware
	 * algorithm is the correct one — this method intentionally has no VRAM
	 * concept at all, rather than faking one, since a fake per-node VRAM
	 * figure large enough to "always fit" causes {@link ShardPlanner}'s greedy
	 * first-node-takes-what-fits algorithm to hand nearly the whole model to
	 * node 0 and one layer each to every other node — correct per its own
	 * contract, but not what an even local split needs.
	 *
	 * @param nodeCount number of local stages to split across; capped to
	 *                  {@code totalLayers} if larger, since a stage cannot
	 *                  hold zero layers
	 */
	public static ShardMap evenSplit(String modelId, int totalLayers, int nodeCount) {
		if (nodeCount < 1)
			throw new IllegalArgumentException("nodeCount must be >= 1");

		int effectiveNodeCount = Math.min(nodeCount, totalLayers);
		int base = totalLayers / effectiveNodeCount;
		int remainder = totalLayers % effectiveNodeCount;

		List<ShardAssignment> assignments = new java.util.ArrayList<>();
		int current = 0;
		for (int i = 0; i < effectiveNodeCount; i++) {
			int layerCount = base + (i < remainder ? 1 : 0);
			int end = current + layerCount;
			assignments.add(new ShardAssignment("node-" + i, "localhost", 9092 + i, current, end,
					current == 0,           // hasEmbeddings — first node only
					end == totalLayers));   // hasOutputProjection — last node only
			current = end;
		}

		return new ShardMap(modelId, totalLayers, assignments, Instant.now());
	}
}