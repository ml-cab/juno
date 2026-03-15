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

/**
 * IMQ-inspired seed node scoring.
 *
 * Scores each node to determine shard assignment priority. Higher score = gets
 * assigned first = gets more layers.
 *
 * Formula: score = w1 * vramScore + w2 * stabilityScore + w3 *
 * connectivityScore
 *
 * Where: vramScore = vramFree / vramTotal (0.0 → 1.0) stabilityScore = 1.0 if
 * READY, 0.5 if IDLE, 0.0 otherwise connectivityScore = provided by caller (0.0
 * → 1.0) typically: 1.0 - (latencyMs / maxLatencyMs)
 *
 * Default weights: vram=0.5, stability=0.3, connectivity=0.2 Weights are
 * configurable — sum must equal 1.0.
 */
public final class SeedScorer {

	private final double wVram;
	private final double wStability;
	private final double wConnectivity;

	private SeedScorer(double wVram, double wStability, double wConnectivity) {
		double sum = wVram + wStability + wConnectivity;
		if (Math.abs(sum - 1.0) > 1e-6)
			throw new IllegalArgumentException("Weights must sum to 1.0, got: " + sum);
		this.wVram = wVram;
		this.wStability = wStability;
		this.wConnectivity = wConnectivity;
	}

	/** Default weights: vram=0.5, stability=0.3, connectivity=0.2 */
	public static SeedScorer defaults() {
		return new SeedScorer(0.5, 0.3, 0.2);
	}

	/** Custom weights — must sum to 1.0. */
	public static SeedScorer withWeights(double wVram, double wStability, double wConnectivity) {
		return new SeedScorer(wVram, wStability, wConnectivity);
	}

	/**
	 * Score a node. Higher = higher priority for shard assignment.
	 *
	 * @param node              the node to score
	 * @param connectivityScore caller-supplied network quality score (0.0 → 1.0)
	 * @return score in range [0.0, 1.0]
	 */
	public double score(NodeDescriptor node, double connectivityScore) {
		double vramScore = vramScore(node);
		double stabilityScore = stabilityScore(node);
		double connScore = clamp(connectivityScore);

		return wVram * vramScore + wStability * stabilityScore + wConnectivity * connScore;
	}

	// ── Component scorers ─────────────────────────────────────────────────────

	double vramScore(NodeDescriptor node) {
		if (node.vramTotalBytes() == 0)
			return 0.0;
		return clamp((double) node.vramFreeBytes() / node.vramTotalBytes());
	}

	double stabilityScore(NodeDescriptor node) {
		return switch (node.status()) {
		case READY -> 1.0;
		case IDLE -> 0.5;
		case LOADING -> 0.3;
		case DEGRADED -> 0.1;
		case OFFLINE -> 0.0;
		};
	}

	private double clamp(double v) {
		return Math.max(0.0, Math.min(1.0, v));
	}

	public double wVram() {
		return wVram;
	}

	public double wStability() {
		return wStability;
	}

	public double wConnectivity() {
		return wConnectivity;
	}
}
