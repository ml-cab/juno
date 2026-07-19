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
package cab.ml.juno.player;

import java.util.List;

import cab.ml.juno.node.LoraProjection;

/**
 * Builder-based configuration for LoRA training orchestration.
 *
 * <p>
 * Tier 1 fields: projection targets, learning rate, gradient accumulation, and
 * max gradient norm. Later tiers extend this builder rather than adding
 * competing {@code LoraTrainer.open} overloads.
 *
 * <p>
 * Rank and alpha live here until Tier 3 introduces {@code LoraAdapterConfig};
 * keep them separable from algorithm-mode metadata.
 */
public final class LoraTrainingConfig {

	private final List<LoraProjection> targets;
	private final int rank;
	private final float alpha;
	private final double learningRate;
	private final int gradientAccumulationSteps;
	private final float maxGradNorm;

	private LoraTrainingConfig(Builder b) {
		this.targets = List.copyOf(b.targets);
		this.rank = b.rank;
		this.alpha = b.alpha;
		this.learningRate = b.learningRate;
		this.gradientAccumulationSteps = b.gradientAccumulationSteps;
		this.maxGradNorm = b.maxGradNorm;
	}

	public List<LoraProjection> targets() {
		return targets;
	}

	public int rank() {
		return rank;
	}

	public float alpha() {
		return alpha;
	}

	public double learningRate() {
		return learningRate;
	}

	public int gradientAccumulationSteps() {
		return gradientAccumulationSteps;
	}

	public float maxGradNorm() {
		return maxGradNorm;
	}

	public static Builder builder() {
		return new Builder();
	}

	public static final class Builder {
		private List<LoraProjection> targets = LoraProjection.qv();
		private int rank = 8;
		private float alpha = 8f;
		private double learningRate = 1e-4;
		private int gradientAccumulationSteps = 1;
		private float maxGradNorm = 1.0f;

		public Builder targets(List<LoraProjection> targets) {
			if (targets == null || targets.isEmpty())
				throw new IllegalArgumentException("targets must not be empty");
			this.targets = List.copyOf(targets);
			return this;
		}

		public Builder targets(String spec) {
			return targets(LoraProjection.parseTargets(spec));
		}

		public Builder rank(int rank) {
			if (rank < 1)
				throw new IllegalArgumentException("rank must be >= 1");
			this.rank = rank;
			return this;
		}

		public Builder alpha(float alpha) {
			this.alpha = alpha;
			return this;
		}

		public Builder learningRate(double learningRate) {
			if (learningRate <= 0)
				throw new IllegalArgumentException("learningRate must be > 0");
			this.learningRate = learningRate;
			return this;
		}

		public Builder gradientAccumulationSteps(int steps) {
			if (steps < 1)
				throw new IllegalArgumentException("gradientAccumulationSteps must be >= 1");
			this.gradientAccumulationSteps = steps;
			return this;
		}

		public Builder maxGradNorm(float maxGradNorm) {
			if (maxGradNorm < 0f)
				throw new IllegalArgumentException("maxGradNorm must be >= 0");
			this.maxGradNorm = maxGradNorm;
			return this;
		}

		public LoraTrainingConfig build() {
			if (alpha < 0f)
				alpha = rank;
			return new LoraTrainingConfig(this);
		}
	}
}
