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

import cab.ml.juno.lora.LoraDropout;
import cab.ml.juno.lora.LoraLearningRateSchedule;
import cab.ml.juno.node.LoraProjection;

/**
 * Builder-based configuration for LoRA training orchestration.
 *
 * <p>
 * Tier 1 fields: projection targets, learning rate, gradient accumulation, and
 * max gradient norm. Tier 2 adds schedule, AdamW decay, LoRA+, dropout, seed,
 * and validation early-stopping. Later tiers extend this builder rather than
 * adding competing {@code LoraTrainer.open} overloads.
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
	private final LoraLearningRateSchedule.Mode lrSchedule;
	private final double minLearningRate;
	private final int warmupUpdates;
	private final double weightDecay;
	private final double loraPlusRatio;
	private final float dropout;
	private final long seed;
	private final float validationSplit;
	private final int validationPatience;
	private final float validationMinDelta;
	private final boolean restoreBest;

	private LoraTrainingConfig(Builder b) {
		this.targets = List.copyOf(b.targets);
		this.rank = b.rank;
		this.alpha = b.alpha;
		this.learningRate = b.learningRate;
		this.gradientAccumulationSteps = b.gradientAccumulationSteps;
		this.maxGradNorm = b.maxGradNorm;
		this.lrSchedule = b.lrSchedule;
		this.minLearningRate = b.minLearningRate;
		this.warmupUpdates = b.warmupUpdates;
		this.weightDecay = b.weightDecay;
		this.loraPlusRatio = b.loraPlusRatio;
		this.dropout = b.dropout;
		this.seed = b.seed;
		this.validationSplit = b.validationSplit;
		this.validationPatience = b.validationPatience;
		this.validationMinDelta = b.validationMinDelta;
		this.restoreBest = b.restoreBest;
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

	public LoraLearningRateSchedule.Mode lrSchedule() {
		return lrSchedule;
	}

	public double minLearningRate() {
		return minLearningRate;
	}

	public int warmupUpdates() {
		return warmupUpdates;
	}

	public double weightDecay() {
		return weightDecay;
	}

	public double loraPlusRatio() {
		return loraPlusRatio;
	}

	public float dropout() {
		return dropout;
	}

	public long seed() {
		return seed;
	}

	public float validationSplit() {
		return validationSplit;
	}

	public int validationPatience() {
		return validationPatience;
	}

	public float validationMinDelta() {
		return validationMinDelta;
	}

	public boolean restoreBest() {
		return restoreBest;
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
		private LoraLearningRateSchedule.Mode lrSchedule = LoraLearningRateSchedule.Mode.CONSTANT;
		private double minLearningRate = 0.0;
		private int warmupUpdates = 0;
		private double weightDecay = 0.01;
		private double loraPlusRatio = 1.0;
		private float dropout = 0f;
		private long seed = 42L;
		private float validationSplit = 0f;
		private int validationPatience = 0;
		private float validationMinDelta = 0f;
		private boolean restoreBest = true;

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
			if (!(learningRate > 0) || !Double.isFinite(learningRate))
				throw new IllegalArgumentException("learningRate must be finite and > 0");
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
			if (maxGradNorm < 0f || !Float.isFinite(maxGradNorm))
				throw new IllegalArgumentException("maxGradNorm must be finite and >= 0");
			this.maxGradNorm = maxGradNorm;
			return this;
		}

		public Builder lrSchedule(LoraLearningRateSchedule.Mode lrSchedule) {
			if (lrSchedule == null)
				throw new IllegalArgumentException("lrSchedule must not be null");
			this.lrSchedule = lrSchedule;
			return this;
		}

		public Builder minLearningRate(double minLearningRate) {
			if (!Double.isFinite(minLearningRate) || minLearningRate < 0)
				throw new IllegalArgumentException("minLearningRate must be finite and >= 0");
			this.minLearningRate = minLearningRate;
			return this;
		}

		public Builder warmupUpdates(int warmupUpdates) {
			if (warmupUpdates < 0)
				throw new IllegalArgumentException("warmupUpdates must be >= 0");
			this.warmupUpdates = warmupUpdates;
			return this;
		}

		public Builder weightDecay(double weightDecay) {
			if (!Double.isFinite(weightDecay) || weightDecay < 0)
				throw new IllegalArgumentException("weightDecay must be finite and >= 0");
			this.weightDecay = weightDecay;
			return this;
		}

		public Builder loraPlusRatio(double loraPlusRatio) {
			if (!Double.isFinite(loraPlusRatio) || loraPlusRatio <= 0)
				throw new IllegalArgumentException("loraPlusRatio must be finite and > 0");
			this.loraPlusRatio = loraPlusRatio;
			return this;
		}

		public Builder dropout(float dropout) {
			LoraDropout.validateRate(dropout);
			this.dropout = dropout;
			return this;
		}

		public Builder seed(long seed) {
			this.seed = seed;
			return this;
		}

		public Builder validationSplit(float validationSplit) {
			if (!Float.isFinite(validationSplit) || validationSplit < 0f || validationSplit >= 1f)
				throw new IllegalArgumentException("validationSplit must be in [0, 1)");
			this.validationSplit = validationSplit;
			return this;
		}

		public Builder validationPatience(int validationPatience) {
			if (validationPatience < 0)
				throw new IllegalArgumentException("validationPatience must be >= 0");
			this.validationPatience = validationPatience;
			return this;
		}

		public Builder validationMinDelta(float validationMinDelta) {
			if (!Float.isFinite(validationMinDelta) || validationMinDelta < 0f)
				throw new IllegalArgumentException("validationMinDelta must be finite and >= 0");
			this.validationMinDelta = validationMinDelta;
			return this;
		}

		public Builder restoreBest(boolean restoreBest) {
			this.restoreBest = restoreBest;
			return this;
		}

		public LoraTrainingConfig build() {
			if (alpha < 0f)
				alpha = rank;
			if (minLearningRate > learningRate)
				throw new IllegalArgumentException("minLearningRate must be <= learningRate");
			return new LoraTrainingConfig(this);
		}
	}
}
