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
package cab.ml.juno.lora;

/**
 * Learning-rate schedule for LoRA optimizer updates.
 *
 * <p>
 * Update indices are one-based. Warmup is linear:
 * {@code peakLr * update / warmupUpdates}. After warmup, cosine decay runs from
 * peak to minimum over the remaining updates, then clamps at minimum.
 */
public final class LoraLearningRateSchedule {

	public enum Mode {
		CONSTANT, COSINE
	}

	private final Mode mode;
	private final double peakLr;
	private final double minLr;
	private final int warmupUpdates;
	private final int totalUpdates;

	private LoraLearningRateSchedule(Mode mode, double peakLr, double minLr, int warmupUpdates, int totalUpdates) {
		this.mode = mode;
		this.peakLr = peakLr;
		this.minLr = minLr;
		this.warmupUpdates = warmupUpdates;
		this.totalUpdates = totalUpdates;
	}

	public static LoraLearningRateSchedule constant(double learningRate) {
		validateRate(learningRate, "learningRate");
		return new LoraLearningRateSchedule(Mode.CONSTANT, learningRate, learningRate, 0, 0);
	}

	/**
	 * Warmup then cosine decay from {@code peakLr} to {@code minLr}.
	 *
	 * @param peakLr         peak learning rate after warmup
	 * @param minLr          floor learning rate ({@code <= peakLr})
	 * @param warmupUpdates  non-negative warmup length in optimizer updates
	 * @param totalUpdates   total planned optimizer updates ({@code >= warmupUpdates})
	 */
	public static LoraLearningRateSchedule warmupCosine(double peakLr, double minLr, int warmupUpdates,
			int totalUpdates) {
		validateRate(peakLr, "peakLr");
		validateRate(minLr, "minLr");
		if (minLr > peakLr)
			throw new IllegalArgumentException("minLr must be <= peakLr");
		if (warmupUpdates < 0)
			throw new IllegalArgumentException("warmupUpdates must be >= 0");
		if (totalUpdates < warmupUpdates)
			throw new IllegalArgumentException("totalUpdates must be >= warmupUpdates");
		if (totalUpdates < 1)
			throw new IllegalArgumentException("totalUpdates must be >= 1");
		return new LoraLearningRateSchedule(Mode.COSINE, peakLr, minLr, warmupUpdates, totalUpdates);
	}

	/**
	 * Learning rate for a one-based optimizer update index.
	 */
	public double learningRate(int update) {
		if (update < 1)
			throw new IllegalArgumentException("update must be >= 1");
		if (mode == Mode.CONSTANT)
			return peakLr;
		if (update >= totalUpdates)
			return minLr;
		if (warmupUpdates > 0 && update <= warmupUpdates)
			return peakLr * ((double) update / (double) warmupUpdates);
		int cosineSpan = totalUpdates - warmupUpdates;
		if (cosineSpan <= 1)
			return minLr;
		// First post-warmup update is progress 0 (peak); final update is progress 1 (min).
		double progress = (double) (update - warmupUpdates - 1) / (double) (cosineSpan - 1);
		if (progress < 0.0)
			progress = 0.0;
		if (progress > 1.0)
			progress = 1.0;
		double cosine = 0.5 * (1.0 + Math.cos(Math.PI * progress));
		return minLr + (peakLr - minLr) * cosine;
	}

	public Mode mode() {
		return mode;
	}

	public double peakLr() {
		return peakLr;
	}

	public double minLr() {
		return minLr;
	}

	public int warmupUpdates() {
		return warmupUpdates;
	}

	public int totalUpdates() {
		return totalUpdates;
	}

	private static void validateRate(double rate, String name) {
		if (!Double.isFinite(rate) || rate < 0.0)
			throw new IllegalArgumentException(name + " must be finite and >= 0");
	}
}
