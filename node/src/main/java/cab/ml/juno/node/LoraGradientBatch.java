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

/**
 * Aggregates {@link LoraGradientResult} values across an accumulation group.
 */
public final class LoraGradientBatch {

	private float lossSum;
	private int predictionCount;
	private int chunkCount;
	private long forwardMs;
	private long backwardMs;
	private final LoraStepTiming timing = new LoraStepTiming();

	public void add(LoraGradientResult r) {
		lossSum += r.lossSum();
		predictionCount += r.predictionCount();
		chunkCount++;
		forwardMs += r.forwardMs();
		backwardMs += r.backwardMs();
		timing.add(r.timing());
	}

	public float lossSum() {
		return lossSum;
	}

	public int predictionCount() {
		return predictionCount;
	}

	public int chunkCount() {
		return chunkCount;
	}

	public long forwardMs() {
		return forwardMs;
	}

	public long backwardMs() {
		return backwardMs;
	}

	public LoraStepTiming timing() {
		return timing;
	}

	/** Token-weighted mean loss across the batch; {@link Float#NaN} if empty. */
	public float meanLoss() {
		if (predictionCount == 0)
			return Float.NaN;
		return lossSum / predictionCount;
	}

	public void clear() {
		lossSum = 0f;
		predictionCount = 0;
		chunkCount = 0;
		forwardMs = 0L;
		backwardMs = 0L;
		timing.clear();
	}

	public boolean isEmpty() {
		return chunkCount == 0;
	}
}
