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

package cab.ml.juno.coordinator;

/**
 * Configuration for the micro-batching scheduler.
 *
 * The scheduler collects incoming requests into batches and dispatches each
 * batch as a single GenerationLoop.generateBatch() call. One forward pass
 * serves all requests in the batch simultaneously — GPU utilization scales with
 * batch size rather than staying at ~5% (single-request throughput).
 *
 * Two trigger conditions fire a batch dispatch (whichever comes first): 1.
 * batchWindowMs has elapsed since the first request in the current batch 2.
 * maxBatchSize requests have accumulated
 *
 * Static batching constraint: All requests in a batch start at generation step
 * 0 (freshly submitted). This avoids the complexity of continuous batching
 * (requests at different decode steps) while capturing 80%+ of the throughput
 * gain.
 *
 * Presets: defaults() — maxBatchSize=8, batchWindowMs=50 (production)
 * disabled() — maxBatchSize=1, batchWindowMs=0 (original per-request dispatch)
 */
public record BatchConfig(int maxBatchSize, long batchWindowMs) {

	public BatchConfig {
		if (maxBatchSize < 1)
			throw new IllegalArgumentException("maxBatchSize must be >= 1, got: " + maxBatchSize);
		if (batchWindowMs < 0)
			throw new IllegalArgumentException("batchWindowMs must be >= 0, got: " + batchWindowMs);
	}

	/** Production default — batch up to 8 requests within a 50ms window. */
	public static BatchConfig defaults() {
		return new BatchConfig(8, 50);
	}

	/**
	 * Disabled — each request dispatched immediately on its own virtual thread.
	 * Identical behavior to the original RequestScheduler.
	 */
	public static BatchConfig disabled() {
		return new BatchConfig(1, 0);
	}

	/** Custom config factory. */
	public static BatchConfig of(int maxBatchSize, long batchWindowMs) {
		return new BatchConfig(maxBatchSize, batchWindowMs);
	}

	/** Whether batching is active (maxBatchSize > 1). */
	public boolean isBatchingEnabled() {
		return maxBatchSize > 1;
	}
}