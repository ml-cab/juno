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
 * Retry configuration for fault-tolerant pipeline calls.
 *
 * maxAttempts: total number of node attempts (1 = no retry, 2 = one retry,
 * etc.) backoffMs: fixed delay between attempts (0 = immediate retry)
 *
 * Retries are spread across different nodes when available — if node A fails on
 * attempt 1, attempt 2 goes to node B. This handles both transient node errors
 * and hard node failures.
 *
 * Presets: none() — 1 attempt — call fails immediately if the pipeline throws
 * once() — 2 attempts, 50ms backoff — default for most requests aggressive() —
 * 3 attempts, 100ms backoff — for HIGH priority requests
 */
public record RetryPolicy(int maxAttempts, long backoffMs) {

	public RetryPolicy {
		if (maxAttempts < 1)
			throw new IllegalArgumentException("maxAttempts must be >= 1, got: " + maxAttempts);
		if (backoffMs < 0)
			throw new IllegalArgumentException("backoffMs must be >= 0, got: " + backoffMs);
	}

	/** No retry — fail immediately on first exception. */
	public static RetryPolicy none() {
		return new RetryPolicy(1, 0);
	}

	/** One retry on a different node after 50ms. Default for most requests. */
	public static RetryPolicy once() {
		return new RetryPolicy(2, 50);
	}

	/** Two retries across up to three nodes. For HIGH priority requests. */
	public static RetryPolicy aggressive() {
		return new RetryPolicy(3, 100);
	}

	/** Custom policy. */
	public static RetryPolicy of(int maxAttempts, long backoffMs) {
		return new RetryPolicy(maxAttempts, backoffMs);
	}

	/** Whether any retry is configured (maxAttempts > 1). */
	public boolean hasRetry() {
		return maxAttempts > 1;
	}
}
