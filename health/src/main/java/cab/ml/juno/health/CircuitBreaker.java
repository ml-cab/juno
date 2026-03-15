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

package cab.ml.juno.health;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Per-node sliding window circuit breaker.
 *
 * Matches Resilience4j config from the architecture doc: failureRateThreshold =
 * 50% of last N calls slidingWindowSize = 10 calls waitDurationInOpen = 30s
 *
 * Thread-safe — single ReentrantLock guards state transitions.
 */
public final class CircuitBreaker {

	private final String nodeId;
	private final int slidingWindowSize;
	private final double failureRateThreshold; // 0.0–1.0
	private final Duration waitDurationInOpen;

	private final Deque<Boolean> window = new ArrayDeque<>(); // true = success
	private final ReentrantLock lock = new ReentrantLock();

	private CircuitState state = CircuitState.CLOSED;
	private Instant openedAt = null;

	public CircuitBreaker(String nodeId, int slidingWindowSize, double failureRateThreshold,
			Duration waitDurationInOpen) {
		if (nodeId == null || nodeId.isBlank())
			throw new IllegalArgumentException("nodeId must not be blank");
		if (slidingWindowSize < 1)
			throw new IllegalArgumentException("slidingWindowSize must be >= 1");
		if (failureRateThreshold <= 0.0 || failureRateThreshold >= 1.0)
			throw new IllegalArgumentException("failureRateThreshold must be in (0.0, 1.0)");

		this.nodeId = nodeId;
		this.slidingWindowSize = slidingWindowSize;
		this.failureRateThreshold = failureRateThreshold;
		this.waitDurationInOpen = waitDurationInOpen;
	}

	/** Default config matching architecture spec. */
	public static CircuitBreaker forNode(String nodeId) {
		return new CircuitBreaker(nodeId, 10, 0.50, Duration.ofSeconds(30));
	}

	/**
	 * Whether calls to this node are currently permitted. OPEN state rejects calls;
	 * CLOSED and HALF_OPEN allow them.
	 */
	public boolean isCallPermitted() {
		lock.lock();
		try {
			return switch (state) {
			case CLOSED -> true;
			case HALF_OPEN -> true;
			case OPEN -> {
				if (waitElapsed()) {
					transitionTo(CircuitState.HALF_OPEN);
					yield true;
				}
				yield false;
			}
			};
		} finally {
			lock.unlock();
		}
	}

	/** Record a successful call. */
	public void recordSuccess() {
		lock.lock();
		try {
			if (state == CircuitState.HALF_OPEN) {
				window.clear(); // fresh start after recovery
				transitionTo(CircuitState.CLOSED);
			}
			addToWindow(true);
			evaluateWindow(); // no-op on empty window
		} finally {
			lock.unlock();
		}
	}

	/** Record a failed call. */
	public void recordFailure() {
		lock.lock();
		try {
			addToWindow(false);
			if (state == CircuitState.HALF_OPEN) {
				transitionTo(CircuitState.OPEN);
				return;
			}
			evaluateWindow();
		} finally {
			lock.unlock();
		}
	}

	/** Force-open the circuit (called on VRAM_CRITICAL or node removal). */
	public void forceOpen() {
		lock.lock();
		try {
			transitionTo(CircuitState.OPEN);
		} finally {
			lock.unlock();
		}
	}

	/** Force-close the circuit (called when node recovers). */
	public void reset() {
		lock.lock();
		try {
			window.clear();
			transitionTo(CircuitState.CLOSED);
		} finally {
			lock.unlock();
		}
	}

	public CircuitState state() {
		return state;
	}

	public String nodeId() {
		return nodeId;
	}

	// ── Private ───────────────────────────────────────────────────────────────

	private void addToWindow(boolean success) {
		if (window.size() >= slidingWindowSize)
			window.pollFirst();
		window.addLast(success);
	}

	private void evaluateWindow() {
		if (window.size() < slidingWindowSize)
			return; // not enough data yet
		long failures = window.stream().filter(r -> !r).count();
		double failureRate = (double) failures / window.size();
		if (failureRate >= failureRateThreshold) {
			transitionTo(CircuitState.OPEN);
		}
	}

	private void transitionTo(CircuitState next) {
		state = next;
		if (next == CircuitState.OPEN)
			openedAt = Instant.now();
	}

	private boolean waitElapsed() {
		return openedAt != null && Duration.between(openedAt, Instant.now()).compareTo(waitDurationInOpen) >= 0;
	}
}
