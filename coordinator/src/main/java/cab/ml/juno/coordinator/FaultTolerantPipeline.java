package cab.ml.juno.coordinator;

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

import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;

import cab.ml.juno.health.CircuitBreaker;
import cab.ml.juno.node.InferencePipeline;

/**
 * Fault-tolerant InferencePipeline that wraps one or more pipeline replicas,
 * each guarded by its own CircuitBreaker.
 *
 * ── Routing
 * ─────────────────────────────────────────────────────────────────── forward()
 * scans the node list in order, skipping OPEN circuits. The first permitted
 * node that succeeds wins. On failure it records the failure, waits backoffMs
 * (if configured), then tries the next permitted node.
 *
 * forwardBatch() uses the same policy — routes the entire batch to the first
 * permitted node that can handle it, falling back on failure.
 *
 * ── Circuit Breaker Integration
 * ─────────────────────────────────────────────── Each NodePipeline carries its
 * own CircuitBreaker. Transitions happen here: forward pass succeeds →
 * recordSuccess() forward pass throws → recordFailure() (may trip the circuit)
 * VRAM_CRITICAL event → forceOpen() immediately NODE_STALE event → forceOpen()
 * immediately NODE_RECOVERED event → reset() — return to CLOSED
 *
 * ── Health Event Hooks
 * ──────────────────────────────────────────────────────── HealthReactor calls
 * these after evaluating HealthEvents from the probe: onVramCritical(nodeId) —
 * force-open that node's circuit onNodeStale(nodeId) — force-open that node's
 * circuit onNodeRecovered(nodeId) — reset that node's circuit to CLOSED
 *
 * ── Thread Safety
 * ───────────────────────────────────────────────────────────── CircuitBreaker
 * is already thread-safe (ReentrantLock inside). The node list is immutable
 * after construction. The nodeIndex map is ConcurrentHashMap — safe for
 * concurrent reads.
 */
public final class FaultTolerantPipeline implements InferencePipeline {

	private static final Logger log = Logger.getLogger(FaultTolerantPipeline.class.getName());

	/**
	 * One pipeline replica + its circuit breaker. In production: pipeline = gRPC
	 * stub to a real node JVM. In tests: pipeline = StubInferencePipeline or a
	 * throwing mock.
	 */
	public record NodePipeline(String nodeId, InferencePipeline pipeline, CircuitBreaker circuitBreaker) {
		public NodePipeline {
			if (nodeId == null || nodeId.isBlank())
				throw new IllegalArgumentException("nodeId must not be blank");
			if (pipeline == null)
				throw new IllegalArgumentException("pipeline must not be null");
			if (circuitBreaker == null)
				throw new IllegalArgumentException("circuitBreaker must not be null");
		}

		/**
		 * Factory: creates a NodePipeline with a default CircuitBreaker for the node.
		 */
		public static NodePipeline of(String nodeId, InferencePipeline pipeline) {
			return new NodePipeline(nodeId, pipeline, CircuitBreaker.forNode(nodeId));
		}
	}

	private final List<NodePipeline> nodes;
	private final ConcurrentHashMap<String, NodePipeline> nodeIndex; // nodeId → node
	private final RetryPolicy retryPolicy;

	public FaultTolerantPipeline(List<NodePipeline> nodes, RetryPolicy retryPolicy) {
		if (nodes == null || nodes.isEmpty())
			throw new IllegalArgumentException("nodes must not be empty");
		if (retryPolicy == null)
			throw new IllegalArgumentException("retryPolicy must not be null");

		this.nodes = List.copyOf(nodes);
		this.retryPolicy = retryPolicy;
		this.nodeIndex = new ConcurrentHashMap<>();
		for (NodePipeline n : nodes)
			nodeIndex.put(n.nodeId(), n);
	}

	/** Convenience constructor: single-node with default retry. */
	public static FaultTolerantPipeline wrapping(String nodeId, InferencePipeline pipeline) {
		return new FaultTolerantPipeline(List.of(NodePipeline.of(nodeId, pipeline)), RetryPolicy.once());
	}

	// ── InferencePipeline impl ────────────────────────────────────────────────

	/**
	 * Execute a single-request forward pass with circuit-breaker + retry.
	 *
	 * Tries nodes in order, skipping OPEN circuits. Retries up to
	 * retryPolicy.maxAttempts() times across different nodes. Throws
	 * PipelineUnavailableException if all attempts fail or all circuits are OPEN.
	 */
	@Override
	public float[] forward(String requestId, int[] tokens, int startPos) {
		int tried = 0;
		Exception lastCause = null;

		for (NodePipeline node : nodes) {
			if (tried >= retryPolicy.maxAttempts())
				break;
			if (!node.circuitBreaker().isCallPermitted())
				continue;

			tried++;
			try {
				float[] result = node.pipeline().forward(requestId, tokens, startPos);
				node.circuitBreaker().recordSuccess();
				return result;
			} catch (Exception e) {
				log.warning(
						"forward() failed on node " + node.nodeId() + " (attempt " + tried + "): " + e.getMessage());
				node.circuitBreaker().recordFailure();
				lastCause = e;
				sleepBackoff();
			}
		}

		if (tried == 0) {
			throw new PipelineUnavailableException(PipelineUnavailableException.Reason.CIRCUIT_OPEN, 0,
					"all " + nodes.size() + " node circuit(s) are OPEN");
		}
		throw new PipelineUnavailableException(PipelineUnavailableException.Reason.RETRIES_EXHAUSTED, tried,
				"all " + tried + " attempt(s) failed", lastCause);
	}

	/**
	 * Execute a batched forward pass with circuit-breaker + retry.
	 *
	 * Routes the entire batch to the first permitted node. On failure the batch is
	 * retried on the next permitted node. This keeps all requests in a batch on the
	 * same node for consistent KV cache behaviour.
	 */
	@Override
	public float[][] forwardBatch(List<String> requestIds, List<int[]> allTokens, List<Integer> startPositions) {
		int tried = 0;
		Exception lastCause = null;

		for (NodePipeline node : nodes) {
			if (tried >= retryPolicy.maxAttempts())
				break;
			if (!node.circuitBreaker().isCallPermitted())
				continue;

			tried++;
			try {
				float[][] result = node.pipeline().forwardBatch(requestIds, allTokens, startPositions);
				node.circuitBreaker().recordSuccess();
				return result;
			} catch (Exception e) {
				log.warning("forwardBatch() failed on node " + node.nodeId() + " (attempt " + tried + "): "
						+ e.getMessage());
				node.circuitBreaker().recordFailure();
				lastCause = e;
				sleepBackoff();
			}
		}

		if (tried == 0) {
			throw new PipelineUnavailableException(PipelineUnavailableException.Reason.CIRCUIT_OPEN, 0,
					"all " + nodes.size() + " node circuit(s) are OPEN");
		}
		throw new PipelineUnavailableException(PipelineUnavailableException.Reason.RETRIES_EXHAUSTED, tried,
				"all " + tried + " batch attempt(s) failed", lastCause);
	}

	/**
	 * Vocab size delegated to the first node. All nodes in a cluster serve the same
	 * model — vocab size is identical.
	 */
	@Override
	public int vocabSize() {
		return nodes.get(0).pipeline().vocabSize();
	}

	// ── Health event hooks ────────────────────────────────────────────────────

	/**
	 * Force-open this node's circuit immediately. Called by HealthReactor on
	 * VRAM_CRITICAL events. No-op if the nodeId is not in this pipeline's node
	 * list.
	 */
	public void onVramCritical(String nodeId) {
		NodePipeline node = nodeIndex.get(nodeId);
		if (node != null) {
			log.warning("VRAM_CRITICAL on " + nodeId + " — circuit force-opened");
			node.circuitBreaker().forceOpen();
		}
	}

	/**
	 * Force-open this node's circuit immediately. Called by HealthReactor on
	 * NODE_STALE events (missed health probes). No-op if the nodeId is not in this
	 * pipeline's node list.
	 */
	public void onNodeStale(String nodeId) {
		NodePipeline node = nodeIndex.get(nodeId);
		if (node != null) {
			log.warning("NODE_STALE: " + nodeId + " — circuit force-opened");
			node.circuitBreaker().forceOpen();
		}
	}

	/**
	 * Reset this node's circuit to CLOSED. Called by HealthReactor on
	 * NODE_RECOVERED events. No-op if the nodeId is not in this pipeline's node
	 * list.
	 */
	public void onNodeRecovered(String nodeId) {
		NodePipeline node = nodeIndex.get(nodeId);
		if (node != null) {
			log.info("NODE_RECOVERED: " + nodeId + " — circuit reset to CLOSED");
			node.circuitBreaker().reset();
		}
	}

	/** Whether all node circuits are currently OPEN (cluster fully unavailable). */
	public boolean isFullyUnavailable() {
		return nodes.stream().noneMatch(n -> n.circuitBreaker().isCallPermitted());
	}

	/** Number of nodes with CLOSED or HALF_OPEN circuits (accepting calls). */
	public long availableNodeCount() {
		return nodes.stream().filter(n -> n.circuitBreaker().isCallPermitted()).count();
	}

	// ── Private ───────────────────────────────────────────────────────────────

	private void sleepBackoff() {
		if (retryPolicy.backoffMs() <= 0)
			return;
		try {
			Thread.sleep(retryPolicy.backoffMs());
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}
	}
}
