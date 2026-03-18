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

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

/**
 * Priority-aware request scheduler with optional micro-batching.
 *
 * ── Batching disabled (BatchConfig.disabled() or 2-arg constructor) ──────────
 * Original behaviour: every request dispatched immediately on its own virtual
 * thread. submit() returns a future that completes when generation finishes.
 *
 * ── Batching enabled (BatchConfig with maxBatchSize > 1) ─────────────────────
 * A single background virtual thread runs the batch-collect loop:
 *
 * loop: 1. Block on queue.poll(batchWindowMs) — wait for first request 2. Drain
 * up to (maxBatchSize - 1) more with non-blocking poll() 3. Dispatch the batch
 * on a new virtual thread via generateBatch() 4. Each result completes the
 * corresponding CompletableFuture 5. Immediately resume collecting the next
 * batch
 *
 * The batch dispatch thread runs concurrently with collection, so the collector
 * never blocks waiting for GPU work to finish.
 *
 * ── Streaming
 * ───────────────────────────────────────────────────────────────── Both paths
 * deliver tokens via TokenConsumer. Batching is transparent.
 *
 * ── Thread safety
 * ─────────────────────────────────────────────────────────────
 * PriorityBlockingQueue: producer/consumer coordination ConcurrentHashMap:
 * request-id → inflight entry mapping volatile running: clean shutdown signal
 */
public final class RequestScheduler {

	private static final Logger log = Logger.getLogger(RequestScheduler.class.getName());

	private final int maxQueueDepth;
	private final PriorityBlockingQueue<InferenceRequest> queue;
	private final GenerationLoop generationLoop;
	private final BatchConfig batchConfig;
	private final ConcurrentHashMap<String, InflightEntry> inflight = new ConcurrentHashMap<>();

	private volatile boolean running = true;

	// ── Constructors ──────────────────────────────────────────────────────────

	/**
	 * Backward-compatible 2-arg constructor — batching disabled. All existing tests
	 * and callers compile and pass without change.
	 */
	public RequestScheduler(int maxQueueDepth, GenerationLoop generationLoop) {
		this(maxQueueDepth, generationLoop, BatchConfig.disabled());
	}

	/**
	 * Full constructor with explicit batching config. Starts the background collect
	 * loop when batching is enabled.
	 */
	public RequestScheduler(int maxQueueDepth, GenerationLoop generationLoop, BatchConfig batchConfig) {
		if (maxQueueDepth < 1)
			throw new IllegalArgumentException("maxQueueDepth must be >= 1");
		if (generationLoop == null)
			throw new IllegalArgumentException("generationLoop must not be null");
		if (batchConfig == null)
			throw new IllegalArgumentException("batchConfig must not be null");

		this.maxQueueDepth = maxQueueDepth;
		this.generationLoop = generationLoop;
		this.batchConfig = batchConfig;
		this.queue = new PriorityBlockingQueue<>(maxQueueDepth);

		if (batchConfig.isBatchingEnabled()) {
			startBatchDispatchLoop();
		}
	}

	// ── Public API ────────────────────────────────────────────────────────────

	/**
	 * Submit a request for async execution. Returns immediately. Tokens delivered
	 * via consumer; future completes on generation finish.
	 *
	 * @throws QueueFullException if queue has reached maxQueueDepth
	 */
	public CompletableFuture<GenerationResult> submit(InferenceRequest request, TokenConsumer consumer) {
		if (queue.size() >= maxQueueDepth) {
			throw new QueueFullException("Request queue full (" + maxQueueDepth + "). Retry later.",
					estimateRetryAfterSeconds());
		}

		CompletableFuture<GenerationResult> future = new CompletableFuture<>();
		// Register to inflight BEFORE queue.offer() — no lost-wakeup risk
		inflight.put(request.requestId(), new InflightEntry(request, consumer, future));
		queue.offer(request);

		if (!batchConfig.isBatchingEnabled()) {
			dispatchSingle(request, consumer, future);
		}
		// Batching enabled: background loop picks it up from the queue

		return future;
	}

	/**
	 * Submit and block until generation completes (non-streaming use case).
	 *
	 * @throws QueueFullException if queue has reached maxQueueDepth
	 */
	public GenerationResult submitAndWait(InferenceRequest request) {
		return submit(request, TokenConsumer.discard()).join();
	}

	/** Stop the batch dispatch loop. No-op if batching was never enabled. */
	public void shutdown() {
		running = false;
	}

	public int queueDepth() {
		return queue.size();
	}

	public int maxQueueDepth() {
		return maxQueueDepth;
	}

	// ── Batch dispatch loop ───────────────────────────────────────────────────

	private void startBatchDispatchLoop() {
		Thread.ofVirtual().name("batch-collector").start(() -> {
			while (running) {
				try {
					List<InferenceRequest> batch = collectBatch();
					if (batch.isEmpty())
						continue;
					dispatchBatch(batch);
				} catch (InterruptedException e) {
					Thread.currentThread().interrupt();
					break;
				} catch (Exception e) {
					log.warning("Batch dispatch error: " + e.getMessage());
				}
			}
			log.fine("Batch dispatch loop stopped");
		});
	}

	/**
	 * Collect up to maxBatchSize requests. Blocks up to batchWindowMs for the
	 * first, then drains the rest immediately.
	 */
	private List<InferenceRequest> collectBatch() throws InterruptedException {
		List<InferenceRequest> batch = new ArrayList<>(batchConfig.maxBatchSize());

		long windowMs = batchConfig.batchWindowMs();
		InferenceRequest first = (windowMs > 0) ? queue.poll(windowMs, TimeUnit.MILLISECONDS) : queue.poll();

		if (first == null)
			return batch;
		batch.add(first);

		while (batch.size() < batchConfig.maxBatchSize()) {
			InferenceRequest next = queue.poll();
			if (next == null)
				break;
			batch.add(next);
		}

		return batch;
	}

	/**
	 * Dispatch a collected batch on a new virtual thread so the collector loop can
	 * resume immediately.
	 */
	private void dispatchBatch(List<InferenceRequest> requests) {
		Thread.ofVirtual().name("batch-gen-" + requests.get(0).requestId()).start(() -> {
			List<BatchEntry> entries = new ArrayList<>(requests.size());
			for (InferenceRequest req : requests) {
				InflightEntry e = inflight.get(req.requestId());
				if (e != null)
					entries.add(new BatchEntry(e.request(), e.consumer()));
			}
			if (entries.isEmpty())
				return;

			try {
				List<GenerationResult> results = generationLoop.generateBatch(entries);
				for (int i = 0; i < entries.size(); i++) {
					String id = entries.get(i).request().requestId();
					InflightEntry inflt = inflight.remove(id);
					if (inflt != null)
						inflt.future().complete(results.get(i));
				}
			} catch (Exception e) {
				log.warning("Batch generation failed: " + e.getMessage());
				for (BatchEntry entry : entries) {
					InflightEntry inflt = inflight.remove(entry.request().requestId());
					if (inflt != null)
						inflt.future().completeExceptionally(e);
				}
			}
		});
	}

	// ── Single-request dispatch (batching disabled) ───────────────────────────

	private void dispatchSingle(InferenceRequest request, TokenConsumer consumer,
			CompletableFuture<GenerationResult> future) {
		Thread.ofVirtual().name("gen-" + request.requestId()).start(() -> {
			try {
				GenerationResult result = generationLoop.generate(request, consumer);
				future.complete(result);
			} catch (Exception e) {
				log.warning("Generation failed for " + request.requestId() + ": " + e.getMessage());
				future.completeExceptionally(e);
			} finally {
				queue.remove(request);
				inflight.remove(request.requestId());
			}
		});
	}

	private int estimateRetryAfterSeconds() {
		return Math.max(1, queue.size() * 2);
	}

	// ── Inner types ───────────────────────────────────────────────────────────

	private record InflightEntry(InferenceRequest request, TokenConsumer consumer,
			CompletableFuture<GenerationResult> future) {
	}

	public static final class QueueFullException extends RuntimeException {
		private static final long serialVersionUID = QueueFullException.class.getName().hashCode();
		private final int retryAfterSeconds;

		public QueueFullException(String message, int retryAfterSeconds) {
			super(message);
			this.retryAfterSeconds = retryAfterSeconds;
		}

		public int retryAfterSeconds() {
			return retryAfterSeconds;
		}
	}
}