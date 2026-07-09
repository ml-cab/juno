/*
 * Created by Yevhen Soldatov
 * Initial implementation: 2026
 *
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

import java.util.Optional;

/**
 * Executes the transformer forward pass for this node's assigned layers.
 *
 * Implementations: CyclicForwardPassHandler — deterministic fake, used in tests
 * + integration tests LlamaTransformerHandler
 *
 * Thread-safe — may be called concurrently for different requests in a batch.
 */
public interface ForwardPassHandler {

	/**
	 * Execute this node's forward pass.
	 *
	 * @param request input (token IDs for first node, activations for others)
	 * @param context this node's shard assignment and model metadata
	 * @return ForwardResult with activations (intermediate) or logits (last node)
	 */
	ForwardResult forward(ForwardRequest request, ShardContext context);

	/** Whether this handler is ready to serve (shard loaded, GPU initialized). */
	boolean isReady();

	/**
	 * Frees GPU-resident weight buffers ({@link DeviceHalfMatrix}, {@link DeviceFloatMatrix}, …)
	 * held by this handler. Safe to call multiple times; default implementation is a no-op.
	 *
	 * <p>Call before discarding a handler (e.g. shard unload or reload) so VRAM is returned
	 * promptly instead of waiting for GC/finalizers.
	 */
	default void releaseGpuResources() {
		// no-op — LlamaTransformerHandler, Phi3TransformerHandler, etc. override when needed
	}

	/**
	 * RMS-normalized final hidden state at the current position, immediately before
	 * the LM head. Only the shard that owns the output projection returns a value;
	 * intermediate shards return empty.
	 *
	 * <p>
	 * Runs the same layer stack as {@link #forward} for this position (including KV
	 * updates). Callers must not invoke both for the same position unless intentional.
	 */
	default Optional<float[]> lastRmsHiddenForEmbedding(ForwardRequest request, ShardContext context) {
		return Optional.empty();
	}

	/**
	 * Execute a batched forward pass over a contiguous window of new prompt tokens.
	 *
	 * <p>This is the prefill batching entry point: instead of calling
	 * {@link #forward} once per token (which re-traverses all weight matrices for
	 * each token independently), this method processes the entire window in a single
	 * call, allowing implementations to issue one GEMM per weight matrix per layer
	 * rather than {@code windowSize} GEMVs.
	 *
	 * <p><b>Correctness-preserving default</b>: loops {@code windowSize} times
	 * through the existing single-token {@link #forward} path, reusing today's
	 * exact code. Any handler that does not override this keeps working, just
	 * without the speedup — mirrors the existing
	 * {@link cab.ml.juno.node.InferencePipeline#forwardBatch} pattern (serial
	 * default, real implementations override).
	 *
	 * <p>For intermediate nodes the result carries all window activations flattened
	 * as {@code float[windowSize * hiddenDim]}. For the final node only the
	 * last-position logits are returned ({@code float[vocabSize]}); every
	 * intermediate logit is discarded, matching the prefill contract.
	 *
	 * @param request carries the new token IDs (first node) or flattened
	 *                activations from the previous node (subsequent nodes)
	 * @param context this node's shard assignment and model metadata
	 * @return per-window activations (intermediate) or last-position logits (final)
	 */
	default BatchForwardResult forwardBatch(BatchForwardRequest request, ShardContext context) {
		int W = request.windowSize();
		int H = context.hiddenDim();
		long totalNanos = 0;
		float[][] allActivations = context.hasOutputProjection() ? null : new float[W][];

		for (int b = 0; b < W; b++) {
			ForwardRequest singleReq;
			if (request.isFirstNode()) {
				singleReq = ForwardRequest.withTokens(request.requestId(),
						new int[]{ request.tokenIds()[b] }, request.startPosition() + b);
			} else {
				float[] row = new float[H];
				System.arraycopy(request.activations(), b * H, row, 0, H);
				singleReq = ForwardRequest.withActivations(request.requestId(), row,
						request.startPosition() + b);
			}
			ForwardResult res = forward(singleReq, context);
			totalNanos += res.computeNanos();

			if (res.isFinalNode()) {
				// Only the last-position logits are needed for prefill
				if (b == W - 1) {
					return new BatchForwardResult(request.requestId(), null, res.logits(), W, totalNanos);
				}
				// Earlier positions: logits discarded, continue loop
			} else {
				if (allActivations != null) {
					allActivations[b] = res.activations();
				}
			}
		}

		// Intermediate node: flatten all window activations
		float[] flat = new float[W * H];
		for (int b = 0; b < W; b++) {
			System.arraycopy(allActivations[b], 0, flat, b * H, H);
		}
		return new BatchForwardResult(request.requestId(), flat, null, W, totalNanos);
	}
}