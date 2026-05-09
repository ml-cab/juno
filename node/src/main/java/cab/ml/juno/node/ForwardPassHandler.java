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
 * + integration tests LlamaTransformerHandler — Cuda/org.bytedeco cublas
 * implementation (GPU required)
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
}