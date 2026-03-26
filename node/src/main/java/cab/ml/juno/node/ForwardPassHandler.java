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
}