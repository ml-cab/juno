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
 * A request + its token-streaming callback, grouped for batch dispatch.
 *
 * Passed to GenerationLoop.generateBatch() — one entry per request in the
 * batch. The consumer receives tokens as they are generated, exactly as in
 * single-request generation. Batching is transparent to the streaming layer.
 */
public record BatchEntry(InferenceRequest request, TokenConsumer consumer) {

	public BatchEntry {
		if (request == null)
			throw new IllegalArgumentException("request must not be null");
		if (consumer == null)
			throw new IllegalArgumentException("consumer must not be null");
	}
}