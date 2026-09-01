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
 * Output from a batched multi-request decode forward pass.
 *
 * <ul>
 *   <li><b>Final node</b>: {@code logits} is {@code float[batchSize][vocabSize]};
 *       {@code activations} is null.</li>
 *   <li><b>Intermediate node</b>: {@code activations} is flattened
 *       {@code batchSize × hiddenDim}; {@code logits} is null.</li>
 * </ul>
 */
public record MultiDecodeForwardResult(
		float[][] logits,
		float[] activations,
		int batchSize,
		long computeNanos) {

	public boolean isFinalNode() {
		return logits != null;
	}
}
