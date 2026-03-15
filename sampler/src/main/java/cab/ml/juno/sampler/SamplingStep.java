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
package cab.ml.juno.sampler;

/**
 * A single step in the sampling pipeline. Stateless and thread-safe — takes
 * logits in, returns modified logits. Steps are chained: temperature → topK →
 * topP → softmax → penalty → sample
 */
@FunctionalInterface
public interface SamplingStep {

	/**
	 * Apply this step to the logits array.
	 *
	 * @param logits          raw or partially processed logit scores (modified in
	 *                        place)
	 * @param params          sampling configuration for this request
	 * @param generatedTokens tokens already generated in this sequence (for
	 *                        repetition penalty)
	 * @return the same logits array, modified in place
	 */
	float[] apply(float[] logits, SamplingParams params, int[] generatedTokens);
}
