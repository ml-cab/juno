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

import jdk.jfr.Category;
import jdk.jfr.Description;
import jdk.jfr.Event;
import jdk.jfr.Label;
import jdk.jfr.Name;
import jdk.jfr.StackTrace;

/**
 * JFR event emitted once per optimizer update (one accumulation group).
 *
 * <p>
 * Captures a timing breakdown of forward, backward, and Adam plus accumulation
 * and clipping metadata so profilers can correlate updates with chunk counts.
 *
 * @see LoraTrainableHandler#trainStep
 * @see LoraTrainableHandler#computeGradients
 */
@Name("juno.LoraTrainStep")
@Label("LoRA Train Step")
@Description("One optimizer update in LoRA fine-tuning: forward, backward, Adam")
@Category({ "Juno", "LoRA" })
@StackTrace(false)
public final class LoraTrainEvent extends Event {

	@Label("Step")
	@Description("Optimizer step index (1-based)")
	public int step;

	@Label("Num Tokens")
	@Description("Sum of input token counts across chunks in this update")
	public int numTokens;

	@Label("Chunk Count")
	@Description("Number of sequence chunks accumulated into this update")
	public int chunkCount;

	@Label("Prediction Count")
	@Description("Total prediction tokens used for gradient normalization")
	public int predictionCount;

	@Label("Loss")
	@Description("Token-weighted mean cross-entropy loss (nats) for this update")
	public float loss;

	@Label("Global Grad Norm")
	@Description("L2 norm of adapter gradients after prediction-count normalization, before clipping")
	public float globalGradNorm;

	@Label("Clip Scale")
	@Description("Multiplicative scale applied to gradients (includes 1/N and optional clip)")
	public float clipScale;

	@Label("Clipped")
	@Description("Whether global-norm clipping was applied")
	public boolean clipped;

	@Label("Forward ms")
	@Description("Wall time for forward passes across all chunks (ms)")
	public long forwardMs;

	@Label("Backward ms")
	@Description("Wall time for backward passes across all chunks (ms)")
	public long backwardMs;

	@Label("Optimizer ms")
	@Description("Wall time for the Adam parameter update (ms)")
	public long optimizerMs;

	@Label("Total ms")
	@Description("Total wall time for this optimizer update (ms)")
	public long totalMs;

	@Label("Learning Rate A")
	@Description("Scheduled learning rate applied to LoRA A matrices")
	public float learningRateA;

	@Label("Learning Rate B")
	@Description("Scheduled learning rate applied to LoRA B matrices (includes LoRA+ ratio)")
	public float learningRateB;

	@Label("LoRA Plus Ratio")
	@Description("B/A learning-rate ratio (1.0 = ordinary non-LoRA+ behavior)")
	public float loraPlusRatio;

	@Label("Dropout")
	@Description("Train-only LoRA dropout rate used for this update (0 = disabled)")
	public float dropout;
}
