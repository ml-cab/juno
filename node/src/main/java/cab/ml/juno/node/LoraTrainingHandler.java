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

import cab.ml.juno.lora.LoraAdapterSet;
import cab.ml.juno.lora.LoraTrainContext;

/**
 * Architecture-specific LoRA training and playback handler.
 *
 * <p>Extends {@link ForwardPassHandler} so the same instance serves
 * {@code --lora-play} inference and gradient computation.
 */
public interface LoraTrainingHandler extends ForwardPassHandler {

	/** Unnormalized summed gradients; every prediction position contributes. */
	LoraGradientResult computeGradients(int[] tokens);

	/** Completion-mask variant; {@code null} mask trains every position. */
	LoraGradientResult computeGradients(int[] tokens, boolean[] lossMask);

	/** Gradient computation with optional train-only dropout context. */
	LoraGradientResult computeGradients(int[] tokens, boolean[] lossMask, LoraTrainContext ctx);

	/** Forward-only loss evaluation (no gradient accumulation). */
	LoraGradientResult evaluateLoss(int[] tokens);

	LoraGradientResult evaluateLoss(int[] tokens, boolean[] lossMask);

	/** Attached adapter set (mutable during training). */
	LoraAdapterSet adapters();

	/** Model architecture label from GGUF ({@code general.architecture}). */
	String architecture();

	/** Layout bindings used for init/validate/merge. */
	LoraModelLayout layout();
}
