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

import java.io.IOException;
import java.nio.file.Path;
import java.util.Optional;
import java.util.logging.Logger;

import cab.ml.juno.lora.LoraAdapterSet;
import cab.ml.juno.lora.LoraTrainContext;

/**
 * Dense Qwen2 / Qwen2.5 LoRA training and playback.
 *
 * <p>Same SwiGLU / GQA / adjacent-pair RoPE structure as
 * {@link LoraTrainableHandler}, plus frozen QKV biases when present in the GGUF
 * ({@code blk.L.attn_q/k/v.bias}). Bias parameters are never trained.
 */
public final class Qwen2LoraTrainableHandler implements LoraTrainingHandler {

	private static final Logger log = Logger.getLogger(Qwen2LoraTrainableHandler.class.getName());

	private final LoraTrainableHandler delegate;

	private Qwen2LoraTrainableHandler(LoraTrainableHandler delegate) {
		this.delegate = delegate;
	}

	public static Qwen2LoraTrainableHandler load(Path modelPath, ShardContext context, LoraAdapterSet adapters)
			throws IOException {
		return load(modelPath, context, adapters, ForwardPassHandlerLoader.selectLoraBackend());
	}

	public static Qwen2LoraTrainableHandler load(Path modelPath, ShardContext context, LoraAdapterSet adapters,
			MatVec backend) throws IOException {
		log.info("Loading Qwen2 LoRA handler: " + modelPath);
		LoraTrainableHandler inner = LoraTrainableHandler.load(modelPath, context, adapters, backend);
		return new Qwen2LoraTrainableHandler(inner);
	}

	/** Package-visible for tests that already hold a {@link LoraTrainableHandler}. */
	static Qwen2LoraTrainableHandler wrap(LoraTrainableHandler handler) {
		return new Qwen2LoraTrainableHandler(handler);
	}

	LoraTrainableHandler delegate() {
		return delegate;
	}

	@Override
	public ForwardResult forward(ForwardRequest request, ShardContext context) {
		return delegate.forward(request, context);
	}

	@Override
	public boolean isReady() {
		return delegate.isReady();
	}

	@Override
	public void releaseGpuResources() {
		delegate.releaseGpuResources();
	}

	@Override
	public Optional<float[]> lastRmsHiddenForEmbedding(ForwardRequest request, ShardContext context) {
		return delegate.lastRmsHiddenForEmbedding(request, context);
	}

	@Override
	public LoraGradientResult computeGradients(int[] tokens) {
		return delegate.computeGradients(tokens);
	}

	@Override
	public LoraGradientResult computeGradients(int[] tokens, boolean[] lossMask) {
		return delegate.computeGradients(tokens, lossMask);
	}

	@Override
	public LoraGradientResult computeGradients(int[] tokens, boolean[] lossMask, LoraTrainContext ctx) {
		return delegate.computeGradients(tokens, lossMask, ctx);
	}

	@Override
	public LoraGradientResult evaluateLoss(int[] tokens) {
		return delegate.evaluateLoss(tokens);
	}

	@Override
	public LoraGradientResult evaluateLoss(int[] tokens, boolean[] lossMask) {
		return delegate.evaluateLoss(tokens, lossMask);
	}

	@Override
	public LoraAdapterSet adapters() {
		return delegate.adapters();
	}

	@Override
	public String architecture() {
		return delegate.architecture();
	}

	@Override
	public LoraModelLayout layout() {
		return LoraModelLayout.qwen2(delegate.config());
	}
}
