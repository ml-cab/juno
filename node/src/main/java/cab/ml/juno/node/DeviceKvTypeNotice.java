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

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Logger;

import cab.ml.juno.kvcache.CacheTypeOptions;

/**
 * What a launch is told when {@code --cache-type-k/-v} selects a quantized host KV ({@code q8_0}, {@code q4_0}) and
 * attention runs on the GPU. The GPU attention kernels read a device copy of each request's KV ({@link DeviceKvCache})
 * that is always FP16, written from the rows the request computes, so the quantized type then sets the host rows only:
 * the layers that attend on the GPU read unquantized keys and values, and the device copy keeps its FP16 size. The host
 * rows still serve CPU attention (a layer that falls back to the CPU, the CPU-only handlers), session flush and
 * restore. Where attention reads the host rows (the CPU backend, a GPU backend without the kernel, LoRA, the Phi-2 and
 * Qwen3-MoE handlers, {@code --gpu-attention off}) the type applies in full and nothing is said.
 *
 * <p>Console front ends print {@link #consoleNotice} (they turn library logging off unless verbose); a handler that
 * opens GPU attention logs {@link #announce} once per process for every other entry point.
 */
public final class DeviceKvTypeNotice {

	private static final Logger log = Logger.getLogger(DeviceKvTypeNotice.class.getName());

	private static final AtomicBoolean ANNOUNCED = new AtomicBoolean();

	private DeviceKvTypeNotice() {
	}

	/**
	 * The one-line notice for this launch, or {@code null} when the cache types apply in full.
	 *
	 * @param architecture {@code general.architecture} of the model, or {@code null}
	 * @param lora         LoRA training or {@code --lora-play}, whose attention runs on the CPU
	 * @param cpu          the launch uses the CPU backend
	 * @param cudaBackend  a GPU launch selects CUDA, the only backend with the attention kernel
	 * @param types        the launch's {@code --cache-type-k/-v}
	 */
	public static String consoleNotice(String architecture, boolean lora, boolean cpu, boolean cudaBackend,
			CacheTypeOptions types) {
		if (!types.usesQuantized() || lora || cpu || !cudaBackend
				|| !GpuAttentionSupport.handlerRunsKernel(architecture)
				|| GpuAttentionOptions.fromEnv().mode() == GpuAttentionOptions.Mode.OFF)
			return null;
		return text(types);
	}

	/** Logs, once per process, that a handler opening GPU attention keeps the device KV in FP16 under {@code types}. */
	static void announce(String handler, CacheTypeOptions types) {
		if (types.usesQuantized() && ANNOUNCED.compareAndSet(false, true))
			log.warning(handler + ": " + text(types));
	}

	static void resetForTest() {
		ANNOUNCED.set(false);
	}

	private static String text(CacheTypeOptions types) {
		return types.policySummary() + " sets the host KV only: GPU attention reads its own FP16 copy of each request's"
				+ " KV, so layers attending on the GPU use unquantized keys and values and the device copy keeps its FP16"
				+ " size; the host rows serve CPU attention and session save and restore";
	}
}
