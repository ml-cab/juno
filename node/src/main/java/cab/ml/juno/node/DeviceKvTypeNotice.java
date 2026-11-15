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
 * What a launch is told when {@code --cache-type-k/-v} selects a quantized KV ({@code q8_0}, {@code q4_0}) and
 * attention runs on the GPU. The GPU attention kernels read a device copy of each request's KV ({@link DeviceKvCache}),
 * which is stored in the selected types ({@link DeviceKvFormats#select}): the host codecs' blocks, so the layers
 * attending on the GPU read the values the host KV holds. Nothing is said then. The device copy stays FP16 for a side
 * whose rows it cannot store as blocks (a head width that is not a multiple of 32) or when the attention kernel for
 * quantized rows does not load; that launch is told, since the quantized type then sets the host rows only. Where
 * attention reads the host rows (the CPU backend, a GPU backend without the kernel, LoRA, the Phi-2 and Qwen3-MoE
 * handlers, {@code --gpu-attention off}) the type applies in full and nothing is said.
 *
 * <p>Console front ends print {@link #consoleNotice} (they turn library logging off unless verbose), which can see the
 * head width in the model's header; a handler that opens GPU attention logs {@link #announce} once per process for
 * every other entry point, and for a kernel that did not load.
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
	 * @param headDim      the model's head width, or 0 when it could not be read
	 * @param lora         LoRA training or {@code --lora-play}, whose attention runs on the CPU
	 * @param cpu          the launch uses the CPU backend
	 * @param cudaBackend  a GPU launch selects CUDA, the only backend with the attention kernel
	 * @param types        the launch's {@code --cache-type-k/-v}
	 */
	public static String consoleNotice(String architecture, int headDim, boolean lora, boolean cpu, boolean cudaBackend,
			CacheTypeOptions types) {
		if (!types.usesQuantized() || lora || cpu || !cudaBackend
				|| !GpuAttentionSupport.handlerRunsKernel(architecture)
				|| GpuAttentionOptions.fromEnv().mode() == GpuAttentionOptions.Mode.OFF)
			return null;
		if (headDim < 1
				|| (DeviceKvFormat.canHold(types.typeK(), headDim) && DeviceKvFormat.canHold(types.typeV(), headDim)))
			return null;
		return text(types, "its head width " + headDim + " is not a multiple of " + DeviceKvFormat.BLOCK);
	}

	/**
	 * Logs, once per process, a WARNING when a handler opening GPU attention keeps an FP16 device KV under quantized
	 * {@code types}, and an INFO line naming the device formats when it stores them.
	 */
	static void announce(String handler, CacheTypeOptions types, DeviceKvFormats formats, int kvDim, int headDim) {
		if (!types.usesQuantized() || !ANNOUNCED.compareAndSet(false, true))
			return;
		if (formats.holds(types)) {
			log.info(handler + ": GPU attention's KV copy stored as " + formats.summary() + ", "
					+ formats.bytesPerPosition(kvDim) + " bytes per position against "
					+ DeviceKvFormats.F16.bytesPerPosition(kvDim) + " for FP16");
			return;
		}
		boolean width = !DeviceKvFormat.canHold(types.typeK(), headDim)
				|| !DeviceKvFormat.canHold(types.typeV(), headDim);
		log.warning(handler + ": " + text(types, width
				? "its head width " + headDim + " is not a multiple of " + DeviceKvFormat.BLOCK
				: "the attention kernel for quantized rows did not load"));
	}

	static void resetForTest() {
		ANNOUNCED.set(false);
	}

	private static String text(CacheTypeOptions types, String why) {
		return types.policySummary() + " sets the host KV only: GPU attention keeps its own copy of each request's KV in"
				+ " FP16 because " + why + ", so layers attending on the GPU use unquantized keys and values and the"
				+ " device copy keeps its FP16 size; the host rows serve CPU attention and session save and restore";
	}
}
