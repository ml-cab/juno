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

/**
 * A resident-weight matrix-vector product that runs out of device memory finishes
 * on the CPU from the quantized weights, which the handler keeps, instead of failing
 * the request. Said once per handler.
 *
 * <p>Usage at a product's call site:
 * <pre>{@code
 * try {
 *     return backend.sgemv(dev, x);
 * } catch (IllegalStateException ex) {
 *     fallback.absorb(ex);
 * }
 * return cpuProduct(quant, x);
 * }</pre>
 */
final class DeviceMatVecFallback {

	private static final Logger log = Logger.getLogger(DeviceMatVecFallback.class.getName());

	private final String handler;
	private final AtomicBoolean warned = new AtomicBoolean();

	DeviceMatVecFallback(String handler) {
		this.handler = handler;
	}

	/**
	 * Returns when {@code ex} is a device out-of-memory failure, after warning once;
	 * rethrows anything else.
	 */
	void absorb(IllegalStateException ex) {
		if (!GpuLayerOffload.isVramOom(ex))
			throw ex;
		if (warned.compareAndSet(false, true))
			log.warning(handler + ": out of device memory in a resident-weight matmul - that matmul runs on"
					+ " the CPU from the quantized weights. Lower --gpu-layers to keep it on the GPU.");
	}
}
