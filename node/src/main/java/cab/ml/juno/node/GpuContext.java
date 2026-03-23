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

import java.util.logging.Logger;

import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.runtime.JCuda;

/**
 * cuBLAS context: device selection and handle lifecycle.
 *
 * One GpuContext per node JVM. Created once at startup, destroyed at shutdown.
 * The cublasHandle it owns is shared across all CudaMatVecBackend calls on that
 * node — cuBLAS handles are thread-safe for concurrent kernel launches.
 *
 * Usage: try (GpuContext ctx = GpuContext.init(0)) { CudaMatVecBackend matVec =
 * new CudaMatVecBackend(ctx); LlamaTransformerHandler handler =
 * LlamaTransformerHandler.load(path, shard, matVec); ... }
 *
 * Throws IllegalStateException if CUDA is not available. Use
 * CudaAvailability.isAvailable() to guard the call site.
 */
public final class GpuContext implements AutoCloseable {

	private static final Logger log = Logger.getLogger(GpuContext.class.getName());

	private final int deviceIndex;
	private final cublasHandle handle;
	private volatile boolean closed = false;

	private GpuContext(int deviceIndex, cublasHandle handle) {
		this.deviceIndex = deviceIndex;
		this.handle = handle;
	}

	/**
	 * Initialise CUDA device {@code deviceIndex} and create a cuBLAS handle.
	 *
	 * @param deviceIndex 0-based GPU index (0 for single-GPU nodes)
	 * @return a ready GpuContext — caller must close() when done
	 * @throws IllegalStateException if CUDA is not available or init fails
	 */
	public static GpuContext init(int deviceIndex) {
		if (!CudaAvailability.isAvailable()) {
			throw new IllegalStateException("CUDA not available — cannot create GpuContext on this node");
		}

		JCuda.setExceptionsEnabled(true);
		JCublas2.setExceptionsEnabled(true);

		JCuda.cudaSetDevice(deviceIndex);

		cublasHandle handle = new cublasHandle();
		JCublas2.cublasCreate(handle);

		String name = CudaAvailability.deviceName(deviceIndex);
		long vram = CudaAvailability.vramBytes(deviceIndex);
		log.info(String.format("GpuContext ready — device %d: %s, %.1f GB VRAM", deviceIndex, name, vram / 1e9));

		return new GpuContext(deviceIndex, handle);
	}

	/** The cuBLAS handle — valid until close(). */
	public cublasHandle handle() {
		if (closed)
			throw new IllegalStateException("GpuContext already closed");
		return handle;
	}

	/** The device index this context is bound to. */
	public int deviceIndex() {
		return deviceIndex;
	}

	/** Whether this context has been closed. */
	public boolean isClosed() {
		return closed;
	}

	/** Destroy the cuBLAS handle and release device resources. */
	@Override
	public void close() {
		if (!closed) {
			closed = true;
			try {
				JCublas2.cublasDestroy(handle);
				log.info("GpuContext closed — device " + deviceIndex);
			} catch (Exception e) {
				log.warning("Error closing GpuContext: " + e.getMessage());
			}
		}
	}
}