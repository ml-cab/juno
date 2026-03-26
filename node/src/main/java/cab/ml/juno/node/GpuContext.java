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

import org.bytedeco.cuda.cublas.cublasContext;
import org.bytedeco.cuda.global.cublas;
import org.bytedeco.cuda.global.cudart;

/**
 * cuBLAS context: device selection and handle lifecycle.
 *
 * One GpuContext per node JVM. Created once at startup, destroyed at shutdown.
 * The cublasContext it owns is shared across all CublasMatVec calls on that node
 * — cuBLAS handles are thread-safe for concurrent kernel launches.
 *
 * Uses org.bytedeco (JavaCPP) cuda/cublas.
 *
 * Usage:
 *   try (GpuContext ctx = GpuContext.init(0)) {
 *       GpuForwardPassHandler handler = GpuForwardPassHandler.loadGpuResident(path, shard, ctx);
 *       ...
 *       handler.releaseGpuResources();
 *   }
 *
 * Throws IllegalStateException if CUDA is not available.
 *   
 * @author Yevhen Soldatov
 * 
 */
public final class GpuContext implements AutoCloseable {

    private static final Logger log = Logger.getLogger(GpuContext.class.getName());

    private final int deviceIndex;
    private final cublasContext handle;
    private volatile boolean closed = false;

    private GpuContext(int deviceIndex, cublasContext handle) {
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
            throw new IllegalStateException(
                "CUDA not available — cannot create GpuContext on this node");
        }

        cudart.cudaSetDevice(deviceIndex);

        cublasContext handle = new cublasContext();
        int rc = cublas.cublasCreate_v2(handle);
        if (rc != 0) {
            throw new IllegalStateException("cublasCreate failed: " + rc);
        }

        String name = CudaAvailability.deviceName(deviceIndex);
        long vram = CudaAvailability.vramBytes(deviceIndex);
        log.info(String.format("GpuContext ready — device %d: %s, %.1f GB VRAM", deviceIndex, name, vram / 1e9));

        return new GpuContext(deviceIndex, handle);
    }

    /** The cuBLAS handle — valid until close(). */
    public cublasContext handle() {
        if (closed) throw new IllegalStateException("GpuContext already closed");
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
                cublas.cublasDestroy_v2(handle);
                log.info("GpuContext closed — device " + deviceIndex);
            } catch (Exception e) {
                log.warning("Error closing GpuContext: " + e.getMessage());
            }
        }
    }
}