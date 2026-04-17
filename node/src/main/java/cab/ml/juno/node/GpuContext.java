/*
 * Created by Yevhen Soldatov
 * Initial implementation: 2026
 *
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

import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;

import org.bytedeco.cuda.cublas.cublasContext;
import org.bytedeco.cuda.global.cublas;
import org.bytedeco.cuda.global.cudart;

/**
 * cuBLAS context: device selection and handle lifecycle.
 *
 * Prefer {@link #shared(int)} for a process-wide handle on long-lived JVMs
 * (embedded node, {@code ForwardPassHandlerLoader} default GPU path). Use
 * {@link #init(int)} when tests or tools need an isolated handle they can
 * {@link #close()} without affecting other code.
 *
 * The cublasContext is shared across all {@link CudaMatVec} calls using the same
 * {@code GpuContext} — concurrent host launches still require external ordering:
 * {@link #cublasSerializationLock()} is used by {@link CudaMatVec} to serialize
 * stream binding and kernel execution per context.
 *
 * Uses org.bytedeco (JavaCPP) cuda/cublas.
 *
 * Usage:
 *   try (GpuContext ctx = GpuContext.init(0)) {
 *       ForwardPassHandler handler = ForwardPassHandlerLoader.load(path, shard, new CudaMatVec(ctx));
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

    /** One shared {@link GpuContext} per device index for the JVM lifetime. */
    private static final Map<Integer, GpuContext> SHARED_INSTANCES = new HashMap<>();
    private static final Object SHARED_LOCK = new Object();

    private final int deviceIndex;
    private final cublasContext handle;
    private final boolean processShared;
    private final Object cublasSerialization = new Object();
    private volatile boolean closed = false;

    private GpuContext(int deviceIndex, cublasContext handle, boolean processShared) {
        this.deviceIndex = deviceIndex;
        this.handle = handle;
        this.processShared = processShared;
    }

    /**
     * Mutex for {@code cublasSetStream} / memcpy / GEMV sequences on this handle.
     * cuBLAS associates a stream with the handle; concurrent threads must not interleave.
     */
    Object cublasSerializationLock() {
        return cublasSerialization;
    }

    /**
     * Process-wide singleton per CUDA device: one cuBLAS handle per {@code deviceIndex}.
     * {@link #close()} does nothing on these instances so embedded servers and
     * {@code selectBackend()} paths do not tear down CUDA under other live code.
     *
     * @param deviceIndex 0-based CUDA device (must be {@code < cudaGetDeviceCount()})
     */
    public static GpuContext shared(int deviceIndex) {
        if (deviceIndex < 0)
            throw new IllegalArgumentException("deviceIndex must be non-negative: " + deviceIndex);
        synchronized (SHARED_LOCK) {
            GpuContext existing = SHARED_INSTANCES.get(deviceIndex);
            if (existing != null && !existing.closed)
                return existing;
            GpuContext created = create(deviceIndex, true);
            SHARED_INSTANCES.put(deviceIndex, created);
            log.info("GpuContext.shared(" + deviceIndex + ") — installed process-wide GPU context");
            return created;
        }
    }

    /**
     * Initialise CUDA device {@code deviceIndex} and create a cuBLAS handle.
     *
     * @param deviceIndex 0-based GPU index (0 for single-GPU nodes)
     * @return a ready GpuContext — caller must close() when done
     * @throws IllegalStateException if CUDA is not available or init fails
     */
    public static GpuContext init(int deviceIndex) {
        return create(deviceIndex, false);
    }

    private static GpuContext create(int deviceIndex, boolean processShared) {
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
        log.info(String.format("GpuContext ready — device %d: %s, %.1f GB VRAM (shared=%b)",
                deviceIndex, name, vram / 1e9, processShared));

        return new GpuContext(deviceIndex, handle, processShared);
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

    /** Whether this is the JVM-wide singleton from {@link #shared(int)}. */
    public boolean isProcessShared() {
        return processShared;
    }

    /** Destroy the cuBLAS handle and release device resources. */
    @Override
    public void close() {
        if (processShared) {
            log.fine("GpuContext.close ignored — process-shared instance (device " + deviceIndex + ")");
            return;
        }
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
