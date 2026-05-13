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

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;

import static java.lang.foreign.ValueLayout.ADDRESS;

/**
 * cuBLAS context: device selection and handle lifecycle, via Panama FFI.
 *
 * The cuBLAS handle is an opaque pointer ({@code cublasHandle_t}) stored as a
 * {@link MemorySegment}. It is created by {@code cublasCreate_v2} and destroyed
 * by {@code cublasDestroy_v2} — both resolved through {@link CudaBindings}.
 *
 * Prefer {@link #shared(int)} for a process-wide handle on long-lived JVMs
 * (embedded node, ForwardPassHandlerLoader GPU path). Use {@link #init(int)}
 * when tests or tools need an isolated handle they can {@link #close()} without
 * affecting other code.
 *
 * The cublasHandle_t is shared across all {@link CudaMatVec} calls on the same
 * GpuContext. {@link #cublasSerializationLock()} serializes stream binding and
 * kernel execution. Replace with ReentrantLock when addressing Loom pinning
 * (point 4 of the HPC audit).
 *
 * Usage:
 *   try (GpuContext ctx = GpuContext.init(0)) {
 *       ForwardPassHandler handler = ForwardPassHandlerLoader.load(path, shard, new CudaMatVec(ctx));
 *       handler.releaseGpuResources();
 *   }
 *
 * @author Yevhen Soldatov
 */
public final class GpuContext implements AutoCloseable {

    private static final Logger log = Logger.getLogger(GpuContext.class.getName());

    private static final Map<Integer, GpuContext> SHARED = new HashMap<>();
    private static final Object SHARED_LOCK = new Object();

    private final int           deviceIndex;
    /** Opaque cublasHandle_t returned by cublasCreate_v2. */
    private final MemorySegment handle;
    private final boolean       processShared;
    private final Object        cublasSerialization = new Object();
    private volatile boolean    closed = false;

    private GpuContext(int deviceIndex, MemorySegment handle, boolean processShared) {
        this.deviceIndex   = deviceIndex;
        this.handle        = handle;
        this.processShared = processShared;
    }

    /**
     * Mutex for cublasSetStream / memcpy / GEMV sequences on this handle.
     * cuBLAS associates a stream with the handle; concurrent threads must not
     * interleave these operations.
     *
     * Note: synchronized on this lock inside a virtual thread causes carrier
     * pinning when the body makes Panama downcalls. Migrate to ReentrantLock
     * when addressing Loom pinning (HPC audit point 4).
     */
    Object cublasSerializationLock() {
        return cublasSerialization;
    }

    /**
     * Process-wide singleton per CUDA device. {@link #close()} is a no-op on
     * these instances so embedded servers do not tear down CUDA under live code.
     *
     * @param deviceIndex 0-based CUDA device index
     */
    public static GpuContext shared(int deviceIndex) {
        if (deviceIndex < 0)
            throw new IllegalArgumentException("deviceIndex must be non-negative: " + deviceIndex);
        synchronized (SHARED_LOCK) {
            GpuContext existing = SHARED.get(deviceIndex);
            if (existing != null && !existing.closed) return existing;
            GpuContext created = create(deviceIndex, true);
            SHARED.put(deviceIndex, created);
            log.info("GpuContext.shared(" + deviceIndex + ") — process-wide GPU context installed");
            return created;
        }
    }

    /**
     * Initialises CUDA device deviceIndex and creates a cuBLAS handle.
     *
     * @param deviceIndex 0-based GPU index (0 for single-GPU nodes)
     * @return a ready GpuContext — caller must close() when done
     * @throws IllegalStateException if CUDA is not available or init fails
     */
    public static GpuContext init(int deviceIndex) {
        return create(deviceIndex, false);
    }

    private static GpuContext create(int deviceIndex, boolean processShared) {
        if (!CudaAvailability.isAvailable())
            throw new IllegalStateException("CUDA not available — cannot create GpuContext on this node");

        CudaBindings cuda = CudaBindings.instance();
        CudaBindings.check(CudaBindings.callInt(cuda.cudaSetDevice, deviceIndex), "cudaSetDevice");

        // cublasCreate_v2(cublasHandle_t *handle) — fills *handle with the opaque pointer.
        MemorySegment handleValue;
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment slot = tmp.allocate(ADDRESS);
            CudaBindings.check(CudaBindings.callInt(cuda.cublasCreate, slot), "cublasCreate_v2");
            // Extract the opaque handle address; it lives until cublasDestroy_v2 is called.
            handleValue = slot.get(ADDRESS, 0);
        }

        String name = CudaAvailability.deviceName(deviceIndex);
        long   vram = CudaAvailability.vramBytes(deviceIndex);
        log.info(String.format("GpuContext ready — device %d: %s, %.1f GB VRAM (shared=%b)",
            deviceIndex, name, vram / 1e9, processShared));

        return new GpuContext(deviceIndex, handleValue, processShared);
    }

    /**
     * The cuBLAS handle (opaque pointer) — valid until close().
     *
     * Pass directly to cuBLAS Panama downcalls (ADDRESS-typed parameter).
     */
    public MemorySegment handle() {
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

    /** Destroys the cuBLAS handle and releases device resources. */
    @Override
    public void close() {
        if (processShared) {
            log.fine("GpuContext.close ignored — process-shared instance (device " + deviceIndex + ")");
            return;
        }
        if (!closed) {
            closed = true;
            try {
                CudaBindings.callInt(CudaBindings.instance().cublasDestroy, handle);
                log.info("GpuContext closed — device " + deviceIndex);
            } catch (Exception e) {
                log.warning("Error closing GpuContext: " + e.getMessage());
            }
        }
    }
}