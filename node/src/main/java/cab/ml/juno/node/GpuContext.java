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
 * GPU context: device selection and BLAS handle lifecycle, via Panama FFI.
 *
 * The BLAS handle is an opaque pointer ({@code cublasHandle_t} or
 * {@code rocblas_handle}) stored as a {@link MemorySegment}. It is created by
 * {@code cublasCreate_v2} / {@code rocblas_create_handle} and destroyed on
 * {@link #close()}, both resolved through {@link GpuBindings}.
 *
 * Prefer {@link #shared(int)} for a process-wide handle on long-lived JVMs
 * (embedded node, ForwardPassHandlerLoader GPU path). Use {@link #init(int)}
 * when tests or tools need an isolated handle they can {@link #close()} without
 * affecting other code.
 *
 * The BLAS handle is shared across all {@link CudaMatVec} calls on the same
 * GpuContext. {@link #cublasSerializationLock()} serializes stream binding and
 * kernel execution. Replace with ReentrantLock when addressing Loom pinning
 * (point 4 of the HPC audit).
 *
 * Backend auto-detection order: CUDA first, then ROCm. Override with system
 * property {@code juno.gpu.backend=cuda|rocm|auto} (default: {@code auto}).
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
    private final GpuBindings   bindings;
    /** Opaque BLAS handle (cublasHandle_t or rocblas_handle). */
    private final MemorySegment handle;
    private final boolean       processShared;
    private final Object        cublasSerialization = new Object();
    private volatile boolean    closed = false;

    private GpuContext(int deviceIndex, GpuBindings bindings,
                       MemorySegment handle, boolean processShared) {
        this.deviceIndex   = deviceIndex;
        this.bindings      = bindings;
        this.handle        = handle;
        this.processShared = processShared;
    }

    /**
     * The vendor-neutral GPU bindings for this context.
     * Returns {@link CudaBindings} on NVIDIA or {@link RocmBindings} on AMD.
     */
    public GpuBindings bindings() {
        if (closed) throw new IllegalStateException("GpuContext already closed");
        return bindings;
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
     * Process-wide singleton per GPU device. {@link #close()} is a no-op on
     * these instances so embedded servers do not tear down the GPU under live code.
     *
     * @param deviceIndex 0-based GPU device index
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
     * Initialises the GPU device and creates a BLAS handle.
     *
     * @param deviceIndex 0-based GPU index (0 for single-GPU nodes)
     * @return a ready GpuContext — caller must close() when done
     * @throws IllegalStateException if no GPU backend is available or init fails
     */
    public static GpuContext init(int deviceIndex) {
        return create(deviceIndex, false);
    }

    private static GpuContext create(int deviceIndex, boolean processShared) {
        GpuBindings gpu = selectBindings();

        GpuBindings.check(GpuBindings.callInt(gpu.gpuSetDevice(), deviceIndex), "setDevice");

        // BLAS create handle: cublasCreate_v2 or rocblas_create_handle
        MemorySegment handleValue;
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment slot = tmp.allocate(ADDRESS);
            GpuBindings.check(GpuBindings.callInt(gpu.blasCreate(), slot), "blasCreate");
            // Extract the opaque handle address; it lives until blasDestroy is called.
            handleValue = slot.get(ADDRESS, 0);
        }

        String name = deviceName(gpu, deviceIndex);
        long   vram = deviceVram(gpu, deviceIndex);
        log.info(String.format("GpuContext ready — [%s] device %d: %s, %.1f GB VRAM (shared=%b)",
            gpu.backendLabel(), deviceIndex, name, vram / 1e9, processShared));

        return new GpuContext(deviceIndex, gpu, handleValue, processShared);
    }

    /**
     * Selects the GPU binding backend based on the {@code juno.gpu.backend}
     * system property (default {@code auto}).
     *
     * <ul>
     *   <li>{@code cuda} — force CUDA
     *   <li>{@code rocm} — force ROCm
     *   <li>{@code auto} — try CUDA first, then ROCm
     * </ul>
     *
     * @throws IllegalStateException if no usable GPU backend is found
     */
    static GpuBindings selectBindings() {
        String pref = System.getProperty("juno.gpu.backend", "auto").toLowerCase();
        return switch (pref) {
            case "cuda" -> {
                if (!CudaBindings.isAvailable())
                    throw new IllegalStateException("juno.gpu.backend=cuda but CUDA not available");
                yield CudaBindings.instance();
            }
            case "rocm" -> {
                if (!RocmBindings.isAvailable())
                    throw new IllegalStateException("juno.gpu.backend=rocm but ROCm not available");
                yield RocmBindings.instance();
            }
            default -> {
                if (CudaBindings.isAvailable()) yield CudaBindings.instance();
                if (RocmBindings.isAvailable()) yield RocmBindings.instance();
                throw new IllegalStateException(
                    "No GPU backend available — CUDA and ROCm libraries not found");
            }
        };
    }

    /**
     * The BLAS handle (opaque pointer) — valid until close().
     *
     * Pass directly to cuBLAS / rocBLAS Panama downcalls (ADDRESS-typed parameter).
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

    /** Backend label: {@code "cuda"} or {@code "rocm"}. */
    public String backendLabel() { return bindings.backendLabel(); }

    /**
     * Creates the appropriate {@link MatVec} implementation for this GPU context.
     *
     * Delegates to {@link GpuBindings#createMatVec(GpuContext)} so that adding
     * a new backend requires no changes here.
     */
    public MatVec createMatVec() {
        return bindings.createMatVec(this);
    }

    /** Destroys the BLAS handle and releases device resources. */
    @Override
    public void close() {
        if (processShared) {
            log.fine("GpuContext.close ignored — process-shared instance (device " + deviceIndex + ")");
            return;
        }
        if (!closed) {
            closed = true;
            try {
                GpuBindings.callInt(bindings.blasDestroy(), handle);
                log.info("GpuContext closed — device " + deviceIndex);
            } catch (Exception e) {
                log.warning("Error closing GpuContext: " + e.getMessage());
            }
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private static String deviceName(GpuBindings gpu, int index) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment prop = arena.allocate(gpu.devicePropBytes());
            int rc = (int) gpu.gpuGetDeviceProperties().invokeExact(prop, index);
            if (rc != 0) return "unknown";
            String name = prop.getString(gpu.propNameOffset());
            return name != null ? name.trim() : "unknown";
        } catch (Throwable t) {
            return "unknown";
        }
    }

    private static long deviceVram(GpuBindings gpu, int index) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment prop = arena.allocate(gpu.devicePropBytes());
            int rc = (int) gpu.gpuGetDeviceProperties().invokeExact(prop, index);
            if (rc != 0) return 0L;
            return prop.get(java.lang.foreign.ValueLayout.JAVA_LONG, gpu.propTotalMemOffset());
        } catch (Throwable t) {
            return 0L;
        }
    }
}