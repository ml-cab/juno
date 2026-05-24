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
 * GPU context: device selection, BLAS handle lifecycle, vendor-neutral via {@link GpuBindings}.
 *
 * <p>The BLAS handle is an opaque pointer stored as a {@link MemorySegment}.
 * For CUDA it is a {@code cublasHandle_t}; for ROCm it is a {@code rocblas_handle} —
 * both are opaque void pointers with identical Java-side representation.
 *
 * <p>Backend auto-detection order in {@link #shared}/{@link #init}:
 * <ol>
 *   <li>CUDA — if {@code libcudart.so.12} + {@code libcublas.so.12} are present.
 *   <li>ROCm — if {@code libamdhip64.so} + {@code librocblas.so} are present.
 * </ol>
 * The first available backend wins. Override with system property
 * {@code juno.gpu.backend=cuda|rocm|auto} (default: {@code auto}).
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
    private final Object        blasSerialization = new Object();
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
     * Use instead of {@code CudaBindings.instance()} in all device-memory code.
     */
    public GpuBindings bindings() {
        if (closed) throw new IllegalStateException("GpuContext already closed");
        return bindings;
    }

    /**
     * Mutex for BLAS stream-binding / memcpy / GEMV sequences on this handle.
     */
    Object cublasSerializationLock() {
        return blasSerialization;
    }

    /**
     * Process-wide singleton per device. {@link #close()} is a no-op so embedded
     * servers do not tear down the GPU context under live code.
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
     * @param deviceIndex 0-based GPU index
     * @return a ready GpuContext — caller must {@link #close()} when done
     * @throws IllegalStateException if no GPU backend is available
     */
    public static GpuContext init(int deviceIndex) {
        return create(deviceIndex, false);
    }

    private static GpuContext create(int deviceIndex, boolean processShared) {
        GpuBindings gpu = selectBindings();

        GpuBindings.check(GpuBindings.callInt(gpu.cudaSetDevice, deviceIndex), "setDevice");

        // BLAS create handle: cublasCreate_v2 or rocblas_create_handle
        MemorySegment handleValue;
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment slot = tmp.allocate(ADDRESS);
            GpuBindings.check(GpuBindings.callInt(gpu.cublasCreate, slot), "blasCreate");
            handleValue = slot.get(ADDRESS, 0);
        }

        // Log device info
        String name = deviceName(gpu, deviceIndex);
        long   vram = deviceVram(gpu, deviceIndex);
        log.info(String.format("GpuContext ready — [%s] device %d: %s, %.1f GB VRAM (shared=%b)",
            gpu.backendLabel, deviceIndex, name, vram / 1e9, processShared));

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

    // ── Accessors ─────────────────────────────────────────────────────────────

    /**
     * The opaque BLAS handle ({@code cublasHandle_t} or {@code rocblas_handle}).
     * Valid until {@link #close()}.
     */
    public MemorySegment handle() {
        if (closed) throw new IllegalStateException("GpuContext already closed");
        return handle;
    }

    /** 0-based GPU device index. */
    public int deviceIndex() { return deviceIndex; }

    /** Whether this context has been closed. */
    public boolean isClosed() { return closed; }

    /** Whether this is the JVM-wide singleton from {@link #shared(int)}. */
    public boolean isProcessShared() { return processShared; }

    /** Backend label: {@code "cuda"} or {@code "rocm"}. */
    public String backendLabel() { return bindings.backendLabel; }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    @Override
    public void close() {
        if (processShared) {
            log.fine("GpuContext.close ignored — process-shared instance (device " + deviceIndex + ")");
            return;
        }
        if (!closed) {
            closed = true;
            try {
                GpuBindings.callInt(bindings.cublasDestroy, handle);
                log.info("GpuContext closed — device " + deviceIndex);
            } catch (Exception e) {
                log.warning("Error closing GpuContext: " + e.getMessage());
            }
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private static String deviceName(GpuBindings gpu, int index) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment prop = arena.allocate(gpu.DEVICE_PROP_BYTES);
            int rc = (int) gpu.cudaGetDeviceProperties.invokeExact(prop, index);
            if (rc != 0) return "unknown";
            String name = prop.getString(gpu.PROP_NAME_OFFSET);
            return name != null ? name.trim() : "unknown";
        } catch (Throwable t) {
            return "unknown";
        }
    }

    private static long deviceVram(GpuBindings gpu, int index) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment prop = arena.allocate(gpu.DEVICE_PROP_BYTES);
            int rc = (int) gpu.cudaGetDeviceProperties.invokeExact(prop, index);
            if (rc != 0) return 0L;
            return prop.get(java.lang.foreign.ValueLayout.JAVA_LONG, gpu.PROP_TOTAL_MEM_OFFSET);
        } catch (Throwable t) {
            return 0L;
        }
    }
}
