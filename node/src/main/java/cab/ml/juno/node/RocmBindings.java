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

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.util.logging.Logger;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Panama FFI downcall handles for the AMD HIP runtime and rocBLAS.
 *
 * <p>Implements {@link GpuBindings} — all field names are private; they are
 * exposed through the interface accessor methods so that {@link CudaMatVec},
 * {@link DeviceFloatMatrix}, and {@link DeviceHalfMatrix} need no changes
 * when switching between CUDA and ROCm backends.
 *
 * <h3>Library mapping</h3>
 * <ul>
 *   <li>HIP runtime  → {@code libamdhip64.so}  (mirrors cudart API)
 *   <li>rocBLAS       → {@code librocblas.so}   (used for both FP32 sgemv and FP16 hssgemv)
 * </ul>
 *
 * <h3>Key differences from CUDA</h3>
 * <ul>
 *   <li>{@code opTranspose() = 112} ({@code rocblas_operation_transpose}; CUDA uses 1).
 *   <li>{@code hipHostMalloc} takes an extra {@code unsigned int flags} argument;
 *       we pre-bind {@code flags=0} via {@link MethodHandles#insertArguments} so the
 *       accessor presents the same arity as {@code cudaMallocHost} to all callers.
 *   <li>FP16 batched GEMV uses {@code rocblas_hssgemv_strided_batched} from rocBLAS
 *       (identical signature to {@code cublasHSSgemvStridedBatched}).
 *   <li>{@code hipDeviceProp_t} struct: sizeof=1472, name@0, totalGlobalMem@288
 *       (measured from ROCm 7.2.3 headers on Linux x86_64).
 * </ul>
 *
 * <p>Requires JVM flag: {@code --enable-native-access=ALL-UNNAMED}.
 *
 * @see <a href="https://github.com/ml-cab/juno/issues/27">Issue #27 — ROCm backend</a>
 */
final class RocmBindings implements GpuBindings {

    private static final Logger log = Logger.getLogger(RocmBindings.class.getName());

    // ── hipDeviceProp_t layout (ROCm 7.2.3, Linux x86_64) ────────────────────
    // Verified via offsetof() compiled with /opt/rocm/bin/hipcc:
    //   sizeof(hipDeviceProp_t)  = 1472
    //   offsetof(name)           = 0
    //   offsetof(totalGlobalMem) = 288
    //   offsetof(gcnArchName)    = 1160
    private static final int  HIP_DEVICE_PROP_BYTES     = 1472;
    private static final long HIP_PROP_NAME_OFFSET      = 0L;
    private static final long HIP_PROP_TOTAL_MEM_OFFSET = 288L;

    // ── Singleton ─────────────────────────────────────────────────────────────
    private static final RocmBindings INSTANCE;
    private static final Throwable    INIT_FAILURE;

    static {
        RocmBindings b   = null;
        Throwable    err = null;
        try {
            b = new RocmBindings();
        } catch (Throwable t) {
            err = t;
            log.info("RocmBindings unavailable — " + t.getMessage());
        }
        INSTANCE     = b;
        INIT_FAILURE = err;
    }

    /** Returns {@code true} if HIP and rocBLAS libraries were resolved. */
    static boolean isAvailable() { return INSTANCE != null; }

    /**
     * Returns the singleton.
     * @throws IllegalStateException if ROCm libraries could not be loaded
     */
    static RocmBindings instance() {
        if (INSTANCE == null)
            throw new IllegalStateException(
                "ROCm not available: " + INIT_FAILURE.getMessage(), INIT_FAILURE);
        return INSTANCE;
    }

    // ── HIP runtime handles (private — exposed via accessors) ─────────────────
    private final MethodHandle hipGetDeviceCount;
    private final MethodHandle hipGetDeviceProperties;
    private final MethodHandle hipSetDevice;
    private final MethodHandle hipMalloc;
    private final MethodHandle hipFree;
    private final MethodHandle hipHostMalloc;       // pre-bound flags=0
    private final MethodHandle hipHostFree;
    private final MethodHandle hipMemcpy;
    private final MethodHandle hipMemcpyAsync;
    private final MethodHandle hipStreamCreateWithFlags;
    private final MethodHandle hipStreamSynchronize;
    private final MethodHandle hipStreamDestroy;

    // ── rocBLAS handles (private — exposed via accessors) ─────────────────────
    private final MethodHandle rocblasCreateHandle;
    private final MethodHandle rocblasDestroyHandle;
    private final MethodHandle rocblasSetStream;
    private final MethodHandle rocblasSetPointerMode;
    private final MethodHandle rocblasSgemv;
    private final MethodHandle rocblasHSSgemvStridedBatched;

    // ── Construction ──────────────────────────────────────────────────────────

    private RocmBindings() {
        Linker       l   = Linker.nativeLinker();
        SymbolLookup hip = GpuBindings.loadLibrary("libamdhip64.so");
        SymbolLookup rb  = GpuBindings.loadLibrary("librocblas.so");

        hipGetDeviceCount        = GpuBindings.bind(l, hip, "hipGetDeviceCount",
            FunctionDescriptor.of(JAVA_INT, ADDRESS));
        // ROCm 6+ exports hipGetDevicePropertiesR0600 as the versioned entry point
        hipGetDeviceProperties   = GpuBindings.bind(l, hip, "hipGetDevicePropertiesR0600",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));
        hipSetDevice             = GpuBindings.bind(l, hip, "hipSetDevice",
            FunctionDescriptor.of(JAVA_INT, JAVA_INT));
        hipMalloc                = GpuBindings.bind(l, hip, "hipMalloc",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG));
        hipFree                  = GpuBindings.bind(l, hip, "hipFree",
            FunctionDescriptor.of(JAVA_INT, ADDRESS));
        // hipHostMalloc has an extra `unsigned int flags` arg; pre-bind flags=0
        // so all callers see the same (ADDRESS, JAVA_LONG) → int shape as cudaMallocHost.
        hipHostMalloc = MethodHandles.insertArguments(
            GpuBindings.bind(l, hip, "hipHostMalloc",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_INT)),
            2, 0);
        hipHostFree              = GpuBindings.bind(l, hip, "hipHostFree",
            FunctionDescriptor.of(JAVA_INT, ADDRESS));
        hipMemcpy                = GpuBindings.bind(l, hip, "hipMemcpy",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT));
        hipMemcpyAsync           = GpuBindings.bind(l, hip, "hipMemcpyAsync",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT, ADDRESS));
        hipStreamCreateWithFlags = GpuBindings.bind(l, hip, "hipStreamCreateWithFlags",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));
        hipStreamSynchronize     = GpuBindings.bind(l, hip, "hipStreamSynchronize",
            FunctionDescriptor.of(JAVA_INT, ADDRESS));
        hipStreamDestroy         = GpuBindings.bind(l, hip, "hipStreamDestroy",
            FunctionDescriptor.of(JAVA_INT, ADDRESS));
        rocblasCreateHandle      = GpuBindings.bind(l, rb, "rocblas_create_handle",
            FunctionDescriptor.of(JAVA_INT, ADDRESS));
        rocblasDestroyHandle     = GpuBindings.bind(l, rb, "rocblas_destroy_handle",
            FunctionDescriptor.of(JAVA_INT, ADDRESS));
        rocblasSetStream         = GpuBindings.bind(l, rb, "rocblas_set_stream",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
        rocblasSetPointerMode    = GpuBindings.bind(l, rb, "rocblas_set_pointer_mode",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));
        rocblasSgemv             = GpuBindings.bind(l, rb, "rocblas_sgemv",
            FunctionDescriptor.of(JAVA_INT,
                ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT,
                ADDRESS, ADDRESS, JAVA_INT,
                ADDRESS, JAVA_INT,
                ADDRESS, ADDRESS, JAVA_INT));
        rocblasHSSgemvStridedBatched = GpuBindings.bind(l, rb, "rocblas_hssgemv_strided_batched",
            FunctionDescriptor.of(JAVA_INT,
                ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT,
                ADDRESS, ADDRESS, JAVA_INT, JAVA_LONG,
                ADDRESS, JAVA_INT, JAVA_LONG,
                ADDRESS, ADDRESS, JAVA_INT, JAVA_LONG,
                JAVA_INT));
        log.info("RocmBindings ready — Panama FFI (libamdhip64 + librocblas)");
    }

    // ── Device memory ─────────────────────────────────────────────────────────

    @Override
    public MemorySegment deviceMalloc(int deviceIndex, long bytes) {
        GpuBindings.check(GpuBindings.callInt(hipSetDevice, deviceIndex), "hipSetDevice");
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment slot = tmp.allocate(ADDRESS);
            GpuBindings.check(GpuBindings.callInt(hipMalloc, slot, bytes), "hipMalloc");
            return slot.get(ADDRESS, 0).reinterpret(bytes);
        }
    }

    @Override
    public void deviceFree(MemorySegment devicePtr) {
        if (devicePtr == null || devicePtr.equals(MemorySegment.NULL)) return;
        GpuBindings.callInt(hipFree, devicePtr);
    }

    // ── GpuBindings accessors ─────────────────────────────────────────────────

    @Override public MethodHandle cudaGetDeviceCount()          { return hipGetDeviceCount; }
    @Override public MethodHandle cudaGetDeviceProperties()     { return hipGetDeviceProperties; }
    @Override public MethodHandle cudaSetDevice()               { return hipSetDevice; }
    @Override public MethodHandle cudaMalloc()                  { return hipMalloc; }
    @Override public MethodHandle cudaFree()                    { return hipFree; }
    @Override public MethodHandle cudaMallocHost()              { return hipHostMalloc; }
    @Override public MethodHandle cudaFreeHost()                { return hipHostFree; }
    @Override public MethodHandle cudaMemcpy()                  { return hipMemcpy; }
    @Override public MethodHandle cudaMemcpyAsync()             { return hipMemcpyAsync; }
    @Override public MethodHandle cudaStreamCreateWithFlags()   { return hipStreamCreateWithFlags; }
    @Override public MethodHandle cudaStreamSynchronize()       { return hipStreamSynchronize; }
    @Override public MethodHandle cudaStreamDestroy()           { return hipStreamDestroy; }
    @Override public MethodHandle cublasCreate()                { return rocblasCreateHandle; }
    @Override public MethodHandle cublasDestroy()               { return rocblasDestroyHandle; }
    @Override public MethodHandle cublasSetStream()             { return rocblasSetStream; }
    @Override public MethodHandle cublasSetPointerMode()        { return rocblasSetPointerMode; }
    @Override public MethodHandle cublasSgemv()                 { return rocblasSgemv; }
    @Override public MethodHandle cublasHSSgemvStridedBatched() { return rocblasHSSgemvStridedBatched; }
    @Override public int    opTranspose()       { return 112; } // rocblas_operation_transpose
    @Override public int    pointerModeHost()   { return 0; }   // rocblas_pointer_mode_host
    @Override public int    devicePropBytes()   { return HIP_DEVICE_PROP_BYTES; }
    @Override public long   propNameOffset()    { return HIP_PROP_NAME_OFFSET; }
    @Override public long   propTotalMemOffset(){ return HIP_PROP_TOTAL_MEM_OFFSET; }
    @Override public String backendLabel()      { return "rocm"; }
}
