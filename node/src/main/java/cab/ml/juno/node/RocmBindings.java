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

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
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
 * <p>Extends {@link GpuBindings} — all field names are inherited from the parent so
 * that {@link CudaMatVec}, {@link DeviceFloatMatrix}, and {@link DeviceHalfMatrix} need
 * no changes; only field values differ from {@link CudaBindings}.
 *
 * <h3>Library mapping</h3>
 * <ul>
 *   <li>HIP runtime  → {@code libamdhip64.so}  (mirrors cudart API)
 *   <li>rocBLAS       → {@code librocblas.so}   (used for both FP32 sgemv and FP16 hssgemv)
 * </ul>
 *
 * <h3>Key differences from CUDA</h3>
 * <ul>
 *   <li>{@code OP_TRANSPOSE = 112} ({@code rocblas_operation_transpose}; CUDA uses 1).
 *   <li>{@code hipHostMalloc} takes an extra {@code unsigned int flags} argument;
 *       we pre-bind {@code flags=0} via {@link MethodHandles#insertArguments} so the
 *       field looks identical to {@code cudaMallocHost} to all callers.
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
final class RocmBindings extends GpuBindings {

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

    // ── Construction ──────────────────────────────────────────────────────────

    private RocmBindings() {
        this(Linker.nativeLinker(),
             loadLibrary("libamdhip64.so"),
             loadLibrary("librocblas.so"));
    }

    private RocmBindings(Linker l, SymbolLookup hip, SymbolLookup rb) {
        super(
            /* opTranspose        */ 112,  // rocblas_operation_transpose
            /* pointerModeHost    */ 0,    // rocblas_pointer_mode_host
            /* devicePropBytes    */ HIP_DEVICE_PROP_BYTES,
            /* propNameOffset     */ HIP_PROP_NAME_OFFSET,
            /* propTotalMemOffset */ HIP_PROP_TOTAL_MEM_OFFSET,
            /* backendLabel       */ "rocm",
            // ── HIP runtime ───────────────────────────────────────────────────
            bind(l, hip, "hipGetDeviceCount",
                FunctionDescriptor.of(JAVA_INT, ADDRESS)),
            bind(l, hip, "hipGetDevicePropertiesR0600",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT)),
            bind(l, hip, "hipSetDevice",
                FunctionDescriptor.of(JAVA_INT, JAVA_INT)),
            bind(l, hip, "hipMalloc",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG)),
            bind(l, hip, "hipFree",
                FunctionDescriptor.of(JAVA_INT, ADDRESS)),
            // hipHostMalloc has an extra `unsigned int flags` argument compared to
            // cudaMallocHost. Pre-bind flags=0 so all callers see the same arity.
            preBind0(bind(l, hip, "hipHostMalloc",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_INT))),
            bind(l, hip, "hipHostFree",
                FunctionDescriptor.of(JAVA_INT, ADDRESS)),
            bind(l, hip, "hipMemcpy",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT)),
            bind(l, hip, "hipMemcpyAsync",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT, ADDRESS)),
            bind(l, hip, "hipStreamCreateWithFlags",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT)),
            bind(l, hip, "hipStreamSynchronize",
                FunctionDescriptor.of(JAVA_INT, ADDRESS)),
            bind(l, hip, "hipStreamDestroy",
                FunctionDescriptor.of(JAVA_INT, ADDRESS)),
            // ── rocBLAS (handle API mirrors cuBLAS) ───────────────────────────
            bind(l, rb, "rocblas_create_handle",
                FunctionDescriptor.of(JAVA_INT, ADDRESS)),
            bind(l, rb, "rocblas_destroy_handle",
                FunctionDescriptor.of(JAVA_INT, ADDRESS)),
            bind(l, rb, "rocblas_set_stream",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS)),
            bind(l, rb, "rocblas_set_pointer_mode",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT)),
            // rocblas_sgemv — identical param layout to cublasSgemv_v2
            bind(l, rb, "rocblas_sgemv",
                FunctionDescriptor.of(JAVA_INT,
                    ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT,
                    ADDRESS, ADDRESS, JAVA_INT,
                    ADDRESS, JAVA_INT,
                    ADDRESS, ADDRESS, JAVA_INT)),
            // rocblas_hssgemv_strided_batched — identical param layout to
            // cublasHSSgemvStridedBatched (FP16 A+x, FP32 y, batched)
            bind(l, rb, "rocblas_hssgemv_strided_batched",
                FunctionDescriptor.of(JAVA_INT,
                    ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT,
                    ADDRESS, ADDRESS, JAVA_INT, JAVA_LONG,
                    ADDRESS, JAVA_INT, JAVA_LONG,
                    ADDRESS, ADDRESS, JAVA_INT, JAVA_LONG,
                    JAVA_INT))
        );
        log.info("RocmBindings ready — Panama FFI (libamdhip64 + librocblas)");
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /**
     * Drops the last {@code int} argument from a 3-arg MethodHandle by pre-binding
     * it to {@code 0}. Used to adapt {@code hipHostMalloc(ptr, size, flags=0)} so
     * it presents the same {@code (ADDRESS, JAVA_LONG) → int} shape as
     * {@code cudaMallocHost}.
     */
    private static MethodHandle preBind0(MethodHandle mh) {
        // insertArguments inserts at position 2 (the flags arg): fixes it to 0.
        return MethodHandles.insertArguments(mh, 2, 0);
    }
}
