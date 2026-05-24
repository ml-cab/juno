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
import java.util.logging.Logger;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Panama FFI downcall handles for the CUDA Runtime (cudart) and cuBLAS.
 *
 * <p>Extends {@link GpuBindings} — all field names and utility methods are
 * inherited. This class resolves symbols against {@code libcudart.so.12} and
 * {@code libcublas.so.12} and passes them to the parent constructor.
 *
 * <p>Requires JVM flag: {@code --enable-native-access=ALL-UNNAMED}.
 */
final class CudaBindings extends GpuBindings {

    private static final Logger log = Logger.getLogger(CudaBindings.class.getName());

    // ── cudaDeviceProp layout (CUDA 12.x, Linux x86_64) ──────────────────────
    private static final int  CUDA_DEVICE_PROP_BYTES     = 1512;
    private static final long CUDA_PROP_NAME_OFFSET      = 0L;
    private static final long CUDA_PROP_TOTAL_MEM_OFFSET = 288L;

    // ── Singleton ─────────────────────────────────────────────────────────────
    private static final CudaBindings INSTANCE;
    private static final Throwable    INIT_FAILURE;

    static {
        CudaBindings b   = null;
        Throwable    err = null;
        try {
            b = new CudaBindings();
        } catch (Throwable t) {
            err = t;
            log.info("CudaBindings unavailable — " + t.getMessage());
        }
        INSTANCE     = b;
        INIT_FAILURE = err;
    }

    /** Returns {@code true} if CUDA and cuBLAS libraries were resolved. */
    static boolean isAvailable() { return INSTANCE != null; }

    /**
     * Returns the singleton.
     * @throws IllegalStateException if CUDA libraries could not be loaded
     */
    static CudaBindings instance() {
        if (INSTANCE == null)
            throw new IllegalStateException(
                "CUDA not available: " + INIT_FAILURE.getMessage(), INIT_FAILURE);
        return INSTANCE;
    }

    // ── Construction ──────────────────────────────────────────────────────────

    private CudaBindings() {
        this(Linker.nativeLinker(),
             loadLibrary("libcudart.so.12", "libcudart.so"),
             loadLibrary("libcublas.so.12", "libcublas.so"));
    }

    private CudaBindings(Linker l, SymbolLookup rt, SymbolLookup blas) {
        super(
            /* opTranspose        */ 1,   // CUBLAS_OP_T
            /* pointerModeHost    */ 0,   // CUBLAS_POINTER_MODE_HOST
            /* devicePropBytes    */ CUDA_DEVICE_PROP_BYTES,
            /* propNameOffset     */ CUDA_PROP_NAME_OFFSET,
            /* propTotalMemOffset */ CUDA_PROP_TOTAL_MEM_OFFSET,
            /* backendLabel       */ "cuda",
            // ── runtime ───────────────────────────────────────────────────────
            bind(l, rt, "cudaGetDeviceCount",
                FunctionDescriptor.of(JAVA_INT, ADDRESS)),
            bind(l, rt, "cudaGetDeviceProperties",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT)),
            bind(l, rt, "cudaSetDevice",
                FunctionDescriptor.of(JAVA_INT, JAVA_INT)),
            bind(l, rt, "cudaMalloc",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG)),
            bind(l, rt, "cudaFree",
                FunctionDescriptor.of(JAVA_INT, ADDRESS)),
            bind(l, rt, "cudaMallocHost",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG)),
            bind(l, rt, "cudaFreeHost",
                FunctionDescriptor.of(JAVA_INT, ADDRESS)),
            bind(l, rt, "cudaMemcpy",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT)),
            bind(l, rt, "cudaMemcpyAsync",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT, ADDRESS)),
            bind(l, rt, "cudaStreamCreateWithFlags",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT)),
            bind(l, rt, "cudaStreamSynchronize",
                FunctionDescriptor.of(JAVA_INT, ADDRESS)),
            bind(l, rt, "cudaStreamDestroy",
                FunctionDescriptor.of(JAVA_INT, ADDRESS)),
            // ── blas ──────────────────────────────────────────────────────────
            bind(l, blas, "cublasCreate_v2",
                FunctionDescriptor.of(JAVA_INT, ADDRESS)),
            bind(l, blas, "cublasDestroy_v2",
                FunctionDescriptor.of(JAVA_INT, ADDRESS)),
            bind(l, blas, "cublasSetStream_v2",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS)),
            bind(l, blas, "cublasSetPointerMode_v2",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT)),
            bind(l, blas, "cublasSgemv_v2",
                FunctionDescriptor.of(JAVA_INT,
                    ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT,
                    ADDRESS, ADDRESS, JAVA_INT,
                    ADDRESS, JAVA_INT,
                    ADDRESS, ADDRESS, JAVA_INT)),
            bind(l, blas, "cublasHSSgemvStridedBatched",
                FunctionDescriptor.of(JAVA_INT,
                    ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT,
                    ADDRESS, ADDRESS, JAVA_INT, JAVA_LONG,
                    ADDRESS, JAVA_INT, JAVA_LONG,
                    ADDRESS, ADDRESS, JAVA_INT, JAVA_LONG,
                    JAVA_INT))
        );
        log.info("CudaBindings ready — Panama FFI (cudart + cublas)");
    }
}
