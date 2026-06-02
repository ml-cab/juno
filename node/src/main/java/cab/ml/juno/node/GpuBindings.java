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

/**
 * Vendor-neutral GPU interface.
 *
 * <p>Implemented by {@link CudaBindings} (NVIDIA CUDA) and {@link RocmBindings}
 * (AMD ROCm/HIP). All callers ({@link CudaMatVec}, {@link DeviceFloatMatrix},
 * {@link DeviceHalfMatrix}, {@link GpuContext}) program against this interface
 * only, switching backend transparently through {@link GpuContext#bindings()}.
 *
 * <h3>Design — Open/Closed Principle</h3>
 * <ul>
 *   <li>Existing classes ({@link CudaBindings}) are <em>closed for modification</em>:
 *       they only gain {@code implements GpuBindings} and the corresponding
 *       {@code @Override} accessor methods — no existing lines are removed.
 *   <li>New backends are <em>open for extension</em>: implement this interface
 *       and register in {@link GpuContext#selectBindings()}.
 * </ul>
 *
 * <h3>Static constants</h3>
 * {@link #H2D}, {@link #D2H}, {@link #STREAM_NON_BLOCKING} are identical in
 * CUDA and HIP — declared here as interface constants.
 *
 * <h3>Static utilities</h3>
 * {@link #check}, {@link #callInt}, {@link #loadLibrary}, {@link #bind} are
 * shared helpers available to all implementations and callers.
 *
 * <h3>Naming convention</h3>
 * All accessor methods use vendor-neutral names (e.g. {@code gpuGetDeviceCount},
 * {@code blasCreate}). Each implementation documents the underlying symbol it
 * resolves (e.g. {@code cudaGetDeviceCount} or {@code hipGetDeviceCount}).
 *
 * <h3>MatVec factory</h3>
 * {@link #createMatVec(GpuContext)} returns the correct {@link MatVec}
 * implementation for each backend, keeping {@link GpuContext#createMatVec()}
 * free of {@code instanceof} checks.
 */
interface GpuBindings {

    // ── cudaMemcpyKind / hipMemcpyKind ────────────────────────────────────────
    /** HostToDevice — identical value in CUDA and HIP. */
    int H2D = 1;
    /** DeviceToHost — identical value in CUDA and HIP. */
    int D2H = 2;

    // ── Stream flag ───────────────────────────────────────────────────────────
    /** {@code cudaStreamNonBlocking} / {@code hipStreamNonBlocking}. */
    int STREAM_NON_BLOCKING = 0x01;

    // ── Handle accessors (runtime) ────────────────────────────────────────────
    /** {@code cudaGetDeviceCount} / {@code hipGetDeviceCount}. */
    MethodHandle gpuGetDeviceCount();
    /** {@code cudaGetDeviceProperties} / {@code hipGetDevicePropertiesR0600}. */
    MethodHandle gpuGetDeviceProperties();
    /** {@code cudaSetDevice} / {@code hipSetDevice}. */
    MethodHandle gpuSetDevice();
    /** {@code cudaMalloc} / {@code hipMalloc}. */
    MethodHandle gpuMalloc();
    /** {@code cudaFree} / {@code hipFree}. */
    MethodHandle gpuFree();
    /** {@code cudaMallocHost} / {@code hipHostMalloc} (flags pre-bound to 0). */
    MethodHandle gpuMallocHost();
    /** {@code cudaFreeHost} / {@code hipHostFree}. */
    MethodHandle gpuFreeHost();
    /** {@code cudaMemcpy} / {@code hipMemcpy}. */
    MethodHandle gpuMemcpy();
    /** {@code cudaMemcpyAsync} / {@code hipMemcpyAsync}. */
    MethodHandle gpuMemcpyAsync();
    /** {@code cudaStreamCreateWithFlags} / {@code hipStreamCreateWithFlags}. */
    MethodHandle gpuStreamCreateWithFlags();
    /** {@code cudaStreamSynchronize} / {@code hipStreamSynchronize}. */
    MethodHandle gpuStreamSynchronize();
    /** {@code cudaStreamDestroy} / {@code hipStreamDestroy}. */
    MethodHandle gpuStreamDestroy();

    // ── Handle accessors (BLAS) ───────────────────────────────────────────────
    /** {@code cublasCreate_v2} / {@code rocblas_create_handle}. */
    MethodHandle blasCreate();
    /** {@code cublasDestroy_v2} / {@code rocblas_destroy_handle}. */
    MethodHandle blasDestroy();
    /** {@code cublasSetStream_v2} / {@code rocblas_set_stream}. */
    MethodHandle blasSetStream();
    /** {@code cublasSetPointerMode_v2} / {@code rocblas_set_pointer_mode}. */
    MethodHandle blasSetPointerMode();
    /** {@code cublasSgemv_v2} / {@code rocblas_sgemv}. */
    MethodHandle blasSgemv();
    /** {@code cublasHSSgemvStridedBatched} / {@code rocblas_hssgemv_strided_batched}. */
    MethodHandle blasHSSgemvStridedBatched();

    // ── Vendor-specific constants ─────────────────────────────────────────────
    /**
     * Operation "transpose" for the BLAS GEMV call.
     * CUDA cuBLAS: {@code CUBLAS_OP_T = 1}.
     * AMD rocBLAS: {@code rocblas_operation_transpose = 112}.
     */
    int opTranspose();

    /**
     * Pointer mode "host" for scalar arguments.
     * Both vendors: {@code 0}.
     */
    int pointerModeHost();

    // ── Device-prop struct layout ─────────────────────────────────────────────
    /** {@code sizeof(cudaDeviceProp)} or {@code sizeof(hipDeviceProp_t)} in bytes. */
    int devicePropBytes();

    /** Byte offset of {@code char name[]} inside the device-prop struct. */
    long propNameOffset();

    /** Byte offset of {@code size_t totalGlobalMem} inside the device-prop struct. */
    long propTotalMemOffset();

    // ── Backend label ─────────────────────────────────────────────────────────
    /** Human-readable backend label: {@code "cuda"} or {@code "rocm"}. */
    String backendLabel();

    // ── Device memory helpers ─────────────────────────────────────────────────

    /**
     * Allocates {@code bytes} of device memory on {@code deviceIndex} and returns
     * a {@link MemorySegment} wrapping the device pointer.
     */
    MemorySegment deviceMalloc(int deviceIndex, long bytes);

    /**
     * Frees a device pointer previously returned by {@link #deviceMalloc}.
     */
    void deviceFree(MemorySegment devicePtr);

    // ── MatVec factory ────────────────────────────────────────────────────────

    /**
     * Creates the {@link MatVec} implementation for this backend.
     * Called by {@link GpuContext#createMatVec()}.
     */
    MatVec createMatVec(GpuContext ctx);

    // ── Static utilities ──────────────────────────────────────────────────────

    /** Throws {@link IllegalStateException} if {@code rc != 0}. */
    static void check(int rc, String op) {
        if (rc != 0)
            throw new IllegalStateException(op + " failed: rc=" + rc);
    }

    /** Reflective invoke returning the {@code int} return value. */
    static int callInt(MethodHandle mh, Object... args) {
        try {
            return (int) mh.invokeWithArguments(args);
        } catch (Throwable t) {
            throw new IllegalStateException("GPU downcall failed", t);
        }
    }

    /** Tries each candidate library name in order; throws if none loads. */
    static SymbolLookup loadLibrary(String... candidates) {
        for (String name : candidates) {
            try {
                return SymbolLookup.libraryLookup(name, Arena.global());
            } catch (IllegalArgumentException ignored) { /* try next */ }
        }
        throw new IllegalStateException(
            "Could not load GPU library — tried: " + java.util.Arrays.toString(candidates));
    }

    /** Resolves {@code symbol} from {@code lib} and creates a downcall handle. */
    static MethodHandle bind(Linker linker, SymbolLookup lib,
                              String symbol, FunctionDescriptor desc) {
        MemorySegment addr = lib.find(symbol)
            .orElseThrow(() -> new IllegalStateException("GPU symbol not found: " + symbol));
        return linker.downcallHandle(addr, desc);
    }
}