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

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Vendor-neutral GPU bindings base class.
 *
 * <p>Subclasses ({@link CudaBindings} for NVIDIA, {@link RocmBindings} for AMD)
 * initialize the same set of {@link MethodHandle} fields to vendor-specific symbols
 * resolved via Panama FFI. All callers ({@link CudaMatVec}, {@link DeviceFloatMatrix},
 * {@link DeviceHalfMatrix}) program against this type only, switching backend
 * transparently through {@link GpuContext#bindings()}.
 *
 * <h3>Constants</h3>
 * <ul>
 *   <li>Static finals {@link #H2D}, {@link #D2H}, {@link #STREAM_NON_BLOCKING} are
 *       identical in CUDA and HIP — kept static.
 *   <li>Instance fields {@link #OP_TRANSPOSE} and {@link #POINTER_MODE_HOST} differ
 *       between vendors (CUDA: 1/0, rocBLAS: 112/0) — kept as instance state.
 * </ul>
 *
 * <h3>Memory helpers</h3>
 * {@link #deviceMalloc} and {@link #deviceFree} are concrete methods built on the
 * subclass-initialised {@link #cudaMalloc} / {@link #cudaFree} handles.
 */
abstract class GpuBindings {

    // ── cudaMemcpyKind / hipMemcpyKind ────────────────────────────────────────
    static final int H2D = 1; // HostToDevice — same value in CUDA and HIP
    static final int D2H = 2; // DeviceToHost — same value in CUDA and HIP

    // ── Stream flag ───────────────────────────────────────────────────────────
    static final int STREAM_NON_BLOCKING = 0x01; // same in CUDA and HIP

    // ── Vendor-specific operation constants ───────────────────────────────────
    /**
     * Operation "transpose" for the BLAS GEMV call.
     * CUDA cuBLAS: {@code CUBLAS_OP_T = 1}.
     * AMD rocBLAS: {@code rocblas_operation_transpose = 112}.
     */
    final int OP_TRANSPOSE;

    /**
     * Pointer mode "host" for BLAS scalar arguments.
     * Both vendors: {@code 0}.
     */
    final int POINTER_MODE_HOST;

    // ── Device-prop struct layout ─────────────────────────────────────────────
    /** {@code sizeof(cudaDeviceProp)} or {@code sizeof(hipDeviceProp_t)} in bytes. */
    final int  DEVICE_PROP_BYTES;
    /** Byte offset of {@code char name[256]} inside the device-prop struct. */
    final long PROP_NAME_OFFSET;
    /** Byte offset of {@code size_t totalGlobalMem} inside the device-prop struct. */
    final long PROP_TOTAL_MEM_OFFSET;

    // ── Runtime handles (cudart / libamdhip64) ────────────────────────────────
    final MethodHandle cudaGetDeviceCount;        // int (int*)
    final MethodHandle cudaGetDeviceProperties;   // int (DeviceProp*, int)
    final MethodHandle cudaSetDevice;             // int (int)
    final MethodHandle cudaMalloc;                // int (void**, size_t)
    final MethodHandle cudaFree;                  // int (void*)
    final MethodHandle cudaMallocHost;            // int (void**, size_t)  [flags=0 pre-bound for HIP]
    final MethodHandle cudaFreeHost;              // int (void*)
    final MethodHandle cudaMemcpy;                // int (void*, void*, size_t, int)
    final MethodHandle cudaMemcpyAsync;           // int (void*, void*, size_t, int, stream)
    final MethodHandle cudaStreamCreateWithFlags; // int (stream*, int)
    final MethodHandle cudaStreamSynchronize;     // int (stream)
    final MethodHandle cudaStreamDestroy;         // int (stream)

    // ── BLAS handles (cuBLAS / rocBLAS) ──────────────────────────────────────
    final MethodHandle cublasCreate;              // int (handle*)
    final MethodHandle cublasDestroy;             // int (handle)
    final MethodHandle cublasSetStream;           // int (handle, stream)
    final MethodHandle cublasSetPointerMode;      // int (handle, int)
    final MethodHandle cublasSgemv;               // FP32 GEMV
    final MethodHandle cublasHSSgemvStridedBatched; // FP16-A/x, FP32-y, batched

    /** Backend label used in JFR events — {@code "cuda"} or {@code "rocm"}. */
    final String backendLabel;

    // ── Protected constructor ─────────────────────────────────────────────────

    protected GpuBindings(
            int opTranspose,
            int pointerModeHost,
            int devicePropBytes,
            long propNameOffset,
            long propTotalMemOffset,
            String backendLabel,
            MethodHandle cudaGetDeviceCount,
            MethodHandle cudaGetDeviceProperties,
            MethodHandle cudaSetDevice,
            MethodHandle cudaMalloc,
            MethodHandle cudaFree,
            MethodHandle cudaMallocHost,
            MethodHandle cudaFreeHost,
            MethodHandle cudaMemcpy,
            MethodHandle cudaMemcpyAsync,
            MethodHandle cudaStreamCreateWithFlags,
            MethodHandle cudaStreamSynchronize,
            MethodHandle cudaStreamDestroy,
            MethodHandle cublasCreate,
            MethodHandle cublasDestroy,
            MethodHandle cublasSetStream,
            MethodHandle cublasSetPointerMode,
            MethodHandle cublasSgemv,
            MethodHandle cublasHSSgemvStridedBatched) {
        this.OP_TRANSPOSE               = opTranspose;
        this.POINTER_MODE_HOST          = pointerModeHost;
        this.DEVICE_PROP_BYTES          = devicePropBytes;
        this.PROP_NAME_OFFSET           = propNameOffset;
        this.PROP_TOTAL_MEM_OFFSET      = propTotalMemOffset;
        this.backendLabel               = backendLabel;
        this.cudaGetDeviceCount         = cudaGetDeviceCount;
        this.cudaGetDeviceProperties    = cudaGetDeviceProperties;
        this.cudaSetDevice              = cudaSetDevice;
        this.cudaMalloc                 = cudaMalloc;
        this.cudaFree                   = cudaFree;
        this.cudaMallocHost             = cudaMallocHost;
        this.cudaFreeHost               = cudaFreeHost;
        this.cudaMemcpy                 = cudaMemcpy;
        this.cudaMemcpyAsync            = cudaMemcpyAsync;
        this.cudaStreamCreateWithFlags  = cudaStreamCreateWithFlags;
        this.cudaStreamSynchronize      = cudaStreamSynchronize;
        this.cudaStreamDestroy          = cudaStreamDestroy;
        this.cublasCreate               = cublasCreate;
        this.cublasDestroy              = cublasDestroy;
        this.cublasSetStream            = cublasSetStream;
        this.cublasSetPointerMode       = cublasSetPointerMode;
        this.cublasSgemv                = cublasSgemv;
        this.cublasHSSgemvStridedBatched = cublasHSSgemvStridedBatched;
    }

    // ── Device memory helpers ─────────────────────────────────────────────────

    /**
     * Allocates {@code bytes} of device memory on {@code deviceIndex} and returns
     * a {@link MemorySegment} wrapping the device pointer.
     */
    MemorySegment deviceMalloc(int deviceIndex, long bytes) {
        check(callInt(cudaSetDevice, deviceIndex), "cudaSetDevice");
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment slot = tmp.allocate(ADDRESS);
            check(callInt(cudaMalloc, slot, bytes), "cudaMalloc");
            return slot.get(ADDRESS, 0).reinterpret(bytes);
        }
    }

    /**
     * Frees a device pointer returned by {@link #deviceMalloc}.
     */
    void deviceFree(MemorySegment devicePtr) {
        if (devicePtr == null || devicePtr.equals(MemorySegment.NULL)) return;
        callInt(cudaFree, devicePtr);
    }

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

    // ── Shared Panama helpers (used by subclasses) ────────────────────────────

    protected static SymbolLookup loadLibrary(String... candidates) {
        for (String name : candidates) {
            try {
                return SymbolLookup.libraryLookup(name, Arena.global());
            } catch (IllegalArgumentException ignored) { /* try next */ }
        }
        throw new IllegalStateException(
            "Could not load GPU library — tried: " + java.util.Arrays.toString(candidates));
    }

    protected static MethodHandle bind(Linker linker, SymbolLookup lib,
                                        String symbol, FunctionDescriptor desc) {
        MemorySegment addr = lib.find(symbol)
            .orElseThrow(() -> new IllegalStateException("GPU symbol not found: " + symbol));
        return linker.downcallHandle(addr, desc);
    }
}
