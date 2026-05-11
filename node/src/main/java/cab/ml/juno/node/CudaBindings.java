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
import java.util.Arrays;
import java.util.logging.Logger;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Panama FFI downcall handles for the CUDA Runtime (cudart) and cuBLAS.
 *
 * <p>Replaces the JavaCPP / bytedeco bridge entirely. All CUDA symbols are
 * resolved once at class-init time via {@link Linker} and
 * {@link SymbolLookup}; the resulting {@link MethodHandle} instances are
 * thread-safe and carry zero per-call Java overhead beyond argument boxing
 * (which the JIT eliminates for typed {@code invokeExact} call sites).
 *
 * <p>Library names tried in order on Linux: {@code libcudart.so.12},
 * {@code libcudart.so} and the cuBLAS equivalents. The CUDA installation must
 * be on {@code LD_LIBRARY_PATH} (or {@code CUDA_PATH/lib64} via the launcher).
 *
 * <p>Requires JVM flag: {@code --enable-native-access=ALL-UNNAMED}.
 *
 * <p>Struct layout constants ({@link #DEVICE_PROP_BYTES}, {@link #PROP_NAME_OFFSET},
 * {@link #PROP_TOTAL_MEM_OFFSET}) reflect {@code cudaDeviceProp} as laid out
 * by CUDA 12.x on Linux x86_64. If the CUDA major version changes, verify
 * these offsets against the SDK headers.
 */
final class CudaBindings {

    private static final Logger log = Logger.getLogger(CudaBindings.class.getName());

    // ── cudaMemcpyKind ────────────────────────────────────────────────────────
    static final int H2D = 1; // cudaMemcpyHostToDevice
    static final int D2H = 2; // cudaMemcpyDeviceToHost

    // ── cublasOperation_t ─────────────────────────────────────────────────────
    static final int CUBLAS_OP_T = 1;

    // ── cublasPointerMode_t ───────────────────────────────────────────────────
    static final int CUBLAS_POINTER_MODE_HOST = 0;

    // ── cudaStream_t non-blocking flag ────────────────────────────────────────
    static final int STREAM_NON_BLOCKING = 0x01;

    // ── cudaDeviceProp layout (CUDA 12.x, Linux x86_64) ──────────────────────
    /** {@code sizeof(cudaDeviceProp)} */
    static final int  DEVICE_PROP_BYTES     = 1512;
    /** Byte offset of {@code char name[256]} inside {@code cudaDeviceProp}. */
    static final long PROP_NAME_OFFSET      = 0;
    /** Byte offset of {@code size_t totalGlobalMem} inside {@code cudaDeviceProp}. */
    static final long PROP_TOTAL_MEM_OFFSET = 288;

    // ── cudart handles ────────────────────────────────────────────────────────
    final MethodHandle cudaGetDeviceCount;       // int (int*)
    final MethodHandle cudaGetDeviceProperties;  // int (cudaDeviceProp*, int)
    final MethodHandle cudaSetDevice;            // int (int)
    final MethodHandle cudaMalloc;               // int (void**, size_t)
    final MethodHandle cudaFree;                 // int (void*)
    final MethodHandle cudaMallocHost;           // int (void**, size_t)
    final MethodHandle cudaFreeHost;             // int (void*)
    final MethodHandle cudaMemcpy;               // int (void*, const void*, size_t, int)
    final MethodHandle cudaMemcpyAsync;          // int (void*, const void*, size_t, int, cudaStream_t)
    final MethodHandle cudaStreamCreateWithFlags;// int (cudaStream_t*, unsigned int)
    final MethodHandle cudaStreamSynchronize;    // int (cudaStream_t)
    final MethodHandle cudaStreamDestroy;        // int (cudaStream_t)

    // ── cuBLAS handles ────────────────────────────────────────────────────────
    final MethodHandle cublasCreate;             // int (cublasHandle_t*)
    final MethodHandle cublasDestroy;            // int (cublasHandle_t)
    final MethodHandle cublasSetStream;          // int (cublasHandle_t, cudaStream_t)
    final MethodHandle cublasSetPointerMode;     // int (cublasHandle_t, int)
    final MethodHandle cublasSgemv;              // int (handle,op,m,n,*α,*A,lda,*x,incx,*β,*y,incy)
    final MethodHandle cublasHSSgemvStridedBatched; // FP16 A+x, FP32 y, batched

    // ── Singleton init ────────────────────────────────────────────────────────
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

    /** True if the CUDA and cuBLAS libraries were resolved successfully. */
    static boolean isAvailable() {
        return INSTANCE != null;
    }

    /**
     * Returns the singleton instance.
     *
     * @throws IllegalStateException if CUDA libraries could not be loaded
     */
    static CudaBindings instance() {
        if (INSTANCE == null)
            throw new IllegalStateException(
                "CUDA not available: " + INIT_FAILURE.getMessage(), INIT_FAILURE);
        return INSTANCE;
    }

    // ── Private construction ──────────────────────────────────────────────────

    private CudaBindings() {
        Linker       linker = Linker.nativeLinker();
        SymbolLookup cudart = loadLibrary("libcudart.so.12", "libcudart.so");
        SymbolLookup cublas = loadLibrary("libcublas.so.12", "libcublas.so");

        cudaGetDeviceCount        = bind(linker, cudart, "cudaGetDeviceCount",
            FunctionDescriptor.of(JAVA_INT, ADDRESS));
        cudaGetDeviceProperties   = bind(linker, cudart, "cudaGetDeviceProperties",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));
        cudaSetDevice             = bind(linker, cudart, "cudaSetDevice",
            FunctionDescriptor.of(JAVA_INT, JAVA_INT));
        cudaMalloc                = bind(linker, cudart, "cudaMalloc",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG));
        cudaFree                  = bind(linker, cudart, "cudaFree",
            FunctionDescriptor.of(JAVA_INT, ADDRESS));
        cudaMallocHost            = bind(linker, cudart, "cudaMallocHost",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG));
        cudaFreeHost              = bind(linker, cudart, "cudaFreeHost",
            FunctionDescriptor.of(JAVA_INT, ADDRESS));
        cudaMemcpy                = bind(linker, cudart, "cudaMemcpy",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT));
        cudaMemcpyAsync           = bind(linker, cudart, "cudaMemcpyAsync",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT, ADDRESS));
        cudaStreamCreateWithFlags = bind(linker, cudart, "cudaStreamCreateWithFlags",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));
        cudaStreamSynchronize     = bind(linker, cudart, "cudaStreamSynchronize",
            FunctionDescriptor.of(JAVA_INT, ADDRESS));
        cudaStreamDestroy         = bind(linker, cudart, "cudaStreamDestroy",
            FunctionDescriptor.of(JAVA_INT, ADDRESS));

        cublasCreate              = bind(linker, cublas, "cublasCreate_v2",
            FunctionDescriptor.of(JAVA_INT, ADDRESS));
        cublasDestroy             = bind(linker, cublas, "cublasDestroy_v2",
            FunctionDescriptor.of(JAVA_INT, ADDRESS));
        cublasSetStream           = bind(linker, cublas, "cublasSetStream_v2",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
        cublasSetPointerMode      = bind(linker, cublas, "cublasSetPointerMode_v2",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));
        cublasSgemv               = bind(linker, cublas, "cublasSgemv_v2",
            FunctionDescriptor.of(JAVA_INT,
                ADDRESS,   // handle
                JAVA_INT,  // trans
                JAVA_INT,  // m
                JAVA_INT,  // n
                ADDRESS,   // *alpha
                ADDRESS,   // *A
                JAVA_INT,  // lda
                ADDRESS,   // *x
                JAVA_INT,  // incx
                ADDRESS,   // *beta
                ADDRESS,   // *y
                JAVA_INT));// incy
        cublasHSSgemvStridedBatched = bind(linker, cublas, "cublasHSSgemvStridedBatched",
            FunctionDescriptor.of(JAVA_INT,
                ADDRESS,   // handle
                JAVA_INT,  // trans
                JAVA_INT,  // m
                JAVA_INT,  // n
                ADDRESS,   // *alpha  (float)
                ADDRESS,   // *A      (__half, device)
                JAVA_INT,  // lda
                JAVA_LONG, // strideA
                ADDRESS,   // *x      (__half, device)
                JAVA_INT,  // incx
                JAVA_LONG, // strideX
                ADDRESS,   // *beta   (float)
                ADDRESS,   // *y      (float, device)
                JAVA_INT,  // incy
                JAVA_LONG, // strideY
                JAVA_INT));// batchCount

        log.info("CudaBindings ready — Panama FFI (cudart + cublas)");
    }

    // ── Device memory helpers (used by DeviceFloatMatrix, DeviceHalfMatrix, CudaMatVec) ──

    /**
     * Calls {@code cudaMalloc} and returns a {@link MemorySegment} sized to
     * {@code bytes} representing the device pointer.
     *
     * <p>The caller is responsible for freeing it via {@link #deviceFree}.
     */
    MemorySegment deviceMalloc(int deviceIndex, long bytes) {
        check(callInt(cudaSetDevice, deviceIndex), "cudaSetDevice");
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment slot = tmp.allocate(ADDRESS);
            check(callInt(cudaMalloc, slot, bytes), "cudaMalloc");
            // Extract the raw device address and annotate it with its byte size.
            // reinterpret is safe here: we know the exact allocation size.
            return slot.get(ADDRESS, 0).reinterpret(bytes);
        }
    }

    /**
     * Calls {@code cudaFree} on a device segment returned by {@link #deviceMalloc}.
     */
    void deviceFree(MemorySegment devicePtr) {
        if (devicePtr == null || devicePtr.equals(MemorySegment.NULL)) return;
        callInt(cudaFree, devicePtr);
    }

    // ── Package-private static utilities ─────────────────────────────────────

    /**
     * Invokes a CUDA downcall that returns {@code int} and throws on non-zero.
     */
    static void check(int rc, String op) {
        if (rc != 0)
            throw new IllegalStateException(op + " failed: rc=" + rc);
    }

    /**
     * Reflective invoke returning the {@code int} return value.
     * Used for non-hot-path calls where exact type is unknown at the call site.
     */
    static int callInt(MethodHandle mh, Object... args) {
        try {
            return (int) mh.invokeWithArguments(args);
        } catch (Throwable t) {
            throw new IllegalStateException("CUDA downcall failed", t);
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private static SymbolLookup loadLibrary(String... candidates) {
        for (String name : candidates) {
            try {
                return SymbolLookup.libraryLookup(name, Arena.global());
            } catch (IllegalArgumentException ignored) {
                // try next candidate
            }
        }
        throw new IllegalStateException(
            "Could not load CUDA library — tried: " + Arrays.toString(candidates)
            + ". Ensure CUDA 12.x is installed and libcudart.so.12 is on LD_LIBRARY_PATH.");
    }

    private static MethodHandle bind(Linker linker, SymbolLookup lib, String symbol, FunctionDescriptor desc) {
        MemorySegment addr = lib.find(symbol)
            .orElseThrow(() -> new IllegalStateException("CUDA symbol not found: " + symbol));
        return linker.downcallHandle(addr, desc);
    }
}