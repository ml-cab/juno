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
 * <p>Implements {@link GpuBindings} — accessor methods expose the existing
 * {@link MethodHandle} fields to vendor-neutral callers. No existing fields or
 * constants have been removed; only {@code implements GpuBindings} and the
 * corresponding {@code @Override} methods are new.
 *
 * <p>Requires JVM flag: {@code --enable-native-access=ALL-UNNAMED}.
 *
 * <p>Struct layout constants ({@link #DEVICE_PROP_BYTES}, {@link #PROP_NAME_OFFSET},
 * {@link #PROP_TOTAL_MEM_OFFSET}) reflect {@code cudaDeviceProp} as laid out
 * by CUDA 12.x on Linux x86_64. If the CUDA major version changes, verify
 * these offsets against the SDK headers.
 */
final class CudaBindings implements GpuBindings {

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
    /** {@code sizeof(cudaDeviceProp)} in bytes. */
    static final int  DEVICE_PROP_BYTES     = 1512;
    /** Byte offset of {@code char name[256]} inside {@code cudaDeviceProp}. */
    static final long PROP_NAME_OFFSET      = 0L;
    /** Byte offset of {@code size_t totalGlobalMem} inside {@code cudaDeviceProp}. */
    static final long PROP_TOTAL_MEM_OFFSET = 288L;

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

    /** Returns {@code true} if CUDA and cuBLAS libraries were resolved successfully. */
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

    // ── cudart handles ────────────────────────────────────────────────────────
    final MethodHandle cudaGetDeviceCount;        // int (int*)
    final MethodHandle cudaGetDeviceProperties;   // int (cudaDeviceProp*, int)
    final MethodHandle cudaSetDevice;             // int (int)
    final MethodHandle cudaMalloc;                // int (void**, size_t)
    final MethodHandle cudaFree;                  // int (void*)
    final MethodHandle cudaMallocHost;            // int (void**, size_t)
    final MethodHandle cudaFreeHost;              // int (void*)
    final MethodHandle cudaMemcpy;                // int (void*, void*, size_t, int)
    final MethodHandle cudaMemcpyAsync;           // int (void*, void*, size_t, int, cudaStream_t)
    final MethodHandle cudaStreamCreateWithFlags; // int (cudaStream_t*, unsigned int)
    final MethodHandle cudaStreamSynchronize;     // int (cudaStream_t)
    final MethodHandle cudaStreamDestroy;         // int (cudaStream_t)

    // ── cuBLAS handles ────────────────────────────────────────────────────────
    final MethodHandle cublasCreate;              // int (cublasHandle_t*)
    final MethodHandle cublasDestroy;             // int (cublasHandle_t)
    final MethodHandle cublasSetStream;           // int (cublasHandle_t, cudaStream_t)
    final MethodHandle cublasSetPointerMode;      // int (cublasHandle_t, int)
    final MethodHandle cublasSgemv;               // FP32 GEMV
    final MethodHandle cublasHSSgemvStridedBatched; // FP16 A+x, FP32 y, batched

    // ── Construction ──────────────────────────────────────────────────────────

    private CudaBindings() {
        Linker       l    = Linker.nativeLinker();
        SymbolLookup rt   = GpuBindings.loadLibrary("libcudart.so.12", "libcudart.so");
        SymbolLookup blas = GpuBindings.loadLibrary("libcublas.so.12", "libcublas.so");

        cudaGetDeviceCount        = GpuBindings.bind(l, rt,   "cudaGetDeviceCount",
            FunctionDescriptor.of(JAVA_INT, ADDRESS));
        cudaGetDeviceProperties   = GpuBindings.bind(l, rt,   "cudaGetDeviceProperties",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));
        cudaSetDevice             = GpuBindings.bind(l, rt,   "cudaSetDevice",
            FunctionDescriptor.of(JAVA_INT, JAVA_INT));
        cudaMalloc                = GpuBindings.bind(l, rt,   "cudaMalloc",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG));
        cudaFree                  = GpuBindings.bind(l, rt,   "cudaFree",
            FunctionDescriptor.of(JAVA_INT, ADDRESS));
        cudaMallocHost            = GpuBindings.bind(l, rt,   "cudaMallocHost",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG));
        cudaFreeHost              = GpuBindings.bind(l, rt,   "cudaFreeHost",
            FunctionDescriptor.of(JAVA_INT, ADDRESS));
        cudaMemcpy                = GpuBindings.bind(l, rt,   "cudaMemcpy",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT));
        cudaMemcpyAsync           = GpuBindings.bind(l, rt,   "cudaMemcpyAsync",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT, ADDRESS));
        cudaStreamCreateWithFlags = GpuBindings.bind(l, rt,   "cudaStreamCreateWithFlags",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));
        cudaStreamSynchronize     = GpuBindings.bind(l, rt,   "cudaStreamSynchronize",
            FunctionDescriptor.of(JAVA_INT, ADDRESS));
        cudaStreamDestroy         = GpuBindings.bind(l, rt,   "cudaStreamDestroy",
            FunctionDescriptor.of(JAVA_INT, ADDRESS));
        cublasCreate              = GpuBindings.bind(l, blas,  "cublasCreate_v2",
            FunctionDescriptor.of(JAVA_INT, ADDRESS));
        cublasDestroy             = GpuBindings.bind(l, blas,  "cublasDestroy_v2",
            FunctionDescriptor.of(JAVA_INT, ADDRESS));
        cublasSetStream           = GpuBindings.bind(l, blas,  "cublasSetStream_v2",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
        cublasSetPointerMode      = GpuBindings.bind(l, blas,  "cublasSetPointerMode_v2",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));
        cublasSgemv               = GpuBindings.bind(l, blas,  "cublasSgemv_v2",
            FunctionDescriptor.of(JAVA_INT,
                ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT,
                ADDRESS, ADDRESS, JAVA_INT,
                ADDRESS, JAVA_INT,
                ADDRESS, ADDRESS, JAVA_INT));
        cublasHSSgemvStridedBatched = GpuBindings.bind(l, blas, "cublasHSSgemvStridedBatched",
            FunctionDescriptor.of(JAVA_INT,
                ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT,
                ADDRESS, ADDRESS, JAVA_INT, JAVA_LONG,
                ADDRESS, JAVA_INT, JAVA_LONG,
                ADDRESS, ADDRESS, JAVA_INT, JAVA_LONG,
                JAVA_INT));
        log.info("CudaBindings ready — Panama FFI (cudart + cublas)");
    }

    // ── Static utilities (kept for backward compatibility) ────────────────────

    /** @see GpuBindings#check(int, String) */
    static void check(int rc, String op) { GpuBindings.check(rc, op); }

    /** @see GpuBindings#callInt(MethodHandle, Object...) */
    static int callInt(MethodHandle mh, Object... args) { return GpuBindings.callInt(mh, args); }

    // ── Device memory ─────────────────────────────────────────────────────────

    @Override
    public MemorySegment deviceMalloc(int deviceIndex, long bytes) {
        GpuBindings.check(GpuBindings.callInt(cudaSetDevice, deviceIndex), "cudaSetDevice");
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment slot = tmp.allocate(ADDRESS);
            GpuBindings.check(GpuBindings.callInt(cudaMalloc, slot, bytes), "cudaMalloc");
            return slot.get(ADDRESS, 0).reinterpret(bytes);
        }
    }

    @Override
    public void deviceFree(MemorySegment devicePtr) {
        if (devicePtr == null || devicePtr.equals(MemorySegment.NULL)) return;
        GpuBindings.callInt(cudaFree, devicePtr);
    }

    // ── GpuBindings accessors ─────────────────────────────────────────────────

    @Override public MethodHandle cudaGetDeviceCount()          { return cudaGetDeviceCount; }
    @Override public MethodHandle cudaGetDeviceProperties()     { return cudaGetDeviceProperties; }
    @Override public MethodHandle cudaSetDevice()               { return cudaSetDevice; }
    @Override public MethodHandle cudaMalloc()                  { return cudaMalloc; }
    @Override public MethodHandle cudaFree()                    { return cudaFree; }
    @Override public MethodHandle cudaMallocHost()              { return cudaMallocHost; }
    @Override public MethodHandle cudaFreeHost()                { return cudaFreeHost; }
    @Override public MethodHandle cudaMemcpy()                  { return cudaMemcpy; }
    @Override public MethodHandle cudaMemcpyAsync()             { return cudaMemcpyAsync; }
    @Override public MethodHandle cudaStreamCreateWithFlags()   { return cudaStreamCreateWithFlags; }
    @Override public MethodHandle cudaStreamSynchronize()       { return cudaStreamSynchronize; }
    @Override public MethodHandle cudaStreamDestroy()           { return cudaStreamDestroy; }
    @Override public MethodHandle cublasCreate()                { return cublasCreate; }
    @Override public MethodHandle cublasDestroy()               { return cublasDestroy; }
    @Override public MethodHandle cublasSetStream()             { return cublasSetStream; }
    @Override public MethodHandle cublasSetPointerMode()        { return cublasSetPointerMode; }
    @Override public MethodHandle cublasSgemv()                 { return cublasSgemv; }
    @Override public MethodHandle cublasHSSgemvStridedBatched() { return cublasHSSgemvStridedBatched; }
    @Override public int    opTranspose()       { return CUBLAS_OP_T; }
    @Override public int    pointerModeHost()   { return CUBLAS_POINTER_MODE_HOST; }
    @Override public int    devicePropBytes()   { return DEVICE_PROP_BYTES; }
    @Override public long   propNameOffset()    { return PROP_NAME_OFFSET; }
    @Override public long   propTotalMemOffset(){ return PROP_TOTAL_MEM_OFFSET; }
    @Override public String backendLabel()      { return "cuda"; }
}
