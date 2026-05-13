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
import java.util.logging.Logger;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * {@link MatVec} backed by {@code cublasSgemv_v2} on an Nvidia GPU, via Panama FFI.
 *
 * <p>All JNI / JavaCPP (bytedeco) dependencies have been replaced with
 * {@link CudaBindings} downcall handles. Native memory is managed exclusively
 * through {@link MemorySegment} and {@link Arena}:
 *
 * <ul>
 *   <li>Device weight matrices ({@link DeviceFloatMatrix}, {@link DeviceHalfMatrix})
 *       are uploaded once and held resident; their {@link MemorySegment} is passed
 *       directly to cuBLAS as an ADDRESS parameter — zero H2D copy per token.
 *   <li>Per-thread x and y scratch buffers on the device are grown lazily and
 *       reused across calls — one {@code cudaMalloc} per thread, per buffer.
 *   <li>H2D upload of x uses {@code MemorySegment.ofArray(x)}: Panama pins the
 *       heap array for the duration of the downcall; CUDA copies to a driver
 *       staging buffer before returning (pageable-host semantics).
 *   <li>D2H download of y uses a short-lived confined arena to avoid handing
 *       a GC-moveable address to an async CUDA stream.
 *   <li>FP16 x staging is packed with {@code Float.floatToFloat16} into a
 *       confined off-heap arena — no heap byte[] allocation in the hot path.
 * </ul>
 *
 * <p>Concurrency: the {@link GpuContext#cublasSerializationLock()} {@code
 * synchronized} block serializes stream-binding and kernel submission on the
 * shared cuBLAS handle. This causes carrier-thread pinning when virtual threads
 * are used. Migrate to {@code ReentrantLock} when addressing Loom pinning
 * (HPC audit point 4).
 *
 * <p>Requires JVM flag: {@code --enable-native-access=ALL-UNNAMED}.
 *
 * @author Yevhen Soldatov
 */
public final class CudaMatVec implements MatVec {

    @SuppressWarnings("unused")
    private static final Logger log = Logger.getLogger(CudaMatVec.class.getName());

    private static final int STREAM_NON_BLOCKING = CudaBindings.STREAM_NON_BLOCKING;

    private final GpuContext     ctx;
    private final CudaBindings   cuda;

    // ── Per-thread device scratch (FP32 resident path) ────────────────────────
    private static final ThreadLocal<Fp32Scratch> FP32_SCRATCH =
        ThreadLocal.withInitial(Fp32Scratch::new);

    // ── Per-thread device scratch (FP16 resident path) ────────────────────────
    private static final ThreadLocal<Fp16Scratch> FP16_SCRATCH =
        ThreadLocal.withInitial(Fp16Scratch::new);

    // ── Per-thread CUDA stream ────────────────────────────────────────────────
    private static final ThreadLocal<MemorySegment> CUDA_STREAM =
        ThreadLocal.withInitial(() -> null);

    // ── Scratch containers ────────────────────────────────────────────────────

    private static final class Fp32Scratch {
        MemorySegment dX;   // device, grown as needed
        MemorySegment dY;   // device, grown as needed
        long dXBytes;
        long dYBytes;
    }

    private static final class Fp16Scratch {
        MemorySegment dXh;  // device FP16 x, grown as needed
        MemorySegment dY;   // device FP32 y, grown as needed
        long dXhBytes;
        long dYBytes;
    }

    // ── Construction ──────────────────────────────────────────────────────────

    /**
     * @param ctx an open GpuContext — must outlive all sgemv calls on this instance
     */
    public CudaMatVec(GpuContext ctx) {
        if (ctx == null) throw new IllegalArgumentException("ctx must not be null");
        this.ctx  = ctx;
        this.cuda = CudaBindings.instance();
    }

    GpuContext gpuContext() { return ctx; }

    // ── Upload helpers (for LlamaTransformerHandler / LoraTrainableHandler) ──

    DeviceFloatMatrix upload(float[] host, int rows, int cols) {
        return DeviceFloatMatrix.upload(ctx, host, rows, cols);
    }

    DeviceHalfMatrix uploadHalf(float[] host, int rows, int cols) {
        return DeviceHalfMatrix.uploadFromFloat32(ctx, host, rows, cols);
    }

    // ── MatVec ────────────────────────────────────────────────────────────────

    /**
     * Full host path: A and x are on the host.
     *
     * Allocates temporary device buffers for A, x, and y; performs a
     * synchronous H2D copy of A and x; runs {@code cublasSgemv_v2}; performs
     * a synchronous D2H copy of y; frees the temporary buffers.
     *
     * Intended for tests and the rare non-resident fallback. The hot inference
     * path uses {@link #sgemv(DeviceFloatMatrix, float[])} with device-resident A.
     */
    @Override
    public float[] sgemv(float[] A, float[] x, int rows, int cols) {
        if (A.length != (long) rows * cols)
            throw new IllegalArgumentException("A.length=" + A.length + " != rows*cols=" + ((long) rows * cols));
        if (x.length != cols)
            throw new IllegalArgumentException("x.length=" + x.length + " != cols=" + cols);

        MatVecEvent evt = new MatVecEvent();
        evt.begin();

        long bytesA = (long) rows * cols * Float.BYTES;
        long bytesX = (long) cols  * Float.BYTES;
        long bytesY = (long) rows  * Float.BYTES;

        MemorySegment dA = cuda.deviceMalloc(ctx.deviceIndex(), bytesA);
        MemorySegment dX = cuda.deviceMalloc(ctx.deviceIndex(), bytesX);
        MemorySegment dY = cuda.deviceMalloc(ctx.deviceIndex(), bytesY);
        try {
            // H2D — synchronous cudaMemcpy; Panama pins the heap arrays during the call.
            CudaBindings.check(
                CudaBindings.callInt(cuda.cudaMemcpy, dA, MemorySegment.ofArray(A), bytesA, CudaBindings.H2D),
                "cudaMemcpy(A H2D)");
            CudaBindings.check(
                CudaBindings.callInt(cuda.cudaMemcpy, dX, MemorySegment.ofArray(x), bytesX, CudaBindings.H2D),
                "cudaMemcpy(x H2D)");

            float[] y = new float[rows];
            synchronized (ctx.cublasSerializationLock()) {
                callSgemvFp32(dA, cols, dX, dY, rows, cols);
                // Synchronous D2H: heap array is safe here (cudaMemcpy blocks until done).
                CudaBindings.check(
                    CudaBindings.callInt(cuda.cudaMemcpy, MemorySegment.ofArray(y), dY, bytesY, CudaBindings.D2H),
                    "cudaMemcpy(y D2H)");
            }
            return y;
        } finally {
            cuda.deviceFree(dA);
            cuda.deviceFree(dX);
            cuda.deviceFree(dY);
            evt.backend = "cuda";
            evt.rows = rows;
            evt.cols = cols;
            evt.commit();
        }
    }

    /**
     * Device-resident FP32 path: A stays on the device across calls.
     *
     * Per-thread scratch buffers for x and y are grown lazily and reused.
     * The D2H copy uses a confined off-heap arena to avoid exposing a
     * GC-moveable address to the async CUDA stream.
     */
    @Override
    public float[] sgemv(DeviceFloatMatrix A, float[] x) {
        if (A == null) throw new IllegalArgumentException("A must not be null");
        if (A.isClosed()) throw new IllegalStateException("DeviceFloatMatrix is closed");
        int rows = A.rows(), cols = A.cols();
        if (x.length != cols)
            throw new IllegalArgumentException("x.length=" + x.length + " != cols=" + cols);

        MatVecEvent evt = new MatVecEvent();
        evt.begin();

        long bytesX = (long) cols * Float.BYTES;
        long bytesY = (long) rows * Float.BYTES;

        Fp32Scratch scratch = FP32_SCRATCH.get();

        try (Arena resultArena = Arena.ofConfined()) {
            synchronized (ctx.cublasSerializationLock()) {
                MemorySegment stream = ensureStream();
                bindStream(stream);
                try {
                    ensureFp32Scratch(scratch, bytesX, bytesY);

                    // H2D: Panama pins the heap array for the duration of this downcall.
                    CudaBindings.check(
                        CudaBindings.callInt(cuda.cudaMemcpyAsync,
                            scratch.dX, MemorySegment.ofArray(x), bytesX, CudaBindings.H2D, stream),
                        "cudaMemcpyAsync(x H2D)");

                    callSgemvFp32(A.devicePointer(), cols, scratch.dX, scratch.dY, rows, cols);

                    // D2H into off-heap staging — the async copy must not target a moveable heap address.
                    MemorySegment stagingY = resultArena.allocate(bytesY);
                    CudaBindings.check(
                        CudaBindings.callInt(cuda.cudaMemcpyAsync,
                            stagingY, scratch.dY, bytesY, CudaBindings.D2H, stream),
                        "cudaMemcpyAsync(y D2H)");
                    CudaBindings.check(
                        CudaBindings.callInt(cuda.cudaStreamSynchronize, stream),
                        "cudaStreamSynchronize");

                    float[] y = new float[rows];
                    MemorySegment.copy(stagingY, JAVA_FLOAT, 0, y, 0, rows);
                    return y;
                } finally {
                    unbindStream();
                }
            }
        } finally {
            evt.backend = "cuda-resident";
            evt.rows = rows;
            evt.cols = cols;
            evt.commit();
        }
    }

    /**
     * Device-resident FP16 path: A is FP16 on the device; x is FP32 on the host.
     *
     * x is converted to FP16 in a confined off-heap arena and uploaded; the
     * cuBLAS mixed-precision kernel accumulates in FP32.
     */
    @Override
    public float[] sgemv(DeviceHalfMatrix A, float[] x) {
        if (A == null) throw new IllegalArgumentException("A must not be null");
        if (A.isClosed()) throw new IllegalStateException("DeviceHalfMatrix is closed");
        int rows = A.rows(), cols = A.cols();
        if (x.length != cols)
            throw new IllegalArgumentException("x.length=" + x.length + " != cols=" + cols);

        MatVecEvent evt = new MatVecEvent();
        evt.begin();

        long bytesXh = (long) cols * Short.BYTES;  // FP16
        long bytesY  = (long) rows * Float.BYTES;  // FP32

        Fp16Scratch scratch = FP16_SCRATCH.get();

        try (Arena callArena = Arena.ofConfined()) {
            synchronized (ctx.cublasSerializationLock()) {
                MemorySegment stream = ensureStream();
                bindStream(stream);
                try {
                    ensureFp16Scratch(scratch, bytesXh, bytesY);

                    // Pack x as FP16 into off-heap staging. JAVA_SHORT has native byte order
                    // (little-endian on x86), matching CUDA __half layout.
                    MemorySegment stagingXh = callArena.allocate(bytesXh);
                    for (int j = 0; j < cols; j++)
                        stagingXh.setAtIndex(JAVA_SHORT, j, Float.floatToFloat16(x[j]));

                    CudaBindings.check(
                        CudaBindings.callInt(cuda.cudaMemcpyAsync,
                            scratch.dXh, stagingXh, bytesXh, CudaBindings.H2D, stream),
                        "cudaMemcpyAsync(xh H2D)");

                    callSgemvFp16(A.devicePointer(), cols, scratch.dXh, scratch.dY, rows, cols);

                    MemorySegment stagingY = callArena.allocate(bytesY);
                    CudaBindings.check(
                        CudaBindings.callInt(cuda.cudaMemcpyAsync,
                            stagingY, scratch.dY, bytesY, CudaBindings.D2H, stream),
                        "cudaMemcpyAsync(y D2H)");
                    CudaBindings.check(
                        CudaBindings.callInt(cuda.cudaStreamSynchronize, stream),
                        "cudaStreamSynchronize");

                    float[] y = new float[rows];
                    MemorySegment.copy(stagingY, JAVA_FLOAT, 0, y, 0, rows);
                    return y;
                } finally {
                    unbindStream();
                }
            }
        } finally {
            evt.backend = "cuda-resident-fp16";
            evt.rows = rows;
            evt.cols = cols;
            evt.commit();
        }
    }

    // ── cuBLAS kernel dispatchers ─────────────────────────────────────────────

    /**
     * cublasSgemv_v2: y = A * x, row-major A[rows×cols].
     *
     * cuBLAS is column-major. A row-major A[rows×cols] equals the transpose of
     * a column-major A^T[cols×rows]. Calling with CUBLAS_OP_T, m=cols, n=rows,
     * lda=cols computes y = A * x correctly.
     */
    private void callSgemvFp32(MemorySegment dA, int lda,
                                MemorySegment dX, MemorySegment dY,
                                int rows, int cols) {
        try (Arena scalars = Arena.ofConfined()) {
            MemorySegment alpha = scalars.allocateFrom(JAVA_FLOAT, 1.0f);
            MemorySegment beta  = scalars.allocateFrom(JAVA_FLOAT, 0.0f);
            CudaBindings.check(
                CudaBindings.callInt(cuda.cublasSetPointerMode, ctx.handle(), CudaBindings.CUBLAS_POINTER_MODE_HOST),
                "cublasSetPointerMode");
            CudaBindings.check(
                CudaBindings.callInt(cuda.cublasSgemv,
                    ctx.handle(), CudaBindings.CUBLAS_OP_T,
                    cols, rows,
                    alpha, dA, lda,
                    dX, 1,
                    beta, dY, 1),
                "cublasSgemv_v2");
        }
    }

    /**
     * cublasHSSgemvStridedBatched: y(FP32) = A(FP16) * x(FP16), batched=1.
     *
     * Same (trans, m, n, lda) mapping as {@link #callSgemvFp32}.
     */
    private void callSgemvFp16(MemorySegment dA, int lda,
                                MemorySegment dXh, MemorySegment dY,
                                int rows, int cols) {
        long strideA = (long) cols * rows;
        long strideX = cols;
        long strideY = rows;
        try (Arena scalars = Arena.ofConfined()) {
            MemorySegment alpha = scalars.allocateFrom(JAVA_FLOAT, 1.0f);
            MemorySegment beta  = scalars.allocateFrom(JAVA_FLOAT, 0.0f);
            CudaBindings.check(
                CudaBindings.callInt(cuda.cublasSetPointerMode, ctx.handle(), CudaBindings.CUBLAS_POINTER_MODE_HOST),
                "cublasSetPointerMode");
            CudaBindings.check(
                CudaBindings.callInt(cuda.cublasHSSgemvStridedBatched,
                    ctx.handle(), CudaBindings.CUBLAS_OP_T,
                    cols, rows,
                    alpha, dA, lda, strideA,
                    dXh, 1, strideX,
                    beta, dY, 1, strideY,
                    1),
                "cublasHSSgemvStridedBatched");
        }
    }

    // ── Stream management ─────────────────────────────────────────────────────

    /** Returns or lazily creates the per-thread non-blocking CUDA stream. */
    private MemorySegment ensureStream() {
        MemorySegment stream = CUDA_STREAM.get();
        if (stream != null) return stream;
        CudaBindings.check(
            CudaBindings.callInt(cuda.cudaSetDevice, ctx.deviceIndex()),
            "cudaSetDevice");
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment slot = tmp.allocate(ADDRESS);
            CudaBindings.check(
                CudaBindings.callInt(cuda.cudaStreamCreateWithFlags, slot, STREAM_NON_BLOCKING),
                "cudaStreamCreateWithFlags");
            stream = slot.get(ADDRESS, 0); // opaque 0-byte segment = stream handle
            CUDA_STREAM.set(stream);
            return stream;
        }
    }

    private void bindStream(MemorySegment stream) {
        CudaBindings.check(
            CudaBindings.callInt(cuda.cublasSetStream, ctx.handle(), stream),
            "cublasSetStream_v2");
    }

    /** Restores the default stream (NULL) on the cuBLAS handle. */
    private void unbindStream() {
        CudaBindings.callInt(cuda.cublasSetStream, ctx.handle(), MemorySegment.NULL);
    }

    // ── Scratch growth ────────────────────────────────────────────────────────

    private void ensureFp32Scratch(Fp32Scratch s, long bytesX, long bytesY) {
        int dev = ctx.deviceIndex();
        if (s.dXBytes < bytesX) {
            cuda.deviceFree(s.dX);
            s.dX     = cuda.deviceMalloc(dev, bytesX);
            s.dXBytes = bytesX;
        }
        if (s.dYBytes < bytesY) {
            cuda.deviceFree(s.dY);
            s.dY     = cuda.deviceMalloc(dev, bytesY);
            s.dYBytes = bytesY;
        }
    }

    private void ensureFp16Scratch(Fp16Scratch s, long bytesXh, long bytesY) {
        int dev = ctx.deviceIndex();
        if (s.dXhBytes < bytesXh) {
            cuda.deviceFree(s.dXh);
            s.dXh     = cuda.deviceMalloc(dev, bytesXh);
            s.dXhBytes = bytesXh;
        }
        if (s.dYBytes < bytesY) {
            cuda.deviceFree(s.dY);
            s.dY     = cuda.deviceMalloc(dev, bytesY);
            s.dYBytes = bytesY;
        }
    }
}