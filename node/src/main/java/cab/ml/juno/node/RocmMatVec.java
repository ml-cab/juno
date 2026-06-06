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
import java.lang.foreign.MemorySegment;
import java.util.logging.Logger;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * {@link MatVec} backed by {@code rocblas_sgemv} on an AMD GPU, via Panama FFI.
 *
 * <p>AMD/ROCm equivalent of {@link CudaMatVec}. Uses {@link RocmBindings}
 * (HIP runtime + rocBLAS) obtained through {@link GpuContext#bindings()}.
 * The {@link GpuContext} is created by {@link GpuContext#createMatVec()} when
 * an AMD GPU with ROCm 6+ is detected.
 *
 * <p>Host FP32 path: allocates temporary device buffers for A, x, y per call;
 * performs synchronous H2D copy of A and x; runs {@code rocblas_sgemv};
 * performs synchronous D2H copy of y; frees the temporary buffers.
 *
 * <p>Device-resident paths ({@link DeviceFloatMatrix}, {@link DeviceHalfMatrix})
 * keep A on the GPU across calls; only x and y cross the bus per matmul. The
 * device matrices allocate through the vendor-neutral {@link GpuBindings} from
 * {@link GpuContext#bindings()}, so they work identically on AMD and NVIDIA.
 *
 * <p>Per-thread HIP streams back the resident async path (same pattern as
 * {@link CudaMatVec}). The serialization lock from
 * {@link GpuContext#cublasSerializationLock()} guards rocBLAS handle usage.
 *
 * <p>Requires JVM flag: {@code --enable-native-access=ALL-UNNAMED}.
 *
 * @see GpuContext#createMatVec()
 * @see RocmBindings
 */
public final class RocmMatVec implements GpuMatVec {

    @SuppressWarnings("unused")
    private static final Logger log = Logger.getLogger(RocmMatVec.class.getName());

    private static final int STREAM_NON_BLOCKING = GpuBindings.STREAM_NON_BLOCKING;

    private final GpuContext  ctx;
    private final RocmBindings rocm;

    // ── Per-thread device scratch ─────────────────────────────────────────────
    // ── Per-thread device scratch (FP32 resident path) ────────────────────
    private static final ThreadLocal<Fp32Scratch> FP32_SCRATCH =
        ThreadLocal.withInitial(Fp32Scratch::new);

    // ── Per-thread device scratch (FP16 resident path) ────────────────────
    private static final ThreadLocal<Fp16Scratch> FP16_SCRATCH =
        ThreadLocal.withInitial(Fp16Scratch::new);

    // ── Per-thread HIP stream ───────────────────────────────────
    // TODO(#streams-leak): HIP streams are created lazily per-thread and held for the
    // thread's lifetime. On a Loom virtual-thread pool this is acceptable because
    // carrier threads are pooled, but if worker threads are terminated between requests
    // the HIP streams leak. The same issue exists in CudaMatVec.
    // Track: https://github.com/ml-cab/juno/issues/35
    private static final ThreadLocal<MemorySegment> HIP_STREAM =
        ThreadLocal.withInitial(() -> null);

    private static final class Fp32Scratch {
        MemorySegment dX;
        MemorySegment dY;
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
     * @param ctx an open {@link GpuContext} whose {@link GpuContext#bindings()}
     *            returns a {@link RocmBindings} instance — must outlive all
     *            sgemv calls on this instance
     */
    RocmMatVec(GpuContext ctx) {
        if (ctx == null) throw new IllegalArgumentException("ctx must not be null");
        GpuBindings b = ctx.bindings();
        if (!(b instanceof RocmBindings)) {
            throw new IllegalArgumentException(
                "RocmMatVec requires a ROCm GpuContext — got: " + b.backendLabel());
        }
        this.ctx  = ctx;
        this.rocm = (RocmBindings) b;
    }

    GpuContext gpuContext() { return ctx; }

    @Override
    public boolean supportsHalfResident() { return rocm.supportsHSSgemv(); }

    // ── Upload helpers (for transformer handlers) ───────────────────────

    @Override
    public DeviceFloatMatrix upload(float[] host, int rows, int cols) {
        return DeviceFloatMatrix.upload(ctx, host, rows, cols);
    }

    @Override
    public DeviceHalfMatrix uploadHalf(float[] host, int rows, int cols) {
        return DeviceHalfMatrix.uploadFromFloat32(ctx, host, rows, cols);
    }

    // ── MatVec ────────────────────────────────────────────────────────────────

    /**
     * Full host path: A and x are on the host.
     *
     * Allocates temporary device buffers for A, x, and y; performs synchronous
     * H2D copy of A and x; runs {@code rocblas_sgemv}; performs synchronous
     * D2H copy of y; frees all temporary buffers.
     */
    @Override
    public float[] sgemv(float[] A, float[] x, int rows, int cols) {
        if (A.length != (long) rows * cols)
            throw new IllegalArgumentException(
                "A.length=" + A.length + " != rows*cols=" + ((long) rows * cols));
        if (x.length != cols)
            throw new IllegalArgumentException(
                "x.length=" + x.length + " != cols=" + cols);

        MatVecEvent evt = new MatVecEvent();
        evt.begin();

        long bytesA = (long) rows * cols * Float.BYTES;
        long bytesX = (long) cols  * Float.BYTES;
        long bytesY = (long) rows  * Float.BYTES;

        // hipMalloc / hipFree interleaved with compute require serialization:
        // hipSetDevice inside deviceMalloc is a per-device global state write.
        synchronized (ctx.cublasSerializationLock()) {
        MemorySegment dA = rocm.deviceMalloc(ctx.deviceIndex(), bytesA);
        MemorySegment dX = rocm.deviceMalloc(ctx.deviceIndex(), bytesX);
        MemorySegment dY = rocm.deviceMalloc(ctx.deviceIndex(), bytesY);
        try {
            // H2D — copy Java heap arrays into native (off-heap) staging buffers first;
            // Panama FFI (Java 25) forbids passing heap-backed MemorySegments directly
            // to native downcalls ("Heap segment not allowed").
            try (Arena hostArena = Arena.ofConfined()) {
                MemorySegment nativeA = hostArena.allocate(bytesA);
                MemorySegment nativeX = hostArena.allocate(bytesX);
                nativeA.copyFrom(MemorySegment.ofArray(A)); // heap→native (Java copy, no FFI)
                nativeX.copyFrom(MemorySegment.ofArray(x));
                GpuBindings.check(
                    GpuBindings.callInt(rocm.gpuMemcpy(), dA, nativeA, bytesA, GpuBindings.H2D),
                    "hipMemcpy(A H2D)");
                GpuBindings.check(
                    GpuBindings.callInt(rocm.gpuMemcpy(), dX, nativeX, bytesX, GpuBindings.H2D),
                    "hipMemcpy(x H2D)");
            }

            // D2H — similarly, copy into native staging first, then into Java array.
            try (Arena resultArena = Arena.ofConfined()) {
                MemorySegment stagingY = resultArena.allocate(bytesY);
                callSgemvFp32(dA, cols, dX, dY, rows, cols);
                GpuBindings.check(
                    GpuBindings.callInt(rocm.gpuMemcpy(), stagingY, dY, bytesY, GpuBindings.D2H),
                    "hipMemcpy(y D2H)");
                float[] y = new float[rows];
                MemorySegment.copy(stagingY, JAVA_FLOAT, 0, y, 0, rows);
                return y;
            }
        } finally {
            rocm.deviceFree(dA);
            rocm.deviceFree(dX);
            rocm.deviceFree(dY);
            evt.backend(MatVecBackend.ROCM);
            evt.rows = rows;
            evt.cols = cols;
            evt.commit();
        }
        }
    }

    /**
     * Device-resident FP32 path: A stays on the device across calls.
     *
     * Per-thread scratch buffers for x and y are grown lazily and reused. The
     * D2H copy uses a confined off-heap arena so the async HIP stream never
     * targets a GC-moveable heap address. Mirrors
     * {@link CudaMatVec#sgemv(DeviceFloatMatrix, float[])}.
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

        try (Arena callArena = Arena.ofConfined()) {
            synchronized (ctx.cublasSerializationLock()) {
                MemorySegment stream = ensureStream();
                bindStream(stream);
                try {
                    ensureFp32Scratch(scratch, bytesX, bytesY);

                    // H2D of x — stage heap→native first (Java 25 forbids heap segments in downcalls).
                    MemorySegment stagingX = callArena.allocate(bytesX);
                    stagingX.copyFrom(MemorySegment.ofArray(x));
                    GpuBindings.check(
                        GpuBindings.callInt(rocm.gpuMemcpyAsync(),
                            scratch.dX, stagingX, bytesX, GpuBindings.H2D, stream),
                        "hipMemcpyAsync(x H2D)");

                    callSgemvFp32(A.devicePointer(), cols, scratch.dX, scratch.dY, rows, cols);

                    MemorySegment stagingY = callArena.allocate(bytesY);
                    GpuBindings.check(
                        GpuBindings.callInt(rocm.gpuMemcpyAsync(),
                            stagingY, scratch.dY, bytesY, GpuBindings.D2H, stream),
                        "hipMemcpyAsync(y D2H)");
                    GpuBindings.check(
                        GpuBindings.callInt(rocm.gpuStreamSynchronize(), stream),
                        "hipStreamSynchronize");

                    float[] y = new float[rows];
                    MemorySegment.copy(stagingY, JAVA_FLOAT, 0, y, 0, rows);
                    return y;
                } finally {
                    unbindStream();
                }
            }
        } finally {
            evt.backend(MatVecBackend.ROCM_RESIDENT);
            evt.rows = rows;
            evt.cols = cols;
            evt.commit();
        }
    }

    /**
     * Device-resident FP16 path: A is FP16 on the device; x is FP32 on the host.
     *
     * x is converted to FP16 in a confined off-heap arena and uploaded; the
     * {@code rocblas_hssgemv_strided_batched} kernel (batch=1) accumulates in
     * FP32. Mirrors {@link CudaMatVec#sgemv(DeviceHalfMatrix, float[])}.
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
                    // (little-endian on x86), matching HIP's __half layout.
                    MemorySegment stagingXh = callArena.allocate(bytesXh);
                    for (int j = 0; j < cols; j++)
                        stagingXh.setAtIndex(JAVA_SHORT, j, Float.floatToFloat16(x[j]));

                    GpuBindings.check(
                        GpuBindings.callInt(rocm.gpuMemcpyAsync(),
                            scratch.dXh, stagingXh, bytesXh, GpuBindings.H2D, stream),
                        "hipMemcpyAsync(xh H2D)");

                    callSgemvFp16(A.devicePointer(), cols, scratch.dXh, scratch.dY, rows, cols);

                    MemorySegment stagingY = callArena.allocate(bytesY);
                    GpuBindings.check(
                        GpuBindings.callInt(rocm.gpuMemcpyAsync(),
                            stagingY, scratch.dY, bytesY, GpuBindings.D2H, stream),
                        "hipMemcpyAsync(y D2H)");
                    GpuBindings.check(
                        GpuBindings.callInt(rocm.gpuStreamSynchronize(), stream),
                        "hipStreamSynchronize");

                    float[] y = new float[rows];
                    MemorySegment.copy(stagingY, JAVA_FLOAT, 0, y, 0, rows);
                    return y;
                } finally {
                    unbindStream();
                }
            }
        } finally {
            evt.backend(MatVecBackend.ROCM_RESIDENT_FP16);
            evt.rows = rows;
            evt.cols = cols;
            evt.commit();
        }
    }

    // ── rocBLAS GEMV ──────────────────────────────────────────────────────────

    /**
     * rocblas_sgemv: y(FP32) = A(FP32) * x(FP32).
     *
     * <p>Uses {@code rocblas_operation_transpose} (op=112) because A is stored
     * row-major [rows×cols] but rocBLAS expects column-major, so we pass
     * op=T and swap (m, n) → (cols, rows), lda=cols. This is identical to
     * the cuBLAS transpose trick in {@link CudaMatVec}.
     */
    private void callSgemvFp32(MemorySegment dA, int lda,
                                MemorySegment dX, MemorySegment dY,
                                int rows, int cols) {
        try (Arena scalars = Arena.ofConfined()) {
            MemorySegment alpha = scalars.allocateFrom(JAVA_FLOAT, 1.0f);
            MemorySegment beta  = scalars.allocateFrom(JAVA_FLOAT, 0.0f);
            GpuBindings.check(
                GpuBindings.callInt(rocm.blasSetPointerMode(), ctx.handle(), rocm.pointerModeHost()),
                "rocblas_set_pointer_mode");
            GpuBindings.check(
                GpuBindings.callInt(rocm.blasSgemv(),
                    ctx.handle(), rocm.opTranspose(),
                    cols, rows,
                    alpha, dA, lda,
                    dX, 1,
                    beta, dY, 1),
                "rocblas_sgemv");
        }
    }

    /**
     * rocblas_hssgemv_strided_batched: y(FP32) = A(FP16) * x(FP16), batch=1.
     *
     * <p>Same (trans, m, n, lda) mapping as {@link #callSgemvFp32}; alpha/beta
     * are FP32 (host pointer mode). This is the rocBLAS analogue of cuBLAS's
     * {@code cublasHSSgemvStridedBatched}.
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
            GpuBindings.check(
                GpuBindings.callInt(rocm.blasSetPointerMode(), ctx.handle(), rocm.pointerModeHost()),
                "rocblas_set_pointer_mode");
            GpuBindings.check(
                GpuBindings.callInt(rocm.blasHSSgemvStridedBatched(),
                    ctx.handle(), rocm.opTranspose(),
                    cols, rows,
                    alpha, dA, lda, strideA,
                    dXh, 1, strideX,
                    beta, dY, 1, strideY,
                    1),
                "rocblas_hssgemv_strided_batched");
        }
    }

    // ── Stream management ─────────────────────────────────────────────────────

    /** Returns or lazily creates the per-thread non-blocking HIP stream. */
    private MemorySegment ensureStream() {
        MemorySegment stream = HIP_STREAM.get();
        if (stream != null) return stream;
        GpuBindings.check(
            GpuBindings.callInt(rocm.gpuSetDevice(), ctx.deviceIndex()),
            "hipSetDevice");
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment slot = tmp.allocate(ADDRESS);
            GpuBindings.check(
                GpuBindings.callInt(rocm.gpuStreamCreateWithFlags(), slot, STREAM_NON_BLOCKING),
                "hipStreamCreateWithFlags");
            stream = slot.get(ADDRESS, 0);
            HIP_STREAM.set(stream);
            return stream;
        }
    }

    private void bindStream(MemorySegment stream) {
        GpuBindings.check(
            GpuBindings.callInt(rocm.blasSetStream(), ctx.handle(), stream),
            "rocblas_set_stream");
    }

    /** Restores the default stream (NULL) on the rocBLAS handle. */
    private void unbindStream() {
        GpuBindings.callInt(rocm.blasSetStream(), ctx.handle(), MemorySegment.NULL);
    }

    // ── Scratch growth ────────────────────────────────────────────────────────

    private void ensureFp32Scratch(Fp32Scratch s, long bytesX, long bytesY) {
        int dev = ctx.deviceIndex();
        if (s.dXBytes < bytesX) {
            rocm.deviceFree(s.dX);
            s.dX     = rocm.deviceMalloc(dev, bytesX);
            s.dXBytes = bytesX;
        }
        if (s.dYBytes < bytesY) {
            rocm.deviceFree(s.dY);
            s.dY     = rocm.deviceMalloc(dev, bytesY);
            s.dYBytes = bytesY;
        }
    }

    private void ensureFp16Scratch(Fp16Scratch s, long bytesXh, long bytesY) {
        int dev = ctx.deviceIndex();
        if (s.dXhBytes < bytesXh) {
            rocm.deviceFree(s.dXh);
            s.dXh     = rocm.deviceMalloc(dev, bytesXh);
            s.dXhBytes = bytesXh;
        }
        if (s.dYBytes < bytesY) {
            rocm.deviceFree(s.dY);
            s.dY     = rocm.deviceMalloc(dev, bytesY);
            s.dYBytes = bytesY;
        }
    }
}