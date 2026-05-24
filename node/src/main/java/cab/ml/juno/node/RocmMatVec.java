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
 * are not yet supported — those classes internally call {@link CudaBindings} for
 * memory management. Use the host-weight path ({@code sgemv(float[], float[], int, int)})
 * or contribute {@code RocmDeviceFloatMatrix} / {@code RocmDeviceHalfMatrix} analogues.
 *
 * <p>Per-thread HIP streams are used for the resident-A async path (same
 * pattern as {@link CudaMatVec}). The serialization lock from
 * {@link GpuContext#cublasSerializationLock()} guards rocBLAS handle usage.
 *
 * <p>Requires JVM flag: {@code --enable-native-access=ALL-UNNAMED}.
 *
 * @see GpuContext#createMatVec()
 * @see RocmBindings
 */
public final class RocmMatVec implements MatVec {

    @SuppressWarnings("unused")
    private static final Logger log = Logger.getLogger(RocmMatVec.class.getName());

    private static final int STREAM_NON_BLOCKING = GpuBindings.STREAM_NON_BLOCKING;

    private final GpuContext  ctx;
    private final RocmBindings rocm;

    // ── Per-thread device scratch ─────────────────────────────────────────────
    private static final ThreadLocal<Scratch> SCRATCH =
        ThreadLocal.withInitial(Scratch::new);

    // ── Per-thread HIP stream ─────────────────────────────────────────────────
    private static final ThreadLocal<MemorySegment> HIP_STREAM =
        ThreadLocal.withInitial(() -> null);

    private static final class Scratch {
        MemorySegment dX;
        MemorySegment dY;
        long dXBytes;
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

        MemorySegment dA = rocm.deviceMalloc(ctx.deviceIndex(), bytesA);
        MemorySegment dX = rocm.deviceMalloc(ctx.deviceIndex(), bytesX);
        MemorySegment dY = rocm.deviceMalloc(ctx.deviceIndex(), bytesY);
        try {
            // H2D — synchronous hipMemcpy; Panama pins the heap arrays during the call.
            GpuBindings.check(
                GpuBindings.callInt(rocm.cudaMemcpy(), dA, MemorySegment.ofArray(A), bytesA, GpuBindings.H2D),
                "hipMemcpy(A H2D)");
            GpuBindings.check(
                GpuBindings.callInt(rocm.cudaMemcpy(), dX, MemorySegment.ofArray(x), bytesX, GpuBindings.H2D),
                "hipMemcpy(x H2D)");

            float[] y = new float[rows];
            synchronized (ctx.cublasSerializationLock()) {
                callSgemvFp32(dA, cols, dX, dY, rows, cols);
                // Synchronous D2H: heap array is safe here (hipMemcpy blocks until done).
                GpuBindings.check(
                    GpuBindings.callInt(rocm.cudaMemcpy(), MemorySegment.ofArray(y), dY, bytesY, GpuBindings.D2H),
                    "hipMemcpy(y D2H)");
            }
            return y;
        } finally {
            rocm.deviceFree(dA);
            rocm.deviceFree(dX);
            rocm.deviceFree(dY);
            evt.backend = "rocm";
            evt.rows = rows;
            evt.cols = cols;
            evt.commit();
        }
    }

    /**
     * Device-resident FP32 path — not yet supported on ROCm.
     *
     * {@link DeviceFloatMatrix} allocates memory via {@link CudaBindings} internally.
     * Contribute {@code RocmDeviceFloatMatrix} to enable this path on AMD GPUs.
     *
     * @throws UnsupportedOperationException always
     */
    @Override
    public float[] sgemv(DeviceFloatMatrix A, float[] x) {
        throw new UnsupportedOperationException(
            "RocmMatVec: device-resident FP32 path not yet supported. " +
            "DeviceFloatMatrix uses CudaBindings internally. " +
            "See GpuContext.createMatVec() javadoc for the host-weight workaround.");
    }

    /**
     * Device-resident FP16 path — not yet supported on ROCm.
     *
     * @throws UnsupportedOperationException always
     */
    @Override
    public float[] sgemv(DeviceHalfMatrix A, float[] x) {
        throw new UnsupportedOperationException(
            "RocmMatVec: device-resident FP16 path not yet supported. " +
            "DeviceHalfMatrix uses CudaBindings internally. " +
            "See GpuContext.createMatVec() javadoc for the host-weight workaround.");
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
                GpuBindings.callInt(rocm.cublasSetPointerMode(), ctx.handle(), rocm.pointerModeHost()),
                "rocblas_set_pointer_mode");
            GpuBindings.check(
                GpuBindings.callInt(rocm.cublasSgemv(),
                    ctx.handle(), rocm.opTranspose(),
                    cols, rows,
                    alpha, dA, lda,
                    dX, 1,
                    beta, dY, 1),
                "rocblas_sgemv");
        }
    }

    // ── Stream management ─────────────────────────────────────────────────────

    /** Returns or lazily creates the per-thread non-blocking HIP stream. */
    private MemorySegment ensureStream() {
        MemorySegment stream = HIP_STREAM.get();
        if (stream != null) return stream;
        GpuBindings.check(
            GpuBindings.callInt(rocm.cudaSetDevice(), ctx.deviceIndex()),
            "hipSetDevice");
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment slot = tmp.allocate(ADDRESS);
            GpuBindings.check(
                GpuBindings.callInt(rocm.cudaStreamCreateWithFlags(), slot, STREAM_NON_BLOCKING),
                "hipStreamCreateWithFlags");
            stream = slot.get(ADDRESS, 0);
            HIP_STREAM.set(stream);
            return stream;
        }
    }

    private void bindStream(MemorySegment stream) {
        GpuBindings.check(
            GpuBindings.callInt(rocm.cublasSetStream(), ctx.handle(), stream),
            "rocblas_set_stream");
    }

    /** Restores the default stream (NULL) on the rocBLAS handle. */
    private void unbindStream() {
        GpuBindings.callInt(rocm.cublasSetStream(), ctx.handle(), MemorySegment.NULL);
    }

    // ── Scratch growth ────────────────────────────────────────────────────────

    private void ensureScratch(Scratch s, long bytesX, long bytesY) {
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
}
