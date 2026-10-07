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
 * <p>One HIP stream and one set of device scratch per instance back the
 * resident async path (same pattern as {@link CudaMatVec}): the serialization
 * lock from {@link GpuContext#cublasSerializationLock()} guards rocBLAS handle
 * usage, every call synchronizes its stream before releasing it, so one set
 * serves every caller. Not per thread, since each request runs on a new thread.
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

    // ── Device scratch and stream: one set per instance, guarded by ───────────
    // ── ctx.cublasSerializationLock() (see the class javadoc) ─────────────────
    private final Fp32Scratch fp32Scratch = new Fp32Scratch();
    private final Fp16Scratch fp16Scratch = new Fp16Scratch();
    private MemorySegment     stream;
    /** Device-side timing of this instance's copies; see {@link DeviceSpanTimer}. */
    private final DeviceSpanTimer spans;

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
        this.spans = new DeviceSpanTimer(rocm);
    }

    @Override
    public GpuContext gpuContext() { return ctx; }

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
        // Allocated inside the try, so the finally frees whichever succeeded when a
        // later one runs out of device memory (deviceFree ignores null).
        MemorySegment dA = null;
        MemorySegment dX = null;
        MemorySegment dY = null;
        try {
            dA = rocm.deviceMalloc(ctx.deviceIndex(), bytesA);
            dX = rocm.deviceMalloc(ctx.deviceIndex(), bytesX);
            dY = rocm.deviceMalloc(ctx.deviceIndex(), bytesY);
            // H2D — copy Java heap arrays into native (off-heap) staging buffers first;
            // Panama FFI (Java 25) forbids passing heap-backed MemorySegments directly
            // to native downcalls ("Heap segment not allowed").
            try (Arena hostArena = Arena.ofConfined()) {
                MemorySegment nativeA = hostArena.allocate(bytesA);
                MemorySegment nativeX = hostArena.allocate(bytesX);
                nativeA.copyFrom(MemorySegment.ofArray(A)); // heap→native (Java copy, no FFI)
                nativeX.copyFrom(MemorySegment.ofArray(x));
                DeviceStaging.copy(rocm, dA, nativeA, bytesA, GpuBindings.H2D, 1, "hipMemcpy(A H2D)");
                DeviceStaging.copy(rocm, dX, nativeX, bytesX, GpuBindings.H2D, 1, "hipMemcpy(x H2D)");
            }

            // D2H — similarly, copy into native staging first, then into Java array.
            try (Arena resultArena = Arena.ofConfined()) {
                MemorySegment stagingY = resultArena.allocate(bytesY);
                callSgemvFp32(rocm.opTranspose(), dA, cols, dX, dY, rows, cols);
                DeviceStaging.copy(rocm, stagingY, dY, bytesY, GpuBindings.D2H, 1, "hipMemcpy(y D2H)");
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

        Fp32Scratch scratch = fp32Scratch;

        try (Arena callArena = Arena.ofConfined()) {
            synchronized (ctx.cublasSerializationLock()) {
                MemorySegment stream = ensureStream();
                spans.reset();
                bindStream(stream);
                try {
                    ensureFp32Scratch(scratch, bytesX, bytesY);

                    // H2D of x — stage heap→native first (Java 25 forbids heap segments in downcalls).
                    MemorySegment stagingX = callArena.allocate(bytesX);
                    stagingX.copyFrom(MemorySegment.ofArray(x));
                    int h2dMark = spans.begin(stream, 1);
                    GpuBindings.check(
                        GpuBindings.callInt(rocm.gpuMemcpyAsync(),
                            scratch.dX, stagingX, bytesX, GpuBindings.H2D, stream),
                        "hipMemcpyAsync(x H2D)");
                    spans.staging(GpuBindings.H2D, bytesX, 1, "hipMemcpyAsync(x H2D)", h2dMark, stream);

                    callSgemvFp32(rocm.opTranspose(), A.devicePointer(), cols, scratch.dX, scratch.dY, rows, cols);

                    MemorySegment stagingY = callArena.allocate(bytesY);
                    int d2hMark = spans.begin(stream, 1);
                    GpuBindings.check(
                        GpuBindings.callInt(rocm.gpuMemcpyAsync(),
                            stagingY, scratch.dY, bytesY, GpuBindings.D2H, stream),
                        "hipMemcpyAsync(y D2H)");
                    spans.staging(GpuBindings.D2H, bytesY, 1, "hipMemcpyAsync(y D2H)", d2hMark, stream);
                    GpuBindings.check(
                        GpuBindings.callInt(rocm.gpuStreamSynchronize(), stream),
                        "hipStreamSynchronize");
                    spans.commit();

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

        Fp16Scratch scratch = fp16Scratch;

        try (Arena callArena = Arena.ofConfined()) {
            synchronized (ctx.cublasSerializationLock()) {
                MemorySegment stream = ensureStream();
                spans.reset();
                bindStream(stream);
                try {
                    ensureFp16Scratch(scratch, bytesXh, bytesY);

                    // Pack x as FP16 into off-heap staging. JAVA_SHORT has native byte order
                    // (little-endian on x86), matching HIP's __half layout.
                    MemorySegment stagingXh = callArena.allocate(bytesXh);
                    for (int j = 0; j < cols; j++)
                        stagingXh.setAtIndex(JAVA_SHORT, j, Float.floatToFloat16(x[j]));

                    int h2dMark = spans.begin(stream, 1);
                    GpuBindings.check(
                        GpuBindings.callInt(rocm.gpuMemcpyAsync(),
                            scratch.dXh, stagingXh, bytesXh, GpuBindings.H2D, stream),
                        "hipMemcpyAsync(xh H2D)");
                    spans.staging(GpuBindings.H2D, bytesXh, 1, "hipMemcpyAsync(xh H2D)", h2dMark, stream);

                    callSgemvFp16(rocm.opTranspose(), A.devicePointer(), cols, scratch.dXh, scratch.dY, rows, cols);

                    MemorySegment stagingY = callArena.allocate(bytesY);
                    int d2hMark = spans.begin(stream, 1);
                    GpuBindings.check(
                        GpuBindings.callInt(rocm.gpuMemcpyAsync(),
                            stagingY, scratch.dY, bytesY, GpuBindings.D2H, stream),
                        "hipMemcpyAsync(y D2H)");
                    spans.staging(GpuBindings.D2H, bytesY, 1, "hipMemcpyAsync(y D2H)", d2hMark, stream);
                    GpuBindings.check(
                        GpuBindings.callInt(rocm.gpuStreamSynchronize(), stream),
                        "hipStreamSynchronize");
                    spans.commit();

                    float[] y = new float[rows];
                    MemorySegment.copy(stagingY, JAVA_FLOAT, 0, y, 0, rows);
                    return y;
                } finally {
                    unbindStream();
                }
            }
        } finally {
            evt.commit();
        }
    }

    /**
     * Shared-activation GEMVs: one FP16 H2D of {@code x}, N rocBLAS calls, one sync.
     */
    @Override
    public float[][] sgemvSameX(DeviceHalfMatrix[] weights, float[] x) {
        if (weights == null || weights.length == 0)
            return new float[0][];
        if (weights.length == 1)
            return new float[][] { sgemv(weights[0], x) };
        int cols = x.length;
        int n = weights.length;
        int maxRows = 0;
        for (int i = 0; i < n; i++) {
            DeviceHalfMatrix A = weights[i];
            if (A == null)
                throw new IllegalArgumentException("weights[" + i + "] is null");
            if (A.isClosed())
                throw new IllegalStateException("DeviceHalfMatrix is closed");
            if (A.cols() != cols)
                throw new IllegalArgumentException(
                        "weights[" + i + "].cols=" + A.cols() + " != x.length=" + cols);
            maxRows = Math.max(maxRows, A.rows());
        }

        MatVecEvent evt = new MatVecEvent();
        evt.begin();
        long bytesXh = (long) cols * Short.BYTES;
        long bytesYMax = (long) maxRows * Float.BYTES;
        Fp16Scratch scratch = fp16Scratch;
        float[][] Y = new float[n][];

        try (Arena callArena = Arena.ofConfined()) {
            synchronized (ctx.cublasSerializationLock()) {
                MemorySegment stream = ensureStream();
                spans.reset();
                bindStream(stream);
                try {
                    ensureFp16Scratch(scratch, bytesXh, bytesYMax);
                    MemorySegment stagingXh = callArena.allocate(bytesXh);
                    for (int j = 0; j < cols; j++)
                        stagingXh.setAtIndex(JAVA_SHORT, j, Float.floatToFloat16(x[j]));
                    int h2dMark = spans.begin(stream, 1);
                    GpuBindings.check(
                            GpuBindings.callInt(rocm.gpuMemcpyAsync(),
                                    scratch.dXh, stagingXh, bytesXh, GpuBindings.H2D, stream),
                            "hipMemcpyAsync(xh H2D sameX)");
                    spans.staging(GpuBindings.H2D, bytesXh, 1, "hipMemcpyAsync(xh H2D sameX)", h2dMark, stream);

                    MemorySegment[] stagingY = new MemorySegment[n];
                    for (int i = 0; i < n; i++) {
                        DeviceHalfMatrix A = weights[i];
                        int rows = A.rows();
                        callSgemvFp16(rocm.opTranspose(), A.devicePointer(), cols,
                                scratch.dXh, scratch.dY, rows, cols);
                        long bytesY = (long) rows * Float.BYTES;
                        stagingY[i] = callArena.allocate(bytesY);
                        int d2hMark = spans.begin(stream, 1);
                        GpuBindings.check(
                                GpuBindings.callInt(rocm.gpuMemcpyAsync(),
                                        stagingY[i], scratch.dY, bytesY, GpuBindings.D2H, stream),
                                "hipMemcpyAsync(y D2H sameX)");
                        spans.staging(GpuBindings.D2H, bytesY, 1, "hipMemcpyAsync(y D2H sameX)", d2hMark, stream);
                    }
                    GpuBindings.check(
                            GpuBindings.callInt(rocm.gpuStreamSynchronize(), stream),
                            "hipStreamSynchronize");
                    spans.commit();
                    for (int i = 0; i < n; i++) {
                        int rows = weights[i].rows();
                        Y[i] = new float[rows];
                        MemorySegment.copy(stagingY[i], JAVA_FLOAT, 0, Y[i], 0, rows);
                    }
                    return Y;
                } finally {
                    unbindStream();
                }
            }
        } finally {
            evt.commit();
        }
    }

    @Override
    public float[][] sgemvSameX(DeviceFloatMatrix[] weights, float[] x) {
        if (weights == null || weights.length == 0)
            return new float[0][];
        if (weights.length == 1)
            return new float[][] { sgemv(weights[0], x) };
        int cols = x.length;
        int n = weights.length;
        int maxRows = 0;
        for (int i = 0; i < n; i++) {
            DeviceFloatMatrix A = weights[i];
            if (A == null)
                throw new IllegalArgumentException("weights[" + i + "] is null");
            if (A.isClosed())
                throw new IllegalStateException("DeviceFloatMatrix is closed");
            if (A.cols() != cols)
                throw new IllegalArgumentException(
                        "weights[" + i + "].cols=" + A.cols() + " != x.length=" + cols);
            maxRows = Math.max(maxRows, A.rows());
        }

        MatVecEvent evt = new MatVecEvent();
        evt.begin();
        long bytesX = (long) cols * Float.BYTES;
        long bytesYMax = (long) maxRows * Float.BYTES;
        Fp32Scratch scratch = fp32Scratch;
        float[][] Y = new float[n][];

        try (Arena callArena = Arena.ofConfined()) {
            synchronized (ctx.cublasSerializationLock()) {
                MemorySegment stream = ensureStream();
                spans.reset();
                bindStream(stream);
                try {
                    ensureFp32Scratch(scratch, bytesX, bytesYMax);
                    MemorySegment stagingX = callArena.allocate(bytesX);
                    stagingX.copyFrom(MemorySegment.ofArray(x));
                    int h2dMark = spans.begin(stream, 1);
                    GpuBindings.check(
                            GpuBindings.callInt(rocm.gpuMemcpyAsync(),
                                    scratch.dX, stagingX, bytesX, GpuBindings.H2D, stream),
                            "hipMemcpyAsync(x H2D sameX)");
                    spans.staging(GpuBindings.H2D, bytesX, 1, "hipMemcpyAsync(x H2D sameX)", h2dMark, stream);

                    MemorySegment[] stagingY = new MemorySegment[n];
                    for (int i = 0; i < n; i++) {
                        DeviceFloatMatrix A = weights[i];
                        int rows = A.rows();
                        callSgemvFp32(rocm.opTranspose(), A.devicePointer(), cols,
                                scratch.dX, scratch.dY, rows, cols);
                        long bytesY = (long) rows * Float.BYTES;
                        stagingY[i] = callArena.allocate(bytesY);
                        int d2hMark = spans.begin(stream, 1);
                        GpuBindings.check(
                                GpuBindings.callInt(rocm.gpuMemcpyAsync(),
                                        stagingY[i], scratch.dY, bytesY, GpuBindings.D2H, stream),
                                "hipMemcpyAsync(y D2H sameX)");
                        spans.staging(GpuBindings.D2H, bytesY, 1, "hipMemcpyAsync(y D2H sameX)", d2hMark, stream);
                    }
                    GpuBindings.check(
                            GpuBindings.callInt(rocm.gpuStreamSynchronize(), stream),
                            "hipStreamSynchronize");
                    spans.commit();
                    for (int i = 0; i < n; i++) {
                        int rows = weights[i].rows();
                        Y[i] = new float[rows];
                        MemorySegment.copy(stagingY[i], JAVA_FLOAT, 0, Y[i], 0, rows);
                    }
                    return Y;
                } finally {
                    unbindStream();
                }
            }
        } finally {
            evt.commit();
        }
    }

    /**
     * Device-resident FP32 transpose: {@code z = W^T * g} for row-major
     * {@code W[rows×cols]}. Uses {@link GpuBindings#opNoTranspose()}.
     */
    @Override
    public float[] sgemvTranspose(DeviceFloatMatrix W, float[] g) {
        if (W == null) throw new IllegalArgumentException("W must not be null");
        if (W.isClosed()) throw new IllegalStateException("DeviceFloatMatrix is closed");
        int rows = W.rows(), cols = W.cols();
        if (g.length != rows)
            throw new IllegalArgumentException("g.length=" + g.length + " != rows=" + rows);

        MatVecEvent evt = new MatVecEvent();
        evt.begin();

        long bytesG = (long) rows * Float.BYTES;
        long bytesZ = (long) cols * Float.BYTES;

        Fp32Scratch scratch = fp32Scratch;

        try (Arena callArena = Arena.ofConfined()) {
            synchronized (ctx.cublasSerializationLock()) {
                MemorySegment stream = ensureStream();
                spans.reset();
                bindStream(stream);
                try {
                    ensureFp32Scratch(scratch, bytesG, bytesZ);

                    MemorySegment stagingG = callArena.allocate(bytesG);
                    stagingG.copyFrom(MemorySegment.ofArray(g));
                    int h2dMark = spans.begin(stream, 1);
                    GpuBindings.check(
                        GpuBindings.callInt(rocm.gpuMemcpyAsync(),
                            scratch.dX, stagingG, bytesG, GpuBindings.H2D, stream),
                        "hipMemcpyAsync(g H2D)");
                    spans.staging(GpuBindings.H2D, bytesG, 1, "hipMemcpyAsync(g H2D)", h2dMark, stream);

                    callSgemvFp32(rocm.opNoTranspose(), W.devicePointer(), cols,
                            scratch.dX, scratch.dY, rows, cols);

                    MemorySegment stagingZ = callArena.allocate(bytesZ);
                    int d2hMark = spans.begin(stream, 1);
                    GpuBindings.check(
                        GpuBindings.callInt(rocm.gpuMemcpyAsync(),
                            stagingZ, scratch.dY, bytesZ, GpuBindings.D2H, stream),
                        "hipMemcpyAsync(z D2H)");
                    spans.staging(GpuBindings.D2H, bytesZ, 1, "hipMemcpyAsync(z D2H)", d2hMark, stream);
                    GpuBindings.check(
                        GpuBindings.callInt(rocm.gpuStreamSynchronize(), stream),
                        "hipStreamSynchronize");
                    spans.commit();

                    float[] z = new float[cols];
                    MemorySegment.copy(stagingZ, JAVA_FLOAT, 0, z, 0, cols);
                    return z;
                } finally {
                    unbindStream();
                }
            }
        } finally {
            evt.backend(MatVecBackend.ROCM_RESIDENT_TRANSPOSE);
            evt.rows = cols;
            evt.cols = rows;
            evt.commit();
        }
    }

    /**
     * Device-resident FP16 transpose. Same layout contract as
     * {@link #sgemvTranspose(DeviceFloatMatrix, float[])}.
     */
    @Override
    public float[] sgemvTranspose(DeviceHalfMatrix W, float[] g) {
        if (W == null) throw new IllegalArgumentException("W must not be null");
        if (W.isClosed()) throw new IllegalStateException("DeviceHalfMatrix is closed");
        int rows = W.rows(), cols = W.cols();
        if (g.length != rows)
            throw new IllegalArgumentException("g.length=" + g.length + " != rows=" + rows);

        MatVecEvent evt = new MatVecEvent();
        evt.begin();

        long bytesGh = (long) rows * Short.BYTES;
        long bytesZ  = (long) cols * Float.BYTES;

        Fp16Scratch scratch = fp16Scratch;

        try (Arena callArena = Arena.ofConfined()) {
            synchronized (ctx.cublasSerializationLock()) {
                MemorySegment stream = ensureStream();
                spans.reset();
                bindStream(stream);
                try {
                    ensureFp16Scratch(scratch, bytesGh, bytesZ);

                    MemorySegment stagingGh = callArena.allocate(bytesGh);
                    for (int r = 0; r < rows; r++)
                        stagingGh.setAtIndex(JAVA_SHORT, r, Float.floatToFloat16(g[r]));

                    int h2dMark = spans.begin(stream, 1);
                    GpuBindings.check(
                        GpuBindings.callInt(rocm.gpuMemcpyAsync(),
                            scratch.dXh, stagingGh, bytesGh, GpuBindings.H2D, stream),
                        "hipMemcpyAsync(gh H2D)");
                    spans.staging(GpuBindings.H2D, bytesGh, 1, "hipMemcpyAsync(gh H2D)", h2dMark, stream);

                    callSgemvFp16(rocm.opNoTranspose(), W.devicePointer(), cols,
                            scratch.dXh, scratch.dY, rows, cols);

                    MemorySegment stagingZ = callArena.allocate(bytesZ);
                    int d2hMark = spans.begin(stream, 1);
                    GpuBindings.check(
                        GpuBindings.callInt(rocm.gpuMemcpyAsync(),
                            stagingZ, scratch.dY, bytesZ, GpuBindings.D2H, stream),
                        "hipMemcpyAsync(z D2H)");
                    spans.staging(GpuBindings.D2H, bytesZ, 1, "hipMemcpyAsync(z D2H)", d2hMark, stream);
                    GpuBindings.check(
                        GpuBindings.callInt(rocm.gpuStreamSynchronize(), stream),
                        "hipStreamSynchronize");
                    spans.commit();

                    float[] z = new float[cols];
                    MemorySegment.copy(stagingZ, JAVA_FLOAT, 0, z, 0, cols);
                    return z;
                } finally {
                    unbindStream();
                }
            }
        } finally {
            evt.backend(MatVecBackend.ROCM_RESIDENT_FP16_TRANSPOSE);
            evt.rows = cols;
            evt.cols = rows;
            evt.commit();
        }
    }

    // ── rocBLAS GEMV ──────────────────────────────────────────────────────────

    /**
     * rocblas_sgemv on row-major {@code W[rows×cols]}.
     *
     * <p>Same {@code (m,n,lda)} mapping as {@link CudaMatVec}:
     * {@code m=cols}, {@code n=rows}, {@code lda=cols}.
     * {@link GpuBindings#opTranspose()} → forward; {@link GpuBindings#opNoTranspose()} →
     * transpose backward.
     */
    private void callSgemvFp32(int op, MemorySegment dA, int lda,
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
                    ctx.handle(), op,
                    cols, rows,
                    alpha, dA, lda,
                    dX, 1,
                    beta, dY, 1),
                "rocblas_sgemv");
        }
    }

    /**
     * rocblas_hssgemv_strided_batched: same {@code (op, m, n, lda)} mapping as
     * {@link #callSgemvFp32}.
     */
    private void callSgemvFp16(int op, MemorySegment dA, int lda,
                                MemorySegment dXh, MemorySegment dY,
                                int rows, int cols) {
        long strideA = (long) cols * rows;
        long strideX = (op == rocm.opNoTranspose()) ? rows : cols;
        long strideY = (op == rocm.opNoTranspose()) ? cols : rows;
        try (Arena scalars = Arena.ofConfined()) {
            MemorySegment alpha = scalars.allocateFrom(JAVA_FLOAT, 1.0f);
            MemorySegment beta  = scalars.allocateFrom(JAVA_FLOAT, 0.0f);
            GpuBindings.check(
                GpuBindings.callInt(rocm.blasSetPointerMode(), ctx.handle(), rocm.pointerModeHost()),
                "rocblas_set_pointer_mode");
            GpuBindings.check(
                GpuBindings.callInt(rocm.blasHSSgemvStridedBatched(),
                    ctx.handle(), op,
                    cols, rows,
                    alpha, dA, lda, strideA,
                    dXh, 1, strideX,
                    beta, dY, 1, strideY,
                    1),
                "rocblas_hssgemv_strided_batched");
        }
    }

    // ── Stream management ─────────────────────────────────────────────────────

    /** Returns or lazily creates the instance's non-blocking HIP stream. Caller holds the lock. */
    private MemorySegment ensureStream() {
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

    // Each grow clears its slot before freeing, so a failed allocation leaves it
    // empty rather than holding a freed pointer: the scratch outlives the call.

    private void ensureFp32Scratch(Fp32Scratch s, long bytesX, long bytesY) {
        int dev = ctx.deviceIndex();
        if (s.dXBytes < bytesX) {
            MemorySegment previous = s.dX;
            s.dX = null;
            s.dXBytes = 0L;
            rocm.deviceFree(previous);
            s.dX     = rocm.deviceMalloc(dev, bytesX);
            s.dXBytes = bytesX;
        }
        if (s.dYBytes < bytesY) {
            MemorySegment previous = s.dY;
            s.dY = null;
            s.dYBytes = 0L;
            rocm.deviceFree(previous);
            s.dY     = rocm.deviceMalloc(dev, bytesY);
            s.dYBytes = bytesY;
        }
    }

    private void ensureFp16Scratch(Fp16Scratch s, long bytesXh, long bytesY) {
        int dev = ctx.deviceIndex();
        if (s.dXhBytes < bytesXh) {
            MemorySegment previous = s.dXh;
            s.dXh = null;
            s.dXhBytes = 0L;
            rocm.deviceFree(previous);
            s.dXh     = rocm.deviceMalloc(dev, bytesXh);
            s.dXhBytes = bytesXh;
        }
        if (s.dYBytes < bytesY) {
            MemorySegment previous = s.dY;
            s.dY = null;
            s.dYBytes = 0L;
            rocm.deviceFree(previous);
            s.dY     = rocm.deviceMalloc(dev, bytesY);
            s.dYBytes = bytesY;
        }
    }

    // ── Scratch lifetime ──────────────────────────────────────────────────────

    /**
     * Frees this instance's device scratch, timing events and stream. Safe while other callers
     * are active: it takes the lock they hold, and the next call grows the
     * scratch again.
     */
    void releaseScratch() {
        synchronized (ctx.cublasSerializationLock()) {
            rocm.deviceFree(fp32Scratch.dX);
            rocm.deviceFree(fp32Scratch.dY);
            fp32Scratch.dX = fp32Scratch.dY = null;
            fp32Scratch.dXBytes = fp32Scratch.dYBytes = 0L;
            rocm.deviceFree(fp16Scratch.dXh);
            rocm.deviceFree(fp16Scratch.dY);
            fp16Scratch.dXh = fp16Scratch.dY = null;
            fp16Scratch.dXhBytes = fp16Scratch.dYBytes = 0L;
            spans.releaseEvents();
            if (stream != null) {
                GpuBindings.callInt(rocm.gpuStreamDestroy(), stream);
                stream = null;
            }
        }
    }

    /** Device bytes this instance's scratch holds now (excluding the stream). */
    long scratchDeviceBytes() {
        synchronized (ctx.cublasSerializationLock()) {
            return fp32Scratch.dXBytes + fp32Scratch.dYBytes + fp16Scratch.dXhBytes + fp16Scratch.dYBytes;
        }
    }
}