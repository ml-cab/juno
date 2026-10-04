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
 *   <li>x and y scratch buffers on the device, the Q4_K dequant buffer and the
 *       CUDA stream belong to the instance, are grown lazily and reused across
 *       calls. They are touched only under the context's serialization lock,
 *       which every call holds until its stream has been synchronized, so one
 *       set serves every caller. Not per thread: the request scheduler runs each
 *       request on a new thread, and scratch tied to a finished thread would
 *       stay allocated on the device. {@link #releaseScratch()} frees the set;
 *       the next call grows it again.
 *   <li>H2D upload of x uses a short-lived confined {@link Arena}: x is copied
 *       from the heap array into native memory with {@code copyFrom} (a pure Java
 *       operation), then the native segment is passed to {@code cudaMemcpyAsync}.
 *       Panama FFI (Java 25) rejects heap-backed segments in native downcalls.
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
public final class CudaMatVec implements GpuMatVec {

    @SuppressWarnings("unused")
    private static final Logger log = Logger.getLogger(CudaMatVec.class.getName());
    private static final java.util.concurrent.atomic.AtomicBoolean tiledFallbackWarned =
            new java.util.concurrent.atomic.AtomicBoolean();

    private static final int STREAM_NON_BLOCKING = CudaBindings.STREAM_NON_BLOCKING;

    private final GpuContext     ctx;
    private final CudaBindings   cuda;
    private GpuBlasOps           blasOps;
    private CudaFp16GemmOps      fp16GemmOps;

    // ── Device scratch and stream: one set per instance, guarded by ───────────
    // ── ctx.cublasSerializationLock() (see the class javadoc) ─────────────────
    private final Fp32Scratch       fp32Scratch    = new Fp32Scratch();
    private final Fp16Scratch       fp16Scratch    = new Fp16Scratch();
    private final Q4KDequantScratch dequantScratch = new Q4KDequantScratch();
    private final Q8WindowScratch q8WindowScratch = new Q8WindowScratch();
    /** {@link #dequantizeBatchedKQuant}: batched K-quant matmuls take the FP16 dequant route. */
    private volatile boolean dequantBatchedKQuant;
    private MemorySegment           stream;
    /** Device-side timing of this instance's copies and dequantizations; see {@link DeviceSpanTimer}. */
    private final DeviceSpanTimer   spans;

    // ── Scratch containers ────────────────────────────────────────────────────

    private static final class Fp32Scratch {
        MemorySegment dX;   // device, grown as needed
        MemorySegment dY;   // device, grown as needed
        MemorySegment dQ8;  // device Q8_1 packing of x (K-quant GEMV)
        long dXBytes;
        long dYBytes;
        long dQ8Bytes;
    }

    private static final class Fp16Scratch {
        MemorySegment dXh;  // device FP16 x, grown as needed
        MemorySegment dY;   // device FP32 y, grown as needed
        long dXhBytes;
        long dYBytes;
        MemorySegment hXh;  // pinned host staging for dXh H2D upload, grown as needed
        MemorySegment hY;   // pinned host staging for dY D2H download, grown as needed
        long hXhBytes;
        long hYBytes;
    }

    // ── Construction ──────────────────────────────────────────────────────────

    /**
     * @param ctx an open GpuContext — must outlive all sgemv calls on this instance
     */
    public CudaMatVec(GpuContext ctx) {
        if (ctx == null) throw new IllegalArgumentException("ctx must not be null");
        this.ctx  = ctx;
        this.cuda = CudaBindings.instance();
        this.spans = new DeviceSpanTimer(cuda);
    }

    @Override
    public GpuContext gpuContext() { return ctx; }

    // ── Upload helpers (for LlamaTransformerHandler / LoraTrainableHandler) ──

    @Override
    public DeviceFloatMatrix upload(float[] host, int rows, int cols) {
        DeviceFloatMatrix m = DeviceFloatMatrix.upload(ctx, host, rows, cols);
        holdGemvScratch(m, rows, cols, false, false);
        return m;
    }

    @Override
    public DeviceHalfMatrix uploadHalf(float[] host, int rows, int cols) {
        DeviceHalfMatrix m = DeviceHalfMatrix.uploadFromFloat32(ctx, host, rows, cols);
        holdGemvScratch(m, rows, cols, true, false);
        return m;
    }

    @Override
    public DeviceQ4KMatrix uploadQ4K(byte[] raw, int rows, int cols) {
        return uploadKQuant(raw, rows, cols, QuantizationLayout.TYPE_Q4_K);
    }

    @Override
    public DeviceQ4KMatrix uploadKQuant(byte[] raw, int rows, int cols, int typeId) {
        if (!supportsQ4KMmq())
            throw new UnsupportedOperationException("K-quant MMQ kernel is not available");
        DeviceQ4KMatrix m = DeviceQ4KMatrix.upload(ctx, raw, rows, cols, typeId);
        holdGemvScratch(m, rows, cols, false, true);
        return m;
    }

    /**
     * Output rows a shared-input product can write at once, in units of the widest
     * matrix's rows: {@link #sgemvSameX} over separate Q, K and V matrices writes up to
     * three times the query projection's rows (multi-head attention, where K and V are
     * as wide as Q), and gate with up writes two.
     */
    private static final int SAME_X_MAX_MATRICES = 3;

    /**
     * Holds the single-row product scratch an uploaded matrix needs, so a decode
     * product allocates no device memory: the input sized for {@code max(rows, cols)}
     * (the transposed product reads {@code rows}), the output for
     * {@link #SAME_X_MAX_MATRICES} times that, and for packed weights the input's
     * 8-bit copy. It is held before the caller's upload stop rule reads the free
     * memory, which makes it part of this process's footprint. Grown lazily on the
     * first product, it could find nothing left when several processes share a device
     * and fill it down to one reserve between them, and fail a request after the
     * weights had loaded. A failure here frees the matrix and rethrows, so the caller's
     * upload handling sees one out-of-memory error for the layer.
     */
    private void holdGemvScratch(AutoCloseable matrix, int rows, int cols, boolean half, boolean packed) {
        long widest = Math.max(rows, cols);
        long bytesX = widest * Float.BYTES;
        long bytesY = SAME_X_MAX_MATRICES * widest * Float.BYTES;
        try {
            synchronized (ctx.cublasSerializationLock()) {
                if (half)
                    ensureFp16Scratch(fp16Scratch, widest * Short.BYTES, bytesY);
                else
                    ensureFp32Scratch(fp32Scratch, bytesX, bytesY, packed ? Q4KMmqKernel.q8Bytes(cols) : 0);
            }
        } catch (RuntimeException ex) {
            try {
                matrix.close();
            } catch (Exception closeFailure) {
                ex.addSuppressed(closeFailure);
            }
            throw ex;
        }
    }

    @Override
    public boolean supportsQ4KMmq() {
        return Q4KMmqKernel.isAvailable() && Q4KMmqKernel.tryLoad() != null;
    }

    // ── MatVec ────────────────────────────────────────────────────────────────

    /**
     * Full host path: A and x are on the host.
     *
     * Copies A and x into confined off-heap staging arenas before the H2D
     * {@code cudaMemcpy} calls. Panama FFI (Java 25) rejects heap-backed
     * {@link MemorySegment}s in native downcalls; {@code MemorySegment.copyFrom}
     * is a pure Java copy and is not subject to that restriction.
     * The D2H result is likewise written into a staging segment and then
     * copied into the returned heap array.
     *
     * Intended for the non-resident forward pass. The resident inference
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
            // H2D — copy heap arrays into confined off-heap staging first.
            // Panama FFI (Java 25) rejects heap-backed MemorySegments in native downcalls;
            // MemorySegment.copyFrom is a pure Java copy and is not subject to that restriction.
            try (Arena h2dArena = Arena.ofConfined()) {
                MemorySegment nativeA = h2dArena.allocate(bytesA);
                MemorySegment nativeX = h2dArena.allocate(bytesX);
                nativeA.copyFrom(MemorySegment.ofArray(A));
                nativeX.copyFrom(MemorySegment.ofArray(x));
                DeviceStaging.copy(cuda, dA, nativeA, bytesA, CudaBindings.H2D, 1, "cudaMemcpy(A H2D)");
                DeviceStaging.copy(cuda, dX, nativeX, bytesX, CudaBindings.H2D, 1, "cudaMemcpy(x H2D)");
            }

            float[] y = new float[rows];
            synchronized (ctx.cublasSerializationLock()) {
                callSgemvFp32(CudaBindings.CUBLAS_OP_T, dA, cols, dX, dY, rows, cols);
                // D2H into off-heap staging; copy into the heap array afterwards.
                try (Arena d2hArena = Arena.ofConfined()) {
                    MemorySegment stagingY = d2hArena.allocate(bytesY);
                    DeviceStaging.copy(cuda, stagingY, dY, bytesY, CudaBindings.D2H, 1, "cudaMemcpy(y D2H)");
                    MemorySegment.copy(stagingY, JAVA_FLOAT, 0, y, 0, rows);
                }
            }
            return y;
        } finally {
            cuda.deviceFree(dA);
            cuda.deviceFree(dX);
            cuda.deviceFree(dY);
            evt.backend(MatVecBackend.CUDA);
            evt.rows = rows;
            evt.cols = cols;
            evt.commit();
        }
    }

    /**
     * Device-resident FP32 path: A stays on the device across calls.
     *
     * Per-thread scratch buffers for x and y are grown lazily and reused.
     * x is staged through a short-lived confined arena before the H2D
     * {@code cudaMemcpyAsync} — Panama FFI (Java 25) rejects heap-backed
     * segments in native downcalls.
     * The D2H copy targets {@code resultArena} (off-heap) and is copied into
     * a heap array only after {@code cudaStreamSynchronize} returns.
     */
    @Override
    public float[] sgemv(DeviceFloatMatrix A, float[] x) {
        if (A == null) throw new IllegalArgumentException("A must not be null");
        float[] y = new float[A.rows()];
        sgemvInto(A, x, y);
        return y;
    }

    /** {@link #sgemv(DeviceFloatMatrix, float[])} into {@code y[0, rows)}. */
    private void sgemvInto(DeviceFloatMatrix A, float[] x, float[] y) {
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

        try (Arena resultArena = Arena.ofConfined()) {
            synchronized (ctx.cublasSerializationLock()) {
                MemorySegment stream = ensureStream();
                spans.reset();
                bindStream(stream);
                try {
                    ensureFp32Scratch(scratch, bytesX, bytesY);

                    // H2D: copy x into a confined off-heap staging segment before the downcall.
                    // Panama FFI (Java 25) rejects heap-backed MemorySegments in native downcalls.
                    try (Arena h2dArena = Arena.ofConfined()) {
                        MemorySegment nativeX = h2dArena.allocate(bytesX);
                        nativeX.copyFrom(MemorySegment.ofArray(x));
                        int h2dMark = spans.begin(stream, 1);
                        CudaBindings.check(
                            CudaBindings.callInt(cuda.cudaMemcpyAsync,
                                scratch.dX, nativeX, bytesX, CudaBindings.H2D, stream),
                            "cudaMemcpyAsync(x H2D)");
                        spans.staging(CudaBindings.H2D, bytesX, 1, "cudaMemcpyAsync(x H2D)", h2dMark, stream);
                    }

                    callSgemvFp32(CudaBindings.CUBLAS_OP_T, A.devicePointer(), cols, scratch.dX, scratch.dY, rows, cols);

                    // D2H into off-heap staging — the async copy must not target a moveable heap address.
                    MemorySegment stagingY = resultArena.allocate(bytesY);
                    int d2hMark = spans.begin(stream, 1);
                    CudaBindings.check(
                        CudaBindings.callInt(cuda.cudaMemcpyAsync,
                            stagingY, scratch.dY, bytesY, CudaBindings.D2H, stream),
                        "cudaMemcpyAsync(y D2H)");
                    spans.staging(CudaBindings.D2H, bytesY, 1, "cudaMemcpyAsync(y D2H)", d2hMark, stream);
                    CudaBindings.check(
                        CudaBindings.callInt(cuda.cudaStreamSynchronize, stream),
                        "cudaStreamSynchronize");
                    spans.commit();

                    MemorySegment.copy(stagingY, JAVA_FLOAT, 0, y, 0, rows);
                } finally {
                    unbindStream();
                }
            }
        } finally {
            evt.backend(MatVecBackend.CUDA_RESIDENT);
            evt.rows = rows;
            evt.cols = cols;
            evt.commit();
        }
    }

    /**
     * Device-resident fused K-quant path: packed weights stay on device; x is
     * quantized to Q8_1 once, then a PTX integer-dot kernel writes FP32 {@code y}.
     */
    @Override
    public float[] sgemv(DeviceQ4KMatrix A, float[] x) {
        if (A == null) throw new IllegalArgumentException("A must not be null");
        float[] y = new float[A.rows()];
        sgemvInto(A, x, y);
        return y;
    }

    /** {@link #sgemv(DeviceQ4KMatrix, float[])} into {@code y[0, rows)}. */
    private void sgemvInto(DeviceQ4KMatrix A, float[] x, float[] y) {
        if (A == null) throw new IllegalArgumentException("A must not be null");
        if (A.isClosed()) throw new IllegalStateException("DeviceQ4KMatrix is closed");
        int rows = A.rows(), cols = A.cols();
        if (x.length != cols)
            throw new IllegalArgumentException("x.length=" + x.length + " != cols=" + cols);
        Q4KMmqKernel kernel = Q4KMmqKernel.tryLoad();
        if (kernel == null)
            throw new IllegalStateException("Q4_K MMQ kernel is not loaded");

        MatVecEvent evt = new MatVecEvent();
        evt.begin();

        long bytesX = (long) cols * Float.BYTES;
        long bytesY = (long) rows * Float.BYTES;
        long bytesQ8 = Q4KMmqKernel.q8Bytes(cols);
        Fp32Scratch scratch = fp32Scratch;

        try (Arena resultArena = Arena.ofConfined()) {
            synchronized (ctx.cublasSerializationLock()) {
                MemorySegment stream = ensureStream();
                spans.reset();
                try {
                    ensureFp32Scratch(scratch, bytesX, bytesY, bytesQ8);
                    try (Arena h2dArena = Arena.ofConfined()) {
                        MemorySegment nativeX = h2dArena.allocate(bytesX);
                        nativeX.copyFrom(MemorySegment.ofArray(x));
                        int h2dMark = spans.begin(stream, 1);
                        CudaBindings.check(
                                CudaBindings.callInt(cuda.cudaMemcpyAsync,
                                        scratch.dX, nativeX, bytesX, CudaBindings.H2D, stream),
                                "cudaMemcpyAsync(x H2D q4k)");
                        spans.staging(CudaBindings.H2D, bytesX, 1, "cudaMemcpyAsync(x H2D q4k)", h2dMark, stream);
                    }
                    kernel.launch(A, scratch.dX, scratch.dQ8, scratch.dY, stream);
                    spans.compute(DeviceComputeEvent.MMQ_PACKED, 1, -1, stream);
                    MemorySegment stagingY = resultArena.allocate(bytesY);
                    int d2hMark = spans.begin(stream, 1);
                    CudaBindings.check(
                            CudaBindings.callInt(cuda.cudaMemcpyAsync,
                                    stagingY, scratch.dY, bytesY, CudaBindings.D2H, stream),
                            "cudaMemcpyAsync(y D2H q4k)");
                    spans.staging(CudaBindings.D2H, bytesY, 1, "cudaMemcpyAsync(y D2H q4k)", d2hMark, stream);
                    CudaBindings.check(
                            CudaBindings.callInt(cuda.cudaStreamSynchronize, stream),
                            "cudaStreamSynchronize");
                    spans.commit();
                    MemorySegment.copy(stagingY, JAVA_FLOAT, 0, y, 0, rows);
                } finally {
                    // no cuBLAS stream bind for this path
                }
            }
        } finally {
            evt.backend(MatVecBackend.CUDA_RESIDENT_Q4K);
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
        float[] y = new float[A.rows()];
        sgemvInto(A, x, y);
        return y;
    }

    /** {@link #sgemv(DeviceHalfMatrix, float[])} into {@code y[0, rows)}. */
    private void sgemvInto(DeviceHalfMatrix A, float[] x, float[] y) {
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
                    // (little-endian on x86), matching CUDA __half layout.
                    MemorySegment stagingXh = callArena.allocate(bytesXh);
                    for (int j = 0; j < cols; j++)
                        stagingXh.setAtIndex(JAVA_SHORT, j, Float.floatToFloat16(x[j]));

                    int h2dMark = spans.begin(stream, 1);
                    CudaBindings.check(
                        CudaBindings.callInt(cuda.cudaMemcpyAsync,
                            scratch.dXh, stagingXh, bytesXh, CudaBindings.H2D, stream),
                        "cudaMemcpyAsync(xh H2D)");
                    spans.staging(CudaBindings.H2D, bytesXh, 1, "cudaMemcpyAsync(xh H2D)", h2dMark, stream);

                    callSgemvFp16(CudaBindings.CUBLAS_OP_T, A.devicePointer(), cols, scratch.dXh, scratch.dY, rows, cols, 1);

                    MemorySegment stagingY = callArena.allocate(bytesY);
                    int d2hMark = spans.begin(stream, 1);
                    CudaBindings.check(
                        CudaBindings.callInt(cuda.cudaMemcpyAsync,
                            stagingY, scratch.dY, bytesY, CudaBindings.D2H, stream),
                        "cudaMemcpyAsync(y D2H)");
                    spans.staging(CudaBindings.D2H, bytesY, 1, "cudaMemcpyAsync(y D2H)", d2hMark, stream);
                    CudaBindings.check(
                        CudaBindings.callInt(cuda.cudaStreamSynchronize, stream),
                        "cudaStreamSynchronize");
                    spans.commit();

                    MemorySegment.copy(stagingY, JAVA_FLOAT, 0, y, 0, rows);
                } finally {
                    unbindStream();
                }
            }
        } finally {
            evt.backend(MatVecBackend.CUDA_RESIDENT_FP16);
            evt.rows = rows;
            evt.cols = cols;
            evt.commit();
        }
    }

    /**
     * Shared-activation GEMVs: one FP16 H2D of {@code x}, N cuBLAS calls, one sync.
     * Used for Q/K/V and gate/up so consecutive projections do not re-upload {@code x}.
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
                    CudaBindings.check(
                            CudaBindings.callInt(cuda.cudaMemcpyAsync,
                                    scratch.dXh, stagingXh, bytesXh, CudaBindings.H2D, stream),
                            "cudaMemcpyAsync(xh H2D sameX)");
                    spans.staging(CudaBindings.H2D, bytesXh, 1, "cudaMemcpyAsync(xh H2D sameX)", h2dMark, stream);

                    MemorySegment[] stagingY = new MemorySegment[n];
                    for (int i = 0; i < n; i++) {
                        DeviceHalfMatrix A = weights[i];
                        int rows = A.rows();
                        callSgemvFp16(CudaBindings.CUBLAS_OP_T, A.devicePointer(), cols,
                                scratch.dXh, scratch.dY, rows, cols, 1);
                        long bytesY = (long) rows * Float.BYTES;
                        stagingY[i] = callArena.allocate(bytesY);
                        int d2hMark = spans.begin(stream, 1);
                        CudaBindings.check(
                                CudaBindings.callInt(cuda.cudaMemcpyAsync,
                                        stagingY[i], scratch.dY, bytesY, CudaBindings.D2H, stream),
                                "cudaMemcpyAsync(y D2H sameX)");
                        spans.staging(CudaBindings.D2H, bytesY, 1, "cudaMemcpyAsync(y D2H sameX)", d2hMark, stream);
                    }
                    CudaBindings.check(
                            CudaBindings.callInt(cuda.cudaStreamSynchronize, stream),
                            "cudaStreamSynchronize");
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
            evt.backend(MatVecBackend.CUDA_RESIDENT_FP16);
            evt.rows = maxRows;
            evt.cols = cols;
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
                    try (Arena h2dArena = Arena.ofConfined()) {
                        MemorySegment nativeX = h2dArena.allocate(bytesX);
                        nativeX.copyFrom(MemorySegment.ofArray(x));
                        int h2dMark = spans.begin(stream, 1);
                        CudaBindings.check(
                                CudaBindings.callInt(cuda.cudaMemcpyAsync,
                                        scratch.dX, nativeX, bytesX, CudaBindings.H2D, stream),
                                "cudaMemcpyAsync(x H2D sameX)");
                        spans.staging(CudaBindings.H2D, bytesX, 1, "cudaMemcpyAsync(x H2D sameX)", h2dMark, stream);
                    }
                    MemorySegment[] stagingY = new MemorySegment[n];
                    for (int i = 0; i < n; i++) {
                        DeviceFloatMatrix A = weights[i];
                        int rows = A.rows();
                        callSgemvFp32(CudaBindings.CUBLAS_OP_T, A.devicePointer(), cols,
                                scratch.dX, scratch.dY, rows, cols);
                        long bytesY = (long) rows * Float.BYTES;
                        stagingY[i] = callArena.allocate(bytesY);
                        int d2hMark = spans.begin(stream, 1);
                        CudaBindings.check(
                                CudaBindings.callInt(cuda.cudaMemcpyAsync,
                                        stagingY[i], scratch.dY, bytesY, CudaBindings.D2H, stream),
                                "cudaMemcpyAsync(y D2H sameX)");
                        spans.staging(CudaBindings.D2H, bytesY, 1, "cudaMemcpyAsync(y D2H sameX)", d2hMark, stream);
                    }
                    CudaBindings.check(
                            CudaBindings.callInt(cuda.cudaStreamSynchronize, stream),
                            "cudaStreamSynchronize");
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
            evt.backend(MatVecBackend.CUDA_RESIDENT);
            evt.rows = maxRows;
            evt.cols = cols;
            evt.commit();
        }
    }

    @Override
    public float[][] sgemvSameX(DeviceQ4KMatrix[] weights, float[] x) {
        if (weights == null || weights.length == 0)
            return new float[0][];
        if (weights.length == 1)
            return new float[][] { sgemv(weights[0], x) };
        Q4KMmqKernel kernel = Q4KMmqKernel.tryLoad();
        if (kernel == null)
            throw new IllegalStateException("Q4_K MMQ kernel is not loaded");
        int cols = x.length;
        int n = weights.length;
        int maxRows = 0;
        for (int i = 0; i < n; i++) {
            DeviceQ4KMatrix A = weights[i];
            if (A == null)
                throw new IllegalArgumentException("weights[" + i + "] is null");
            if (A.isClosed())
                throw new IllegalStateException("DeviceQ4KMatrix is closed");
            if (A.cols() != cols)
                throw new IllegalArgumentException(
                        "weights[" + i + "].cols=" + A.cols() + " != x.length=" + cols);
            maxRows = Math.max(maxRows, A.rows());
        }

        MatVecEvent evt = new MatVecEvent();
        evt.begin();
        long bytesX = (long) cols * Float.BYTES;
        long bytesQ8 = Q4KMmqKernel.q8Bytes(cols);
        int[] yOffElems = new int[n];
        int totalY = 0;
        for (int i = 0; i < n; i++) {
            yOffElems[i] = totalY;
            totalY += weights[i].rows();
        }
        Fp32Scratch scratch = fp32Scratch;
        float[][] Y = new float[n][];

        try (Arena callArena = Arena.ofConfined()) {
            synchronized (ctx.cublasSerializationLock()) {
                MemorySegment stream = ensureStream();
                spans.reset();
                try {
                    ensureFp32Scratch(scratch, bytesX, (long) totalY * Float.BYTES, bytesQ8);
                    try (Arena h2dArena = Arena.ofConfined()) {
                        MemorySegment nativeX = h2dArena.allocate(bytesX);
                        nativeX.copyFrom(MemorySegment.ofArray(x));
                        int h2dMark = spans.begin(stream, 1);
                        CudaBindings.check(
                                CudaBindings.callInt(cuda.cudaMemcpyAsync,
                                        scratch.dX, nativeX, bytesX, CudaBindings.H2D, stream),
                                "cudaMemcpyAsync(x H2D q4k sameX)");
                        spans.staging(CudaBindings.H2D, bytesX, 1, "cudaMemcpyAsync(x H2D q4k sameX)", h2dMark, stream);
                    }
                    kernel.quantizeX(scratch.dX, scratch.dQ8, cols, stream);
                    MemorySegment[] stagingY = new MemorySegment[n];
                    for (int i = 0; i < n; i++) {
                        DeviceQ4KMatrix A = weights[i];
                        int rows = A.rows();
                        long bytesY = (long) rows * Float.BYTES;
                        MemorySegment dYi = scratch.dY.asSlice((long) yOffElems[i] * Float.BYTES, bytesY);
                        kernel.launchPacked(A.devicePointer(), scratch.dQ8, dYi, rows, cols, A.quantType(), stream);
                        spans.compute(DeviceComputeEvent.MMQ_PACKED, 1, -1, stream);
                        stagingY[i] = callArena.allocate(bytesY);
                        int d2hMark = spans.begin(stream, 1);
                        CudaBindings.check(
                                CudaBindings.callInt(cuda.cudaMemcpyAsync,
                                        stagingY[i], dYi, bytesY, CudaBindings.D2H, stream),
                                "cudaMemcpyAsync(y D2H q4k sameX)");
                        spans.staging(CudaBindings.D2H, bytesY, 1, "cudaMemcpyAsync(y D2H q4k sameX)", d2hMark, stream);
                    }
                    CudaBindings.check(
                            CudaBindings.callInt(cuda.cudaStreamSynchronize, stream),
                            "cudaStreamSynchronize");
                    spans.commit();
                    for (int i = 0; i < n; i++) {
                        int rows = weights[i].rows();
                        Y[i] = new float[rows];
                        MemorySegment.copy(stagingY[i], JAVA_FLOAT, 0, Y[i], 0, rows);
                    }
                    return Y;
                } finally {
                    // no cuBLAS stream bind
                }
            }
        } finally {
            evt.backend(MatVecBackend.CUDA_RESIDENT_Q4K);
            evt.rows = maxRows;
            evt.cols = cols;
            evt.commit();
        }
    }

    /**
     * Batched FP32 GEMM: one {@code cublasSgemm} for {@code batch > 1}; otherwise
     * delegates to {@link #sgemv(DeviceFloatMatrix, float[])}.
     */
    @Override
    public float[][] sgemm(DeviceFloatMatrix A, float[][] X) {
        if (A == null) throw new IllegalArgumentException("A must not be null");
        float[][] Y = new float[X.length][A.rows()];
        sgemmInto(A, X, Y);
        return Y;
    }

    /**
     * {@link #sgemm(DeviceFloatMatrix, float[][])} into the caller's rows. The rows
     * are not allocated; the FP32 GEMM's own packed host staging still is
     * ({@link GpuBlasOps#forwardInto}).
     */
    @Override
    public void sgemmInto(DeviceFloatMatrix A, float[][] X, float[][] Y) {
        if (A == null) throw new IllegalArgumentException("A must not be null");
        SgemmOutput.require(X, Y, A.rows());
        if (X.length <= 1) {
            for (int b = 0; b < X.length; b++)
                sgemvInto(A, X[b], Y[b]);
            return;
        }
        blasOps().forwardInto(A, X, X.length, Y);
    }

    /** Max batch for the strided-batched GEMV kernel (multi-request decode); larger (prefill) windows use the tiled GEMM. */
    private static final int HALF_SGEMM_BATCH_MAX = 8;

    /**
     * Batched FP16-weight GEMM: {@code cublasHSSgemvStridedBatched} for small decode
     * batches (bounded per-call overhead dominates, no weight reuse needed);
     * {@code cublasGemmEx} tiled GEMM for large prefill windows (weight-stationary
     * reuse across the batch — see {@link CudaFp16GemmOps}); {@code batch <= 1} stays
     * serial {@link #sgemv}.
     */
    @Override
    public float[][] sgemm(DeviceHalfMatrix A, float[][] X) {
        if (A == null) throw new IllegalArgumentException("A must not be null");
        float[][] Y = new float[X.length][A.rows()];
        sgemmInto(A, X, Y);
        return Y;
    }

    /** {@link #sgemm(DeviceHalfMatrix, float[][])} into the caller's rows, without allocating them. */
    @Override
    public void sgemmInto(DeviceHalfMatrix A, float[][] X, float[][] Y) {
        if (A == null) throw new IllegalArgumentException("A must not be null");
        SgemmOutput.require(X, Y, A.rows());
        if (X.length <= 1) {
            for (int b = 0; b < X.length; b++)
                sgemvInto(A, X[b], Y[b]);
            return;
        }
        if (X.length > HALF_SGEMM_BATCH_MAX)
            sgemmHalfBatchedGemm(A, X, Y);
        else
            sgemmHalfBatched(A, X, Y);
    }

    private void sgemmHalfBatched(DeviceHalfMatrix A, float[][] X, float[][] Y) {
        int batch = X.length;
        int rows = A.rows();
        int cols = A.cols();
        for (int b = 0; b < batch; b++) {
            if (X[b].length != cols)
                throw new IllegalArgumentException("X[" + b + "].length != cols");
        }

        MatVecEvent evt = new MatVecEvent();
        evt.windowSize = batch;
        evt.begin();

        long bytesXh = (long) cols * batch * Short.BYTES;
        long bytesY = (long) rows * batch * Float.BYTES;
        Fp16Scratch scratch = fp16Scratch;

        try {
            synchronized (ctx.cublasSerializationLock()) {
                MemorySegment stream = ensureStream();
                spans.reset();
                bindStream(stream);
                try {
                    ensureFp16Scratch(scratch, bytesXh, bytesY);

                    MemorySegment stagingXh = scratch.hXh;
                    packFp16Window(stagingXh, X, batch, cols, bytesXh);

                    int h2dMark = spans.begin(stream, batch);
                    CudaBindings.check(
                            CudaBindings.callInt(cuda.cudaMemcpyAsync,
                                    scratch.dXh, stagingXh, bytesXh, CudaBindings.H2D, stream),
                            "cudaMemcpyAsync(xh H2D batched)");
                    spans.staging(CudaBindings.H2D, bytesXh, batch, "cudaMemcpyAsync(xh H2D batched)", h2dMark, stream);

                    int gemvMark = spans.begin(stream, batch);
                    callSgemvFp16(CudaBindings.CUBLAS_OP_T, A.devicePointer(), cols, scratch.dXh, scratch.dY,
                            rows, cols, batch);
                    spans.compute(DeviceComputeEvent.GEMV_HALF_BATCHED, batch, gemvMark, stream);

                    MemorySegment stagingY = scratch.hY;
                    int d2hMark = spans.begin(stream, batch);
                    CudaBindings.check(
                            CudaBindings.callInt(cuda.cudaMemcpyAsync,
                                    stagingY, scratch.dY, bytesY, CudaBindings.D2H, stream),
                            "cudaMemcpyAsync(y D2H batched)");
                    spans.staging(CudaBindings.D2H, bytesY, batch, "cudaMemcpyAsync(y D2H batched)", d2hMark, stream);
                    CudaBindings.check(
                            CudaBindings.callInt(cuda.cudaStreamSynchronize, stream),
                            "cudaStreamSynchronize");
                    spans.commit();

                    for (int b = 0; b < batch; b++)
                        MemorySegment.copy(stagingY, JAVA_FLOAT, (long) b * rows * Float.BYTES, Y[b], 0, rows);
                } finally {
                    unbindStream();
                }
            }
        } finally {
            evt.backend(MatVecBackend.CUDA_RESIDENT_FP16);
            evt.rows = rows;
            evt.cols = cols;
            evt.commit();
        }
    }

    /**
     * Tiled-GEMM counterpart to {@link #sgemmHalfBatched}: same staging/H2D/D2H shape
     * (and the same {@link Fp16Scratch} buffer — {@code sgemmHalfBatched} already grows
     * it to {@code batch}-scaled sizes, so no separate scratch container is needed for
     * larger batches), but compute goes through {@link CudaFp16GemmOps#gemmHalf} —
     * {@code cublasGemmEx}, a real weight-stationary tiled GEMM — instead of
     * {@code cublasHSSgemvStridedBatched}, which does not gain compute/bandwidth reuse
     * across the batch as batch size grows.
     */
    private void sgemmHalfBatchedGemm(DeviceHalfMatrix A, float[][] X, float[][] Y) {
        int batch = X.length;
        int rows = A.rows();
        int cols = A.cols();
        for (int b = 0; b < batch; b++) {
            if (X[b].length != cols)
                throw new IllegalArgumentException("X[" + b + "].length != cols");
        }

        MatVecEvent evt = new MatVecEvent();
        evt.windowSize = batch;
        evt.begin();

        long bytesXh = (long) cols * batch * Short.BYTES;
        long bytesY = (long) rows * batch * Float.BYTES;
        Fp16Scratch scratch = fp16Scratch;

        try {
            synchronized (ctx.cublasSerializationLock()) {
                MemorySegment stream = ensureStream();
                spans.reset();
                bindStream(stream);
                try {
                    ensureFp16Scratch(scratch, bytesXh, bytesY);

                    MemorySegment stagingXh = scratch.hXh;
                    packFp16Window(stagingXh, X, batch, cols, bytesXh);

                    int h2dMark = spans.begin(stream, batch);
                    CudaBindings.check(
                            CudaBindings.callInt(cuda.cudaMemcpyAsync,
                                    scratch.dXh, stagingXh, bytesXh, CudaBindings.H2D, stream),
                            "cudaMemcpyAsync(xh H2D batched-gemm)");
                    spans.staging(CudaBindings.H2D, bytesXh, batch, "cudaMemcpyAsync(xh H2D batched-gemm)", h2dMark, stream);

                    int gemmMark = spans.begin(stream, batch);
                    fp16GemmOps().gemmHalf(A.devicePointer(), scratch.dXh, scratch.dY, rows, cols, batch);
                    spans.compute(DeviceComputeEvent.GEMM_HALF, batch, gemmMark, stream);

                    MemorySegment stagingY = scratch.hY;
                    int d2hMark = spans.begin(stream, batch);
                    CudaBindings.check(
                            CudaBindings.callInt(cuda.cudaMemcpyAsync,
                                    stagingY, scratch.dY, bytesY, CudaBindings.D2H, stream),
                            "cudaMemcpyAsync(y D2H batched-gemm)");
                    spans.staging(CudaBindings.D2H, bytesY, batch, "cudaMemcpyAsync(y D2H batched-gemm)", d2hMark, stream);
                    CudaBindings.check(
                            CudaBindings.callInt(cuda.cudaStreamSynchronize, stream),
                            "cudaStreamSynchronize");
                    spans.commit();

                    for (int b = 0; b < batch; b++)
                        MemorySegment.copy(stagingY, JAVA_FLOAT, (long) b * rows * Float.BYTES, Y[b], 0, rows);
                } finally {
                    unbindStream();
                }
            }
        } finally {
            evt.backend(MatVecBackend.CUDA_RESIDENT_FP16_GEMM);
            evt.rows = rows;
            evt.cols = cols;
            evt.commit();
        }
    }

    /** The {@code juno.DeviceStaging} site of the host FP16 packing {@link #packFp16Window} times. */
    static final String PACK_FP16_HOST = "pack_fp16_host";

    /**
     * {@link #packFp16Rows}, counted as {@code HOST} staging work (timed on the host
     * clock) when a recording asks for {@code juno.DeviceStaging}: it exists only to
     * stage the window's upload, and is otherwise inside {@code juno.MatVec} with no
     * span of its own.
     */
    private static void packFp16Window(MemorySegment dst, float[][] X, int batch, int cols, long bytes) {
        if (!DeviceSpanTally.stagingWanted()) {
            packFp16Rows(dst, X, batch, cols);
            return;
        }
        long t0 = System.nanoTime();
        packFp16Rows(dst, X, batch, cols);
        DeviceSpanTally.staging(PACK_FP16_HOST, DeviceStagingEvent.HOST_WORK, batch, bytes, System.nanoTime() - t0);
    }

    /**
     * Packs {@code batch} rows of {@code X} as FP16 into {@code dst}, row-major.
     *
     * <p>Its own method on purpose: the batched callers are large and deoptimize
     * whenever an uncommon branch runs (a scratch buffer growing to a new largest
     * window, for example), and with the loop inline, the rest of that call packed
     * a prefill window in the interpreter - measured at 17 to 38 ms instead of
     * about 1 ms per matmul. Compiled on its own, the loop stays compiled when its
     * caller does not.
     */
    private static void packFp16Rows(MemorySegment dst, float[][] X, int batch, int cols) {
        for (int b = 0; b < batch; b++) {
            float[] row = X[b];
            long base = (long) b * cols;
            for (int j = 0; j < cols; j++)
                dst.setAtIndex(JAVA_SHORT, base + j, Float.floatToFloat16(row[j]));
        }
    }

    private CudaFp16GemmOps fp16GemmOps() {
        if (fp16GemmOps == null)
            fp16GemmOps = new CudaFp16GemmOps(ctx);
        return fp16GemmOps;
    }

    /**
     * Batched Q4_K/Q5_K/Q6_K GEMM: {@code batch <= HALF_SGEMM_BATCH_MAX} stays serial
     * {@link #sgemv(DeviceQ4KMatrix, float[])}; {@code batch > HALF_SGEMM_BATCH_MAX}
     * stages the window as FP16, as the FP16-weight path does, and multiplies the
     * still-packed weights with the tiled integer kernel ({@link KQuantGemmKernel}):
     * no FP16 copy of the weights is made. Does not touch the single-token
     * {@link #sgemv(DeviceQ4KMatrix, float[])} decode path.
     */
    @Override
    public float[][] sgemm(DeviceQ4KMatrix A, float[][] X) {
        if (A == null) throw new IllegalArgumentException("A must not be null");
        float[][] Y = new float[X.length][A.rows()];
        sgemmInto(A, X, Y);
        return Y;
    }

    /** {@link #sgemm(DeviceQ4KMatrix, float[][])} into the caller's rows, without allocating them. */
    @Override
    public void sgemmInto(DeviceQ4KMatrix A, float[][] X, float[][] Y) {
        if (A == null) throw new IllegalArgumentException("A must not be null");
        if (A.isClosed()) throw new IllegalStateException("DeviceQ4KMatrix is closed");
        SgemmOutput.require(X, Y, A.rows());
        if (X.length <= HALF_SGEMM_BATCH_MAX) {
            for (int b = 0; b < X.length; b++)
                sgemvInto(A, X[b], Y[b]);
            return;
        }
        sgemmQ4KBatchedGemm(A, X, Y);
    }

    private void sgemmQ4KBatchedGemm(DeviceQ4KMatrix A, float[][] X, float[][] Y) {
        int batch = X.length;
        int rows = A.rows();
        int cols = A.cols();
        for (int b = 0; b < batch; b++) {
            if (X[b].length != cols)
                throw new IllegalArgumentException("X[" + b + "].length != cols");
        }

        MatVecEvent evt = new MatVecEvent();
        evt.windowSize = batch;
        evt.begin();

        long bytesXh = (long) cols * batch * Short.BYTES;
        long bytesY = (long) rows * batch * Float.BYTES;
        Fp16Scratch scratch = fp16Scratch;

        try {
            synchronized (ctx.cublasSerializationLock()) {
                MemorySegment stream = ensureStream();
                spans.reset();
                bindStream(stream);
                try {
                    ensureFp16Scratch(scratch, bytesXh, bytesY);

                    MemorySegment stagingXh = scratch.hXh;
                    packFp16Window(stagingXh, X, batch, cols, bytesXh);

                    int h2dMark = spans.begin(stream, batch);
                    CudaBindings.check(
                            CudaBindings.callInt(cuda.cudaMemcpyAsync,
                                    scratch.dXh, stagingXh, bytesXh, CudaBindings.H2D, stream),
                            "cudaMemcpyAsync(xh H2D q4k-batched-gemm)");
                    spans.staging(CudaBindings.H2D, bytesXh, batch, "cudaMemcpyAsync(xh H2D q4k-batched-gemm)", h2dMark, stream);

                    gemmPackedOnStream(A, scratch.dXh, scratch.dY, rows, batch, stream, spans);

                    MemorySegment stagingY = scratch.hY;
                    int d2hMark = spans.begin(stream, batch);
                    CudaBindings.check(
                            CudaBindings.callInt(cuda.cudaMemcpyAsync,
                                    stagingY, scratch.dY, bytesY, CudaBindings.D2H, stream),
                            "cudaMemcpyAsync(y D2H q4k-batched-gemm)");
                    spans.staging(CudaBindings.D2H, bytesY, batch, "cudaMemcpyAsync(y D2H q4k-batched-gemm)", d2hMark, stream);
                    CudaBindings.check(
                            CudaBindings.callInt(cuda.cudaStreamSynchronize, stream),
                            "cudaStreamSynchronize");
                    spans.commit();

                    for (int b = 0; b < batch; b++)
                        MemorySegment.copy(stagingY, JAVA_FLOAT, (long) b * rows * Float.BYTES, Y[b], 0, rows);
                } finally {
                    unbindStream();
                }
            }
        } finally {
            evt.backend(MatVecBackend.CUDA_RESIDENT_Q4K_GEMM);
            evt.rows = rows;
            evt.cols = cols;
            evt.commit();
        }
    }

    /**
     * {@code Y = A X} on operands already on the device, issued on {@code stream}
     * without waiting: the prefill-window region's matmul. {@code dXh} holds
     * {@code batch} FP16 input rows of {@code A.cols()} values; row {@code b} of the
     * FP32 result is written at {@code dY + b * ldc} floats. The packed weights are
     * multiplied as they are ({@link #gemmPackedOnStream}): the same kernels and the
     * same bits as {@link #sgemm(DeviceQ4KMatrix, float[][])}'s batched path.
     *
     * <p>The caller holds {@link GpuContext#cublasSerializationLock()} and
     * synchronizes {@code stream} before releasing it: the Q8_1 window scratch is
     * shared with this backend's own calls, which run on another stream. Device time
     * is counted on {@code timer} (the caller's stream timer), not on this backend's.
     */
    void gemmOnStream(DeviceQ4KMatrix A, MemorySegment dXh, MemorySegment dY, int ldc, int batch,
            MemorySegment stream, DeviceSpanTimer timer) {
        if (A == null) throw new IllegalArgumentException("A must not be null");
        if (A.isClosed()) throw new IllegalStateException("DeviceQ4KMatrix is closed");
        gemmPackedOnStream(A, dXh, dY, ldc, batch, stream, timer);
    }

    /**
     * The batched K-quant matmul both prefill paths share: packs the FP16 window
     * {@code dXh} as Q8_1 into this backend's scratch and multiplies the still-packed
     * weights with the tiled integer kernel ({@link KQuantGemmKernel}), timed as one
     * {@code gemm_kquant} site. Nothing is dequantized. When the tiled kernel's module
     * did not load, falls back to dequantizing to FP16 and the FP16 GEMM, which gives
     * the same product at higher precision, and says so once. Same locking contract
     * as {@link #gemmOnStream}.
     */
    private void gemmPackedOnStream(DeviceQ4KMatrix A, MemorySegment dXh, MemorySegment dY, int ldc, int batch,
            MemorySegment stream, DeviceSpanTimer timer) {
        KQuantGemmKernel tiled = dequantBatchedKQuant ? null : KQuantGemmKernel.tryLoad();
        if (tiled == null) {
            if (!dequantBatchedKQuant && tiledFallbackWarned.compareAndSet(false, true))
                log.warning("Tiled K-quant GEMM kernel unavailable: batched K-quant matmuls dequantize their"
                        + " weights to FP16 instead (slower, more device memory)");
            MemorySegment dW = dequantOnStream(A, batch, stream, timer);
            gemmHalfOnStream(dW, A.rows(), A.cols(), dXh, dY, ldc, batch, stream, timer);
            return;
        }
        MemorySegment dQ8 = q8WindowScratch.ensure(cuda, ctx.deviceIndex(), KQuantGemmKernel.q8Bytes(batch, A.cols()));
        int mark = timer.begin(stream, batch);
        tiled.multiplyHalf(A, dXh, dQ8, dY, batch, ldc, stream);
        timer.compute(DeviceComputeEvent.GEMM_KQUANT, batch, mark, stream);
    }

    /**
     * Sends this backend's batched K-quant matmuls down the FP16 dequant route
     * instead of the tiled integer kernel. Not a product setting: for parity tests
     * that compare two runs differing in some other operation (attention on or off,
     * say) and were calibrated with FP16 activations. The tiled kernel rounds each
     * matmul's input to 8 bits, which turns a difference of 3e-4 between the two
     * runs into whole rounding steps, and would hide what such a test measures.
     */
    void dequantizeBatchedKQuant(boolean on) {
        dequantBatchedKQuant = on;
    }

    /**
     * Dequantizes {@code A} to FP16 into this backend's scratch on {@code stream} and
     * returns the scratch: row-major {@code A.rows() x A.cols()} halves, valid until
     * the next dequantization on this backend. The route batched K-quant matmuls
     * took before the tiled kernel; kept as {@link #gemmPackedOnStream}'s fallback
     * and as the higher-precision reference the tests compare against. Same locking
     * contract as {@link #gemmOnStream}.
     */
    MemorySegment dequantOnStream(DeviceQ4KMatrix A, int batch, MemorySegment stream, DeviceSpanTimer timer) {
        if (A == null) throw new IllegalArgumentException("A must not be null");
        if (A.isClosed()) throw new IllegalStateException("DeviceQ4KMatrix is closed");
        Q4KMmqKernel kernel = Q4KMmqKernel.tryLoad();
        if (kernel == null)
            throw new IllegalStateException("Q4_K MMQ kernel is not loaded");
        MemorySegment dW = dequantScratch.ensure(cuda, ctx.deviceIndex(), A.rows(), A.cols());
        int dequantMark = timer.begin(stream, batch);
        kernel.launchDequant(A, dW, stream);
        timer.dequant(A.quantType(), dequantMark, stream);
        return dW;
    }

    /** As {@link #gemmOnStream(DeviceQ4KMatrix, MemorySegment, MemorySegment, int, int, MemorySegment, DeviceSpanTimer)}, for FP16 weights. */
    void gemmOnStream(DeviceHalfMatrix A, MemorySegment dXh, MemorySegment dY, int ldc, int batch,
            MemorySegment stream, DeviceSpanTimer timer) {
        if (A == null) throw new IllegalArgumentException("A must not be null");
        gemmHalfOnStream(A.devicePointer(), A.rows(), A.cols(), dXh, dY, ldc, batch, stream, timer);
    }

    /**
     * {@code Y = A X} for FP16 weights {@code dA} (row-major {@code rows x cols},
     * which may be a row range of a larger matrix) on device operands, issued on
     * {@code stream}. Same layout and locking contract as {@link #gemmOnStream}.
     */
    void gemmHalfOnStream(MemorySegment dA, int rows, int cols, MemorySegment dXh, MemorySegment dY, int ldc,
            int batch, MemorySegment stream, DeviceSpanTimer timer) {
        bindStream(stream);
        try {
            int gemmMark = timer.begin(stream, batch);
            fp16GemmOps().gemmHalf(dA, dXh, dY, rows, cols, batch, ldc);
            timer.compute(DeviceComputeEvent.GEMM_HALF, batch, gemmMark, stream);
        } finally {
            unbindStream();
        }
    }

    private GpuBlasOps blasOps() {
        if (blasOps == null)
            blasOps = new GpuBlasOps(ctx);
        return blasOps;
    }

    /**
     * Device-resident FP32 transpose: {@code z = W^T * g} for row-major
     * {@code W[rows×cols]}. {@code g} length {@code rows}; result length {@code cols}.
     *
     * <p>Uses {@link CudaBindings#CUBLAS_OP_N} with the same {@code (m,n,lda)}
     * mapping as forward ({@code m=cols}, {@code n=rows}, {@code lda=cols}).
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

        // Input g has length rows; output z has length cols (roles swapped vs forward).
        long bytesG = (long) rows * Float.BYTES;
        long bytesZ = (long) cols * Float.BYTES;

        Fp32Scratch scratch = fp32Scratch;

        try (Arena resultArena = Arena.ofConfined()) {
            synchronized (ctx.cublasSerializationLock()) {
                MemorySegment stream = ensureStream();
                spans.reset();
                bindStream(stream);
                try {
                    ensureFp32Scratch(scratch, bytesG, bytesZ);

                    try (Arena h2dArena = Arena.ofConfined()) {
                        MemorySegment nativeG = h2dArena.allocate(bytesG);
                        nativeG.copyFrom(MemorySegment.ofArray(g));
                        int h2dMark = spans.begin(stream, 1);
                        CudaBindings.check(
                            CudaBindings.callInt(cuda.cudaMemcpyAsync,
                                scratch.dX, nativeG, bytesG, CudaBindings.H2D, stream),
                            "cudaMemcpyAsync(g H2D)");
                        spans.staging(CudaBindings.H2D, bytesG, 1, "cudaMemcpyAsync(g H2D)", h2dMark, stream);
                    }

                    callSgemvFp32(CudaBindings.CUBLAS_OP_N, W.devicePointer(), cols,
                            scratch.dX, scratch.dY, rows, cols);

                    MemorySegment stagingZ = resultArena.allocate(bytesZ);
                    int d2hMark = spans.begin(stream, 1);
                    CudaBindings.check(
                        CudaBindings.callInt(cuda.cudaMemcpyAsync,
                            stagingZ, scratch.dY, bytesZ, CudaBindings.D2H, stream),
                        "cudaMemcpyAsync(z D2H)");
                    spans.staging(CudaBindings.D2H, bytesZ, 1, "cudaMemcpyAsync(z D2H)", d2hMark, stream);
                    CudaBindings.check(
                        CudaBindings.callInt(cuda.cudaStreamSynchronize, stream),
                        "cudaStreamSynchronize");
                    spans.commit();

                    float[] z = new float[cols];
                    MemorySegment.copy(stagingZ, JAVA_FLOAT, 0, z, 0, cols);
                    return z;
                } finally {
                    unbindStream();
                }
            }
        } finally {
            evt.backend(MatVecBackend.CUDA_RESIDENT_TRANSPOSE);
            evt.rows = cols; // output length
            evt.cols = rows; // input length
            evt.commit();
        }
    }

    /**
     * Device-resident FP16 transpose: same layout contract as
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
                    CudaBindings.check(
                        CudaBindings.callInt(cuda.cudaMemcpyAsync,
                            scratch.dXh, stagingGh, bytesGh, CudaBindings.H2D, stream),
                        "cudaMemcpyAsync(gh H2D)");
                    spans.staging(CudaBindings.H2D, bytesGh, 1, "cudaMemcpyAsync(gh H2D)", h2dMark, stream);

                    callSgemvFp16(CudaBindings.CUBLAS_OP_N, W.devicePointer(), cols,
                            scratch.dXh, scratch.dY, rows, cols, 1);

                    MemorySegment stagingZ = callArena.allocate(bytesZ);
                    int d2hMark = spans.begin(stream, 1);
                    CudaBindings.check(
                        CudaBindings.callInt(cuda.cudaMemcpyAsync,
                            stagingZ, scratch.dY, bytesZ, CudaBindings.D2H, stream),
                        "cudaMemcpyAsync(z D2H)");
                    spans.staging(CudaBindings.D2H, bytesZ, 1, "cudaMemcpyAsync(z D2H)", d2hMark, stream);
                    CudaBindings.check(
                        CudaBindings.callInt(cuda.cudaStreamSynchronize, stream),
                        "cudaStreamSynchronize");
                    spans.commit();

                    float[] z = new float[cols];
                    MemorySegment.copy(stagingZ, JAVA_FLOAT, 0, z, 0, cols);
                    return z;
                } finally {
                    unbindStream();
                }
            }
        } finally {
            evt.backend(MatVecBackend.CUDA_RESIDENT_FP16_TRANSPOSE);
            evt.rows = cols;
            evt.cols = rows;
            evt.commit();
        }
    }

    // ── cuBLAS kernel dispatchers ─────────────────────────────────────────────

    /**
     * cublasSgemv_v2 on row-major {@code W[rows×cols]}.
     *
     * <p>cuBLAS is column-major. The same buffer is interpreted with
     * {@code m=cols}, {@code n=rows}, {@code lda=cols}:
     * <ul>
     *   <li>{@link CudaBindings#CUBLAS_OP_T} → forward {@code y = W * x}
     *       ({@code x} length cols, {@code y} length rows)</li>
     *   <li>{@link CudaBindings#CUBLAS_OP_N} → transpose {@code z = W^T * g}
     *       ({@code g} length rows, {@code z} length cols)</li>
     * </ul>
     */
    private void callSgemvFp32(int op, MemorySegment dA, int lda,
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
                    ctx.handle(), op,
                    cols, rows,
                    alpha, dA, lda,
                    dX, 1,
                    beta, dY, 1),
                "cublasSgemv_v2");
        }
    }

    /**
     * cublasHSSgemvStridedBatched: same {@code (op, m, n, lda)} mapping as
     * {@link #callSgemvFp32}.
     */
    private void callSgemvFp16(int op, MemorySegment dA, int lda,
                                MemorySegment dXh, MemorySegment dY,
                                int rows, int cols, int batch) {
        // strideA=0: one shared weight matrix for every batch column (decode / microbatch).
        long strideA = 0;
        long strideX = (op == CudaBindings.CUBLAS_OP_N) ? rows : cols;
        long strideY = (op == CudaBindings.CUBLAS_OP_N) ? cols : rows;
        try (Arena scalars = Arena.ofConfined()) {
            MemorySegment alpha = scalars.allocateFrom(JAVA_FLOAT, 1.0f);
            MemorySegment beta  = scalars.allocateFrom(JAVA_FLOAT, 0.0f);
            CudaBindings.check(
                CudaBindings.callInt(cuda.cublasSetPointerMode, ctx.handle(), CudaBindings.CUBLAS_POINTER_MODE_HOST),
                "cublasSetPointerMode");
            CudaBindings.check(
                CudaBindings.callInt(cuda.cublasHSSgemvStridedBatched,
                    ctx.handle(), op,
                    cols, rows,
                    alpha, dA, lda, strideA,
                    dXh, 1, strideX,
                    beta, dY, 1, strideY,
                    batch),
                "cublasHSSgemvStridedBatched");
        }
    }

    // ── Stream management ─────────────────────────────────────────────────────

    /** Returns or lazily creates the instance's non-blocking CUDA stream. Caller holds the lock. */
    private MemorySegment ensureStream() {
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

    // ── Scratch lifetime ──────────────────────────────────────────────────────

    /**
     * Frees this instance's device scratch, pinned host staging, timing events and stream.
     * Safe while other callers are active: it takes the same lock they hold, and
     * the next call grows the scratch again. Not needed per request; for callers
     * that retire a backend, and for tests that need the device back to baseline.
     */
    void releaseScratch() {
        synchronized (ctx.cublasSerializationLock()) {
            freeDevice(fp32Scratch.dX);
            freeDevice(fp32Scratch.dY);
            freeDevice(fp32Scratch.dQ8);
            fp32Scratch.dX = fp32Scratch.dY = fp32Scratch.dQ8 = null;
            fp32Scratch.dXBytes = fp32Scratch.dYBytes = fp32Scratch.dQ8Bytes = 0L;
            freeDevice(fp16Scratch.dXh);
            freeDevice(fp16Scratch.dY);
            fp16Scratch.dXh = fp16Scratch.dY = null;
            fp16Scratch.dXhBytes = fp16Scratch.dYBytes = 0L;
            if (fp16Scratch.hXh != null)
                cuda.hostFree(fp16Scratch.hXh);
            if (fp16Scratch.hY != null)
                cuda.hostFree(fp16Scratch.hY);
            fp16Scratch.hXh = fp16Scratch.hY = null;
            fp16Scratch.hXhBytes = fp16Scratch.hYBytes = 0L;
            dequantScratch.release(cuda);
            q8WindowScratch.release(cuda);
            spans.releaseEvents();
            if (stream != null) {
                CudaBindings.callInt(cuda.cudaStreamDestroy, stream);
                stream = null;
            }
        }
    }

    /** Device bytes this instance's scratch holds now (excluding the stream). */
    long scratchDeviceBytes() {
        synchronized (ctx.cublasSerializationLock()) {
            return fp32Scratch.dXBytes + fp32Scratch.dYBytes + fp32Scratch.dQ8Bytes
                    + fp16Scratch.dXhBytes + fp16Scratch.dYBytes + dequantScratch.heldBytes()
                    + q8WindowScratch.heldBytes();
        }
    }

    private void freeDevice(MemorySegment p) {
        if (p != null)
            cuda.deviceFree(p);
    }

    // ── Scratch growth ────────────────────────────────────────────────────────

    private void ensureFp32Scratch(Fp32Scratch s, long bytesX, long bytesY) {
        ensureFp32Scratch(s, bytesX, bytesY, 0);
    }

    /**
     * Grows the FP32 scratch. As in {@link #ensureFp16Scratch}, each grow clears
     * its field before freeing, so a failed allocation leaves the slot empty
     * rather than holding a freed pointer the next caller would reuse: the
     * scratch now outlives the call and the thread that failed.
     */
    private void ensureFp32Scratch(Fp32Scratch s, long bytesX, long bytesY, long bytesQ8) {
        int dev = ctx.deviceIndex();
        if (s.dXBytes < bytesX) {
            MemorySegment previous = s.dX;
            s.dX = null;
            s.dXBytes = 0L;
            if (previous != null)
                cuda.deviceFree(previous);
            s.dX     = cuda.deviceMalloc(dev, bytesX);
            s.dXBytes = bytesX;
        }
        if (s.dYBytes < bytesY) {
            MemorySegment previous = s.dY;
            s.dY = null;
            s.dYBytes = 0L;
            if (previous != null)
                cuda.deviceFree(previous);
            s.dY     = cuda.deviceMalloc(dev, bytesY);
            s.dYBytes = bytesY;
        }
        if (bytesQ8 > 0 && s.dQ8Bytes < bytesQ8) {
            MemorySegment previous = s.dQ8;
            s.dQ8 = null;
            s.dQ8Bytes = 0L;
            if (previous != null)
                cuda.deviceFree(previous);
            s.dQ8 = cuda.deviceMalloc(dev, bytesQ8);
            s.dQ8Bytes = bytesQ8;
        }
    }

    /**
     * Grows the FP16 staging buffers. Each grow clears its field before
     * freeing the old buffer, so an allocation that fails leaves that slot empty
     * rather than holding a pointer to freed memory: a caller that survives the
     * failure and retries would otherwise hand a dangling pointer to a later
     * memcpy, which reports an invalid argument far from the real cause.
     */
    private void ensureFp16Scratch(Fp16Scratch s, long bytesXh, long bytesY) {
        int dev = ctx.deviceIndex();
        if (s.dXhBytes < bytesXh) {
            MemorySegment previous = s.dXh;
            s.dXh = null;
            s.dXhBytes = 0L;
            if (previous != null)
                cuda.deviceFree(previous);
            s.dXh     = cuda.deviceMalloc(dev, bytesXh);
            s.dXhBytes = bytesXh;
        }
        if (s.dYBytes < bytesY) {
            MemorySegment previous = s.dY;
            s.dY = null;
            s.dYBytes = 0L;
            if (previous != null)
                cuda.deviceFree(previous);
            s.dY     = cuda.deviceMalloc(dev, bytesY);
            s.dYBytes = bytesY;
        }
        if (s.hXhBytes < bytesXh) {
            MemorySegment previous = s.hXh;
            s.hXh = null;
            s.hXhBytes = 0L;
            if (previous != null)
                cuda.hostFree(previous);
            s.hXh      = cuda.hostMalloc(dev, bytesXh);
            s.hXhBytes = bytesXh;
        }
        if (s.hYBytes < bytesY) {
            MemorySegment previous = s.hY;
            s.hY = null;
            s.hYBytes = 0L;
            if (previous != null)
                cuda.hostFree(previous);
            s.hY      = cuda.hostMalloc(dev, bytesY);
            s.hYBytes = bytesY;
        }
    }
}
