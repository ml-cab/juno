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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.logging.Logger;

import org.bytedeco.cuda.cudart.__half;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.cuda.global.cublas;
import org.bytedeco.cuda.global.cudart;

/**
 * MatVecBackend backed by cublasSgemv on an Nvidia GPU.
 *
 * Uses org.bytedeco (JavaCPP) cuda/cublas. Works with various NVIDIA GPUs
 * (e.g. GTX 1080, T4, and newer).
 *
 * Computes y[rows] = A[rows, cols] x x[cols] where A is row-major.
 *
 * Row-major to cuBLAS mapping: cuBLAS is column-major. A row-major matrix
 * A[rows x cols] is identical in memory to the transpose of a column-major
 * matrix A^T[cols x rows]. To compute y = A * x using cuBLAS we call
 * cublasSgemv with CUBLAS_OP_T, m=cols, n=rows, lda=cols.
 *
 * Memory management:
 *   - {@link #sgemv(float[], float[], int, int)} — full host path: allocates
 *     device buffers for A, x, y each call (tests and legacy use).
 *   - {@link #sgemv(DeviceFloatMatrix, float[])} / {@link #sgemv(DeviceHalfMatrix, float[])}
 *     — {@code A} stays on device; per-thread scratch buffers for {@code x} and {@code y}
 *     are grown as needed and reused to avoid per-call cudaMalloc/cudaFree.
 *  
 * @author Yevhen Soldatov    
 */
public final class CudaMatVec implements MatVec {

    @SuppressWarnings("unused")
    private static final Logger log = Logger.getLogger(CudaMatVec.class.getName());

    private static final int CUBLAS_OP_T = cublas.CUBLAS_OP_T;
    private static final int H2D = cudart.cudaMemcpyHostToDevice;
    private static final int D2H = cudart.cudaMemcpyDeviceToHost;

    private final GpuContext ctx;

    /** Per-thread reusable device buffers for FP32 resident GEMV. */
    private static final ThreadLocal<Fp32ResidentScratch> FP32_RESIDENT = ThreadLocal.withInitial(Fp32ResidentScratch::new);

    /** Per-thread reusable buffers for FP16-weight GEMV (pinned host staging when possible). */
    private static final ThreadLocal<Fp16ResidentScratch> FP16_RESIDENT = ThreadLocal.withInitial(Fp16ResidentScratch::new);

    private static final class Fp32ResidentScratch {
        Pointer dX;
        Pointer dY;
        long bytesXAlloc;
        long bytesYAlloc;

        void ensure(int deviceIndex, long bytesX, long bytesY) {
            checkCuda(cudart.cudaSetDevice(deviceIndex), "cudaSetDevice");
            if (bytesXAlloc < bytesX) {
                if (dX != null)
                    cudart.cudaFree(dX);
                PointerPointer pp = new PointerPointer(1);
                try {
                    checkCuda(cudart.cudaMalloc(pp, bytesX), "cudaMalloc d_x");
                    dX = pp.get(0);
                    bytesXAlloc = bytesX;
                } finally {
                    pp.close();
                }
            }
            if (bytesYAlloc < bytesY) {
                if (dY != null)
                    cudart.cudaFree(dY);
                PointerPointer pp = new PointerPointer(1);
                try {
                    checkCuda(cudart.cudaMalloc(pp, bytesY), "cudaMalloc d_y");
                    dY = pp.get(0);
                    bytesYAlloc = bytesY;
                } finally {
                    pp.close();
                }
            }
        }
    }

    private static final class Fp16ResidentScratch {
        Pointer dXh;
        Pointer dY;
        long bytesXhAlloc;
        long bytesYAlloc;
        /** Reused pageable staging for FP16 x; avoids per-call allocation. */
        byte[] halfBytes;
        /** Pinned host buffer for xh when cudaMallocHost succeeds (faster H2D). */
        Pointer pinnedHalf;
        long pinnedBytesAlloc;

        void ensureDevice(int deviceIndex, long bytesXh, long bytesY) {
            checkCuda(cudart.cudaSetDevice(deviceIndex), "cudaSetDevice");
            if (bytesXhAlloc < bytesXh) {
                if (dXh != null)
                    cudart.cudaFree(dXh);
                PointerPointer pp = new PointerPointer(1);
                try {
                    checkCuda(cudart.cudaMalloc(pp, bytesXh), "cudaMalloc d_xh");
                    dXh = pp.get(0);
                    bytesXhAlloc = bytesXh;
                } finally {
                    pp.close();
                }
            }
            if (bytesYAlloc < bytesY) {
                if (dY != null)
                    cudart.cudaFree(dY);
                PointerPointer pp = new PointerPointer(1);
                try {
                    checkCuda(cudart.cudaMalloc(pp, bytesY), "cudaMalloc d_y");
                    dY = pp.get(0);
                    bytesYAlloc = bytesY;
                } finally {
                    pp.close();
                }
            }
        }

        void ensurePinnedHalf(int needBytes) {
            if (pinnedBytesAlloc >= needBytes && pinnedHalf != null)
                return;
            if (pinnedHalf != null) {
                cudart.cudaFreeHost(pinnedHalf);
                pinnedHalf = null;
                pinnedBytesAlloc = 0;
            }
            PointerPointer pp = new PointerPointer(1);
            try {
                if (cudart.cudaMallocHost(pp, needBytes) == 0) {
                    pinnedHalf = pp.get(0);
                    pinnedBytesAlloc = needBytes;
                }
            } finally {
                pp.close();
            }
        }

        /** Pack {@code x} as FP16 little-endian and H2D into {@code d_xh}. */
        void packXHalfAndUpload(float[] x, int cols, Pointer d_xh, long bytesXh) {
            int needBytes = cols * 2;
            ensurePinnedHalf(needBytes);
            if (pinnedHalf != null) {
                try (BytePointer bp = new BytePointer(pinnedHalf).limit(needBytes)) {
                    for (int j = 0; j < cols; j++)
                        bp.putShort(j * 2, Float.floatToFloat16(x[j]));
                }
                try (BytePointer hXh = new BytePointer(pinnedHalf)) {
                    checkCuda(cudart.cudaMemcpy(d_xh, hXh, bytesXh, H2D), "cudaMemcpy(xh H2D)");
                }
            } else {
                if (halfBytes == null || halfBytes.length < needBytes)
                    halfBytes = new byte[needBytes];
                ByteBuffer bb = ByteBuffer.wrap(halfBytes).order(ByteOrder.LITTLE_ENDIAN);
                for (int j = 0; j < cols; j++)
                    bb.putShort(j * 2, Float.floatToFloat16(x[j]));
                try (BytePointer hXh = new BytePointer(halfBytes)) {
                    checkCuda(cudart.cudaMemcpy(d_xh, hXh, bytesXh, H2D), "cudaMemcpy(xh H2D)");
                }
            }
        }
    }

    /**
     * @param ctx an open GpuContext — must outlive all sgemv calls on this instance
     */
    public CudaMatVec(GpuContext ctx) {
        if (ctx == null)
            throw new IllegalArgumentException("ctx must not be null");
        this.ctx = ctx;
    }

    /** The CUDA/cuBLAS context backing this backend (package scope for tests). */
    GpuContext gpuContext() {
        return ctx;
    }

    /**
     * Allocates device memory and uploads {@code host} once (H2D).
     *
     * <p>Convenience factory so {@link LlamaTransformerHandler} can upload
     * dequantized weight matrices without a direct reference to {@link GpuContext}.
     *
     * @param host row-major float weights, length {@code rows * cols}
     * @return a {@link DeviceFloatMatrix} — must be closed when the handler shuts down
     */
    DeviceFloatMatrix upload(float[] host, int rows, int cols) {
        return DeviceFloatMatrix.upload(ctx, host, rows, cols);
    }

    /**
     * Upload float32 host weights as FP16 on device (≈2× less VRAM than {@link #upload}).
     */
    DeviceHalfMatrix uploadHalf(float[] host, int rows, int cols) {
        return DeviceHalfMatrix.uploadFromFloat32(ctx, host, rows, cols);
    }

    @Override
    public float[] sgemv(float[] A, float[] x, int rows, int cols) {
        if (A.length != (long) rows * cols)
            throw new IllegalArgumentException("A.length=" + A.length + " != rows*cols=" + ((long) rows * cols));
        if (x.length != cols)
            throw new IllegalArgumentException("x.length=" + x.length + " != cols=" + cols);

        MatVecEvent evt = new MatVecEvent();
        evt.begin();

        float[] y = new float[rows];
        long bytesA = (long) rows * cols * 4;
        long bytesX = (long) cols * 4;
        long bytesY = (long) rows * 4;

        PointerPointer pA = new PointerPointer(1);
        PointerPointer pX = new PointerPointer(1);
        PointerPointer pY = new PointerPointer(1);
        try {
            // CUDA device selection is thread-local; bind on every call.
            int setDeviceRc = cudart.cudaSetDevice(ctx.deviceIndex());
            checkCuda(setDeviceRc, "cudaSetDevice");

            int rA = cudart.cudaMalloc(pA, bytesA);
            int rX = cudart.cudaMalloc(pX, bytesX);
            int rY = cudart.cudaMalloc(pY, bytesY);
            if (rA != 0 || rX != 0 || rY != 0) {
                if (rA == 0) cudart.cudaFree(pA.get(0));
                if (rX == 0) cudart.cudaFree(pX.get(0));
                if (rY == 0) cudart.cudaFree(pY.get(0));
                throw new IllegalStateException("cudaMalloc failed: d_A=" + rA + " d_x=" + rX + " d_y=" + rY);
            }

            Pointer d_A = pA.get(0);
            Pointer d_x = pX.get(0);
            Pointer d_y = pY.get(0);

            try (FloatPointer hA = new FloatPointer(A); FloatPointer hX = new FloatPointer(x)) {
                checkCuda(cudart.cudaMemcpy(d_A, hA, bytesA, H2D), "cudaMemcpy(A H2D)");
                checkCuda(cudart.cudaMemcpy(d_x, hX, bytesX, H2D), "cudaMemcpy(x H2D)");
            }

            try (
                FloatPointer alpha = new FloatPointer(1.0f);
                FloatPointer beta = new FloatPointer(0.0f);
                FloatPointer d_A_f = new FloatPointer(d_A);
                FloatPointer d_x_f = new FloatPointer(d_x);
                FloatPointer d_y_f = new FloatPointer(d_y);
            ) {
                int pointerModeRc = cublas.cublasSetPointerMode_v2(
                    ctx.handle(), cublas.CUBLAS_POINTER_MODE_HOST);
                if (pointerModeRc != 0)
                    throw new IllegalStateException("cublasSetPointerMode failed: " + pointerModeRc);

                int rc = cublas.cublasSgemv_v2(ctx.handle(), CUBLAS_OP_T,
                    cols, rows,
                    alpha, d_A_f, cols,
                    d_x_f, 1,
                    beta, d_y_f, 1);
                if (rc != 0)
                    throw new IllegalStateException("cublasSgemv failed: " + rc);
            }

            try (FloatPointer hy = new FloatPointer(y)) {
                checkCuda(cudart.cudaMemcpy(hy, d_y, bytesY, D2H), "cudaMemcpy(y D2H)");
                // FloatPointer(y) is native memory initialized from y[]; copy back to heap array.
                hy.get(y);
            }

            return y;
        } finally {
            cudart.cudaFree(pA.get(0));
            cudart.cudaFree(pX.get(0));
            cudart.cudaFree(pY.get(0));
            pA.close();
            pX.close();
            pY.close();
            evt.backend = "cuda";
            evt.rows = rows;
            evt.cols = cols;
            evt.commit();
        }
    }

    @Override
    public float[] sgemv(DeviceFloatMatrix A, float[] x) {
        if (A == null)
            throw new IllegalArgumentException("A must not be null");
        if (A.isClosed())
            throw new IllegalStateException("DeviceFloatMatrix is closed");
        int rows = A.rows();
        int cols = A.cols();
        if (x.length != cols)
            throw new IllegalArgumentException("x.length=" + x.length + " != cols=" + cols);

        MatVecEvent evt = new MatVecEvent();
        evt.begin();

        float[] y = new float[rows];
        long bytesX = (long) cols * 4;
        long bytesY = (long) rows * 4;

        Fp32ResidentScratch s = FP32_RESIDENT.get();
        try {
            s.ensure(ctx.deviceIndex(), bytesX, bytesY);
            Pointer d_x = s.dX;
            Pointer d_y = s.dY;
            Pointer d_A = A.devicePointer();

            try (FloatPointer hX = new FloatPointer(x)) {
                checkCuda(cudart.cudaMemcpy(d_x, hX, bytesX, H2D), "cudaMemcpy(x H2D)");
            }

            try (
                FloatPointer alpha = new FloatPointer(1.0f);
                FloatPointer beta = new FloatPointer(0.0f);
                FloatPointer d_A_f = new FloatPointer(d_A);
                FloatPointer d_x_f = new FloatPointer(d_x);
                FloatPointer d_y_f = new FloatPointer(d_y);
            ) {
                int pointerModeRc = cublas.cublasSetPointerMode_v2(
                    ctx.handle(), cublas.CUBLAS_POINTER_MODE_HOST);
                if (pointerModeRc != 0)
                    throw new IllegalStateException("cublasSetPointerMode failed: " + pointerModeRc);

                int rc = cublas.cublasSgemv_v2(ctx.handle(), CUBLAS_OP_T,
                    cols, rows,
                    alpha, d_A_f, cols,
                    d_x_f, 1,
                    beta, d_y_f, 1);
                if (rc != 0)
                    throw new IllegalStateException("cublasSgemv failed: " + rc);
            }

            try (FloatPointer hy = new FloatPointer(y)) {
                checkCuda(cudart.cudaMemcpy(hy, d_y, bytesY, D2H), "cudaMemcpy(y D2H)");
                hy.get(y);
            }

            return y;
        } finally {
            evt.backend = "cuda-resident";
            evt.rows = rows;
            evt.cols = cols;
            evt.commit();
        }
    }

    /**
     * {@code y = A × x} with row-major {@code A[rows, cols]} in FP16 on the device and
     * {@code x} in FP32 — converts {@code x} to FP16 for the multiply, accumulates in FP32
     * via {@code cublasHSSgemvStridedBatched} (same {@code trans, m, n, lda} contract as
     * {@link #sgemv(DeviceFloatMatrix, float[])}).
     */
    @Override
    public float[] sgemv(DeviceHalfMatrix A, float[] x) {
        if (A == null)
            throw new IllegalArgumentException("A must not be null");
        if (A.isClosed())
            throw new IllegalStateException("DeviceHalfMatrix is closed");
        int rows = A.rows();
        int cols = A.cols();
        if (x.length != cols)
            throw new IllegalArgumentException("x.length=" + x.length + " != cols=" + cols);

        MatVecEvent evt = new MatVecEvent();
        evt.begin();

        float[] y = new float[rows];
        long bytesXh = (long) cols * 2;
        long bytesY = (long) rows * 4;

        Fp16ResidentScratch s = FP16_RESIDENT.get();
        try {
            s.ensureDevice(ctx.deviceIndex(), bytesXh, bytesY);
            Pointer d_xh = s.dXh;
            Pointer d_y = s.dY;
            Pointer d_A = A.devicePointer();

            s.packXHalfAndUpload(x, cols, d_xh, bytesXh);

            try (
                    FloatPointer alpha = new FloatPointer(1.0f);
                    FloatPointer beta = new FloatPointer(0.0f);
                    FloatPointer d_y_f = new FloatPointer(d_y);
                    __half d_A_h = new __half(d_A);
                    __half d_x_h = new __half(d_xh);
            ) {
                int pointerModeRc = cublas.cublasSetPointerMode_v2(
                        ctx.handle(), cublas.CUBLAS_POINTER_MODE_HOST);
                if (pointerModeRc != 0)
                    throw new IllegalStateException("cublasSetPointerMode failed: " + pointerModeRc);

                // Same (trans, m, n, lda) as cublasSgemv_v2 in sgemv(DeviceFloatMatrix, …).
                long strideA = (long) cols * rows;
                long strideX = cols;
                long strideY = rows;
                int rc = cublas.cublasHSSgemvStridedBatched(ctx.handle(), CUBLAS_OP_T,
                        cols, rows,
                        alpha, d_A_h, cols, strideA,
                        d_x_h, 1, strideX,
                        beta, d_y_f, 1, strideY,
                        1);
                if (rc != 0)
                    throw new IllegalStateException("cublasHSSgemvStridedBatched failed: " + rc);
            }

            try (FloatPointer hy = new FloatPointer(y)) {
                checkCuda(cudart.cudaMemcpy(hy, d_y, bytesY, D2H), "cudaMemcpy(y D2H)");
                hy.get(y);
            }

            return y;
        } finally {
            evt.backend = "cuda-resident-fp16";
            evt.rows = rows;
            evt.cols = cols;
            evt.commit();
        }
    }

    private static void checkCuda(int rc, String op) {
        if (rc != 0) throw new IllegalStateException(op + " failed: " + rc);
    }
}
