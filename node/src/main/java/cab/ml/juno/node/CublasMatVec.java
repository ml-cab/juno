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

import java.util.logging.Logger;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.cuda.global.cublas;
import org.bytedeco.cuda.global.cudart;

/**
 * GpuMatVec backed by cublasSgemv on an Nvidia GPU.
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
 *   - {@link #sgemv(DeviceFloatMatrix, float[])} — A stays on device; only x and y
 *     use per-call device buffers. Thread-safe for concurrent calls.
 *  
 * @author Yevhen Soldatov    
 */
public final class CublasMatVec implements GpuMatVec {

    @SuppressWarnings("unused")
    private static final Logger log = Logger.getLogger(CublasMatVec.class.getName());

    private static final int CUBLAS_OP_T = cublas.CUBLAS_OP_T;
    private static final int H2D = cudart.cudaMemcpyHostToDevice;
    private static final int D2H = cudart.cudaMemcpyDeviceToHost;

    private final GpuContext ctx;

    /**
     * @param ctx an open GpuContext — must outlive all sgemv calls on this instance
     */
    public CublasMatVec(GpuContext ctx) {
        if (ctx == null)
            throw new IllegalArgumentException("ctx must not be null");
        this.ctx = ctx;
    }

    @Override
    public float[] sgemv(float[] A, float[] x, int rows, int cols) {
        if (A.length != (long) rows * cols)
            throw new IllegalArgumentException("A.length=" + A.length + " != rows*cols=" + ((long) rows * cols));
        if (x.length != cols)
            throw new IllegalArgumentException("x.length=" + x.length + " != cols=" + cols);

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

        float[] y = new float[rows];
        long bytesX = (long) cols * 4;
        long bytesY = (long) rows * 4;

        PointerPointer pX = new PointerPointer(1);
        PointerPointer pY = new PointerPointer(1);
        try {
            checkCuda(cudart.cudaSetDevice(ctx.deviceIndex()), "cudaSetDevice");

            int rX = cudart.cudaMalloc(pX, bytesX);
            int rY = cudart.cudaMalloc(pY, bytesY);
            if (rX != 0 || rY != 0) {
                if (rX == 0) cudart.cudaFree(pX.get(0));
                if (rY == 0) cudart.cudaFree(pY.get(0));
                throw new IllegalStateException("cudaMalloc failed: d_x=" + rX + " d_y=" + rY);
            }

            Pointer d_x = pX.get(0);
            Pointer d_y = pY.get(0);
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
            cudart.cudaFree(pX.get(0));
            cudart.cudaFree(pY.get(0));
            pX.close();
            pY.close();
        }
    }

    private static void checkCuda(int rc, String op) {
        if (rc != 0) throw new IllegalStateException(op + " failed: " + rc);
    }
}