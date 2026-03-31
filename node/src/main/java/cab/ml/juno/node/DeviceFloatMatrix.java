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

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.cuda.global.cudart;

/**
 * Row-major weight matrix resident on the GPU (single contiguous allocation).
 *
 * Created via {@link #upload}; released with {@link #close}. Used with
 * {@link CudaMatVec#sgemv(DeviceFloatMatrix, float[])} so each matmul avoids
 * re-uploading {@code A}.
 *   
 * @author Yevhen Soldatov
 * 
 */
public final class DeviceFloatMatrix implements AutoCloseable {

    private final GpuContext ctx;
    private final Pointer dA;
    private final int rows;
    private final int cols;
    private volatile boolean closed;

    private DeviceFloatMatrix(GpuContext ctx, Pointer dA, int rows, int cols) {
        this.ctx = ctx;
        this.dA = dA;
        this.rows = rows;
        this.cols = cols;
    }

    /**
     * Allocates device memory and copies {@code host} once (H2D).
     *
     * @param host row-major {@code A}, length {@code rows * cols}
     */
    public static DeviceFloatMatrix upload(GpuContext ctx, float[] host, int rows, int cols) {
        if (ctx == null)
            throw new IllegalArgumentException("ctx must not be null");
        if (host.length != (long) rows * cols)
            throw new IllegalArgumentException(
                "host.length=" + host.length + " != rows*cols=" + ((long) rows * cols));
        long bytes = (long) rows * cols * 4;
        PointerPointer<Pointer> pp = new PointerPointer<>(1);
        try {
            checkCuda(cudart.cudaSetDevice(ctx.deviceIndex()), "cudaSetDevice");
            int rc = cudart.cudaMalloc(pp, bytes);
            if (rc != 0)
                throw new IllegalStateException("cudaMalloc failed: " + rc);
            Pointer d = pp.get(0);
            try (FloatPointer h = new FloatPointer(host)) {
                checkCuda(
                    cudart.cudaMemcpy(d, h, bytes, cudart.cudaMemcpyHostToDevice),
                    "cudaMemcpy(A H2D)");
            }
            return new DeviceFloatMatrix(ctx, d, rows, cols);
        } finally {
            pp.close();
        }
    }

    public int rows() {
        return rows;
    }

    public int cols() {
        return cols;
    }

    /** Device pointer; valid until {@link #close()}. */
    Pointer devicePointer() {
        if (closed)
            throw new IllegalStateException("DeviceFloatMatrix already closed");
        return dA;
    }

    public boolean isClosed() {
        return closed;
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            if (dA != null) {
                cudart.cudaSetDevice(ctx.deviceIndex());
                cudart.cudaFree(dA);
            }
        }
    }

    private static void checkCuda(int rc, String op) {
        if (rc != 0)
            throw new IllegalStateException(op + " failed: " + rc);
    }
}