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

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

/**
 * Row-major FP32 weight matrix resident on the GPU.
 *
 * The device memory is allocated once via {@link GpuBindings#deviceMalloc} and
 * freed by {@link #close}. The backing {@link MemorySegment} is sized to
 * {@code rows * cols * 4} bytes so bounds checks work at the Java level.
 *
 * Works with both CUDA and ROCm backends via {@link GpuBindings}.
 * Used with {@link CudaMatVec#sgemv(DeviceFloatMatrix, float[])} so each
 * matmul skips re-uploading A.
 *
 * @author Yevhen Soldatov
 */
public final class DeviceFloatMatrix implements AutoCloseable {

    private final GpuContext    ctx;
    /** Device pointer, sized to rows * cols * Float.BYTES. */
    private final MemorySegment dA;
    private final int           rows;
    private final int           cols;
    private volatile boolean    closed;

    private DeviceFloatMatrix(GpuContext ctx, MemorySegment dA, int rows, int cols) {
        this.ctx  = ctx;
        this.dA   = dA;
        this.rows = rows;
        this.cols = cols;
    }

    /**
     * Allocates device memory and copies host float32 weights once (H2D).
     *
     * The H2D transfer uses synchronous {@code cudaMemcpy}; the method returns
     * only after the data is on the device.
     *
     * @param host row-major A, length {@code rows * cols}
     */
    public static DeviceFloatMatrix upload(GpuContext ctx, float[] host, int rows, int cols) {
        if (ctx == null)
            throw new IllegalArgumentException("ctx must not be null");
        if (host.length != (long) rows * cols)
            throw new IllegalArgumentException(
                "host.length=" + host.length + " != rows*cols=" + ((long) rows * cols));

        GpuBindings    gpu   = ctx.bindings();
        long           bytes = (long) rows * cols * Float.BYTES;
        MemorySegment  dA    = gpu.deviceMalloc(ctx.deviceIndex(), bytes);

        // MemorySegment.ofArray pins the heap array for the duration of the downcall.
        GpuBindings.check(
            GpuBindings.callInt(gpu.cudaMemcpy(), dA, MemorySegment.ofArray(host), bytes, GpuBindings.H2D),
            "cudaMemcpy(A H2D)");

        return new DeviceFloatMatrix(ctx, dA, rows, cols);
    }

    public int rows() { return rows; }
    public int cols() { return cols; }

    /**
     * Device-side {@link MemorySegment} for the weight matrix.
     * Sized to {@code rows * cols * Float.BYTES}; valid until {@link #close()}.
     */
    MemorySegment devicePointer() {
        if (closed) throw new IllegalStateException("DeviceFloatMatrix already closed");
        return dA;
    }

    public boolean isClosed() { return closed; }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            GpuBindings gpu = ctx.bindings();
            GpuBindings.callInt(gpu.cudaSetDevice(), ctx.deviceIndex());
            gpu.deviceFree(dA);
        }
    }
}
