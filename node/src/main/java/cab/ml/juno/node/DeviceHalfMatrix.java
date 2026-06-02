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

import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * Row-major FP16 (IEEE binary16) weight matrix on the GPU — half the VRAM of
 * {@link DeviceFloatMatrix} for the same logical shape.
 *
 * Float32 host weights are converted to FP16 in an off-heap confined arena
 * before upload, avoiding any GC-visible byte array allocation for the staging
 * buffer. The H2D transfer uses synchronous {@code cudaMemcpy}.
 *
 * Used with {@link MatVec#sgemv(DeviceHalfMatrix, float[])} (mixed
 * FP16/FP32 BLAS path) so activations stay float32 while weights stay compact.
 *
 * Vendor-neutral: device operations go through the {@link GpuBindings} obtained
 * from {@link GpuContext#bindings()} (captured at construction), so the same
 * class serves both CUDA and ROCm backends.
 */
public final class DeviceHalfMatrix implements AutoCloseable {

    private final GpuContext    ctx;
    private final GpuBindings   gpu;
    /** Device pointer, sized to rows * cols * Short.BYTES. */
    private final MemorySegment dA;
    private final int           rows;
    private final int           cols;
    private volatile boolean    closed;

    private DeviceHalfMatrix(GpuContext ctx, MemorySegment dA, int rows, int cols) {
        this.ctx  = ctx;
        this.gpu  = ctx.bindings();
        this.dA   = dA;
        this.rows = rows;
        this.cols = cols;
    }

    /**
     * Converts float32 host weights to FP16 (IEEE 754 binary16, little-endian)
     * and uploads {@code rows × cols} to the device.
     *
     * Conversion and staging use a confined off-heap arena — no heap byte[]
     * allocation. The staging segment is released immediately after transfer.
     */
    public static DeviceHalfMatrix uploadFromFloat32(GpuContext ctx, float[] host, int rows, int cols) {
        if (ctx == null)
            throw new IllegalArgumentException("ctx must not be null");
        if (host.length != (long) rows * cols)
            throw new IllegalArgumentException(
                "host.length=" + host.length + " != rows*cols=" + ((long) rows * cols));

        GpuBindings   gpu       = ctx.bindings();
        int           n         = rows * cols;
        long          halfBytes = (long) n * Short.BYTES;
        MemorySegment dA        = gpu.deviceMalloc(ctx.deviceIndex(), halfBytes);

        // Pack float32 → FP16 into off-heap staging; JAVA_SHORT has native byte order
        // (little-endian on x86), matching CUDA's __half / HIP's __half memory layout.
        try (Arena staging = Arena.ofConfined()) {
            MemorySegment stagingHost = staging.allocate(halfBytes);
            for (int i = 0; i < n; i++)
                stagingHost.setAtIndex(JAVA_SHORT, i, Float.floatToFloat16(host[i]));

            GpuBindings.check(
                GpuBindings.callInt(gpu.gpuMemcpy(), dA, stagingHost, halfBytes, GpuBindings.H2D),
                "memcpy(A FP16 H2D)");
        }

        return new DeviceHalfMatrix(ctx, dA, rows, cols);
    }

    public int rows() { return rows; }
    public int cols() { return cols; }

    /**
     * Device-side {@link MemorySegment} for the FP16 weight matrix.
     * Sized to {@code rows * cols * Short.BYTES}; valid until {@link #close()}.
     */
    MemorySegment devicePointer() {
        if (closed) throw new IllegalStateException("DeviceHalfMatrix already closed");
        return dA;
    }

    public boolean isClosed() { return closed; }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            GpuBindings.callInt(gpu.gpuSetDevice(), ctx.deviceIndex());
            gpu.deviceFree(dA);
        }
    }
}