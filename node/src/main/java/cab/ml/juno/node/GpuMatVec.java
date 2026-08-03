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

/**
 * A {@link MatVec} backed by a GPU that can hold weight matrices resident in
 * device memory.
 *
 * <p>Both {@link CudaMatVec} (NVIDIA) and {@link RocmMatVec} (AMD) implement
 * this interface. Transformer handlers ({@link LlamaTransformerHandler},
 * {@link Phi3TransformerHandler}, {@link LoraTrainableHandler}) depend on this
 * abstraction — not on a concrete vendor backend — so the device-resident
 * upload path is taken on <em>any</em> GPU, not just CUDA.
 *
 * <h3>Design — Dependency Inversion / Open–Closed</h3>
 * Previously the handlers gated GPU weight upload on {@code instanceof
 * CudaMatVec}, which silently routed AMD nodes onto the CPU path. Depending on
 * {@code GpuMatVec} instead keeps the handlers closed for modification while
 * remaining open to additional GPU backends.
 *
 * <p>The interface is {@code sealed}: only the in-module GPU backends may
 * implement it. CPU-only backends ({@link CpuMatVec}) intentionally implement
 * {@link MatVec} directly and do not expose device-resident uploads.
 */
sealed interface GpuMatVec extends MatVec permits CudaMatVec, RocmMatVec {

    /** Open {@link GpuContext} that owns this backend's BLAS handle. */
    GpuContext gpuContext();

    /**
     * Uploads a row-major FP32 weight matrix to device memory, held resident
     * across subsequent {@link #sgemv(DeviceFloatMatrix, float[])} calls.
     *
     * @param host row-major A, length {@code rows * cols}
     */
    DeviceFloatMatrix upload(float[] host, int rows, int cols);

    /**
     * Uploads a row-major weight matrix as FP16 (IEEE binary16) to device
     * memory, held resident across subsequent
     * {@link #sgemv(DeviceHalfMatrix, float[])} calls. Activations stay FP32;
     * the BLAS kernel accumulates in FP32.
     *
     * @param host row-major A (FP32 source), length {@code rows * cols}
     */
    DeviceHalfMatrix uploadHalf(float[] host, int rows, int cols);

    /**
     * Returns true if this backend supports FP16 device-resident weight matrices
     * via {@link #sgemv(DeviceHalfMatrix, float[])}.
     *
     * CUDA (cuBLAS): always true — cublasHSSgemvStridedBatched ships kernels for
     * all supported NVIDIA GPUs.
     * ROCm: false on gfx1010/gfx1011 (Navi12, g4ad) where
     * rocblas_hssgemv_strided_batched has no GPU kernel code object in ROCm 7.x.
     * When false, callers must use {@link #upload} + {@link #sgemv(DeviceFloatMatrix, float[])}
     * (FP32 resident, twice the VRAM but functionally correct).
     */
    default boolean supportsHalfResident() { return true; }

    /**
     * Resident frozen backward: {@code z = W^T * g} for row-major
     * {@code W[rows,cols]} already on device.
     *
     * <p>{@code g} has length {@code rows} (gradient w.r.t. forward output);
     * result has length {@code cols} (gradient w.r.t. forward input). Uses the
     * same device buffer as {@link #sgemv(DeviceFloatMatrix, float[])} with BLAS
     * no-transpose mode (see {@link GpuBindings#opNoTranspose()}).
     */
    float[] sgemvTranspose(DeviceFloatMatrix W, float[] g);

    /**
     * FP16-resident variant of {@link #sgemvTranspose(DeviceFloatMatrix, float[])}.
     * Activations stay FP32 on the host; the BLAS kernel accumulates in FP32.
     */
    float[] sgemvTranspose(DeviceHalfMatrix W, float[] g);
}