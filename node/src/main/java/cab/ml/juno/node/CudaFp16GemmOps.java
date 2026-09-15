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

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

/**
 * Tiled FP16 GEMM via {@code cublasGemmEx} (CUDA_R_16F A/B, CUDA_R_32F accumulate/C) —
 * a real weight-stationary matrix-matrix kernel, unlike {@code cublasHSSgemvStridedBatched}
 * (a strided-batched GEMV: {@code batch} independent per-column reductions with no
 * compute/bandwidth reuse across the batch dimension). Used for large (prefill-sized)
 * batches where re-streaming the weight matrix once instead of {@code batch} times
 * matters; small (decode) batches stay on {@link CudaMatVec#sgemm(DeviceHalfMatrix, float[][])}'s
 * existing strided-batched path.
 *
 * <p>CUDA-only: does not go through the vendor-neutral {@link GpuBindings} /
 * {@link GpuBlasOps} — {@link CudaMatVec}'s existing FP16 decode path already calls
 * {@link CudaBindings#cublasHSSgemvStridedBatched} directly for the same reason, and
 * {@link GpuBlasOps} is documented FP32-only/vendor-neutral.
 *
 * <p>Scratch buffers (device X staging, device Y) are owned by the caller
 * ({@link CudaMatVec}'s existing {@code Fp16Scratch}) — this class only issues the
 * {@code cublasGemmEx} call against already-populated device pointers.
 */
final class CudaFp16GemmOps {

    private final GpuContext   ctx;
    private final CudaBindings cuda;

    CudaFp16GemmOps(GpuContext ctx) {
        if (ctx == null) throw new IllegalArgumentException("ctx must not be null");
        this.ctx  = ctx;
        this.cuda = CudaBindings.instance();
    }

    /**
     * {@code Y = A^T X} for row-major {@code A[rows x cols]} reinterpreted column-major
     * with {@code lda = cols} (same convention as {@link CudaMatVec}'s existing GEMV/GEMM
     * calls). {@code dXh}: FP16 device buffer, column-major {@code [cols x batch]}
     * (element {@code (j,b)} at offset {@code b*cols+j} — the same layout
     * {@code sgemmHalfBatched} already stages). {@code dY}: FP32 device buffer,
     * column-major {@code [rows x batch]}. Must be called with the caller's CUDA
     * stream already bound via {@code cublasSetStream_v2} and under the shared
     * {@link GpuContext#cublasSerializationLock()}.
     *
     * @param dA    device pointer to FP16 row-major {@code A[rows x cols]}
     * @param dXh   device pointer to FP16 column-major {@code [cols x batch]} input
     * @param dY    device pointer to FP32 column-major {@code [rows x batch]} output
     */
    void gemmHalf(MemorySegment dA, MemorySegment dXh, MemorySegment dY,
                  int rows, int cols, int batch) {
        try (Arena scalars = Arena.ofConfined()) {
            MemorySegment alpha = scalars.allocateFrom(JAVA_FLOAT, 1.0f);
            MemorySegment beta  = scalars.allocateFrom(JAVA_FLOAT, 0.0f);
            CudaBindings.check(
                CudaBindings.callInt(cuda.cublasSetPointerMode, ctx.handle(), CudaBindings.CUBLAS_POINTER_MODE_HOST),
                "cublasSetPointerMode");
            CudaBindings.check(
                CudaBindings.callInt(cuda.cublasGemmEx,
                    ctx.handle(), CudaBindings.CUBLAS_OP_T, CudaBindings.CUBLAS_OP_N,
                    rows, batch, cols,
                    alpha,
                    dA, CudaBindings.CUDA_R_16F, cols,
                    dXh, CudaBindings.CUDA_R_16F, cols,
                    beta,
                    dY, CudaBindings.CUDA_R_32F, rows,
                    CudaBindings.CUBLAS_COMPUTE_32F, CudaBindings.CUBLAS_GEMM_DEFAULT),
                "cublasGemmEx");
        }
    }
}
