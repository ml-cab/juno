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
 * Compute backend reported on each {@link MatVecEvent}. The single source of
 * truth for the {@code juno.MatVec.backend} JFR dimension — previously every
 * {@code MatVec} implementation assigned the label as an ad-hoc string literal
 * ({@code "cuda-resident-fp16"} and friends), which was easy to mistype and
 * impossible to enumerate.
 *
 * <p>The {@link #label()} strings are part of the JFR contract consumed by
 * tooling (JDK Mission Control, {@code juno.MatVec.backend.*.count}); keep them
 * stable.
 *
 * <p>The GPU vendor variants follow the pattern {@code <vendor>[-resident][-fp16]}:
 * <ul>
 *   <li>{@code <vendor>} — full host path (A and x copied per call)</li>
 *   <li>{@code <vendor>-resident} — A held resident on the device (FP32)</li>
 *   <li>{@code <vendor>-resident-fp16} — A held resident as FP16</li>
 * </ul>
 */
enum MatVecBackend {

    /** Pure-Java parallel CPU path (also used by all quantized GGUF matVecs). */
    CPU("cpu"),

    /** cuBLAS {@code cublasSgemv_v2}, A and x copied host→device per call. */
    CUDA("cuda"),
    /** cuBLAS {@code cublasSgemv_v2} with device-resident FP32 A. */
    CUDA_RESIDENT("cuda-resident"),
    /** cuBLAS {@code cublasHSSgemvStridedBatched} with device-resident FP16 A. */
    CUDA_RESIDENT_FP16("cuda-resident-fp16"),

    /** rocBLAS {@code rocblas_sgemv}, A and x copied host→device per call. */
    ROCM("rocm"),
    /** rocBLAS {@code rocblas_sgemv} with device-resident FP32 A. */
    ROCM_RESIDENT("rocm-resident"),
    /** rocBLAS {@code rocblas_hssgemv_strided_batched} with device-resident FP16 A. */
    ROCM_RESIDENT_FP16("rocm-resident-fp16");

    private final String label;

    MatVecBackend(String label) {
        this.label = label;
    }

    /** Stable JFR label, e.g. {@code "rocm-resident-fp16"}. */
    String label() {
        return label;
    }
}
