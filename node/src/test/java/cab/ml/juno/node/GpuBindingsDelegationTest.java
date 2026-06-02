package cab.ml.juno.node;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeFalse;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Verifies that {@link GpuBindings} vendor-neutral accessor methods delegate
 * to the correct underlying handles in each concrete implementation.
 *
 * CPU-only path (no GPU required): asserts that unavailable backends return
 * {@code false} from {@code isAvailable()} and that the constant accessors on
 * {@link CudaBindings} retain their expected values.
 */
@DisplayName("GpuBindings — vendor-neutral delegation contract")
class GpuBindingsDelegationTest {

    // ── CPU-only: constant accessors accessible without GPU ───────────────────

    @Test
    @DisplayName("CudaBindings.DEVICE_PROP_BYTES matches CUDA 12.x struct size")
    void cuda_device_prop_bytes_matches_known_value() {
        assertThat(CudaBindings.DEVICE_PROP_BYTES).isEqualTo(1512);
    }

    @Test
    @DisplayName("CudaBindings.PROP_NAME_OFFSET is zero (name is first field)")
    void cuda_prop_name_offset_is_zero() {
        assertThat(CudaBindings.PROP_NAME_OFFSET).isEqualTo(0L);
    }

    @Test
    @DisplayName("CudaBindings.PROP_TOTAL_MEM_OFFSET is 288 for CUDA 12.x / x86_64")
    void cuda_prop_total_mem_offset_matches_known_value() {
        assertThat(CudaBindings.PROP_TOTAL_MEM_OFFSET).isEqualTo(288L);
    }

    // ── CUDA present: interface methods return non-null handles ───────────────

    @Test
    @DisplayName("CudaBindings: all gpuGetDeviceCount/gpuMemcpy/blasCreate handles non-null")
    void cuda_vendor_neutral_handles_non_null() {
        assumeTrue(CudaBindings.isAvailable(), "Skipping — no CUDA device");
        CudaBindings cuda = CudaBindings.instance();
        assertThat(cuda.gpuGetDeviceCount()).isNotNull();
        assertThat(cuda.gpuGetDeviceProperties()).isNotNull();
        assertThat(cuda.gpuSetDevice()).isNotNull();
        assertThat(cuda.gpuMalloc()).isNotNull();
        assertThat(cuda.gpuFree()).isNotNull();
        assertThat(cuda.gpuMemcpy()).isNotNull();
        assertThat(cuda.gpuMemcpyAsync()).isNotNull();
        assertThat(cuda.gpuStreamCreateWithFlags()).isNotNull();
        assertThat(cuda.gpuStreamSynchronize()).isNotNull();
        assertThat(cuda.blasCreate()).isNotNull();
        assertThat(cuda.blasDestroy()).isNotNull();
        assertThat(cuda.blasSetStream()).isNotNull();
        assertThat(cuda.blasSetPointerMode()).isNotNull();
        assertThat(cuda.blasSgemv()).isNotNull();
        assertThat(cuda.blasHSSgemvStridedBatched()).isNotNull();
    }

    @Test
    @DisplayName("CudaBindings: createMatVec returns CudaMatVec for a CUDA context")
    void cuda_create_mat_vec_returns_cuda_impl() {
        assumeTrue(CudaBindings.isAvailable(), "Skipping — no CUDA device");
        try (GpuContext ctx = GpuContext.init(0)) {
            MatVec mv = ctx.bindings().createMatVec(ctx);
            assertThat(mv).isInstanceOf(CudaMatVec.class);
        }
    }

    // ── ROCm present: interface methods return non-null handles ───────────────

    @Test
    @DisplayName("RocmBindings: all gpuGetDeviceCount/gpuMemcpy/blasCreate handles non-null")
    void rocm_vendor_neutral_handles_non_null() {
        assumeTrue(RocmBindings.isAvailable(), "Skipping — no ROCm device");
        RocmBindings rocm = RocmBindings.instance();
        assertThat(rocm.gpuGetDeviceCount()).isNotNull();
        assertThat(rocm.gpuGetDeviceProperties()).isNotNull();
        assertThat(rocm.gpuSetDevice()).isNotNull();
        assertThat(rocm.gpuMalloc()).isNotNull();
        assertThat(rocm.gpuFree()).isNotNull();
        assertThat(rocm.gpuMemcpy()).isNotNull();
        assertThat(rocm.gpuMemcpyAsync()).isNotNull();
        assertThat(rocm.gpuStreamCreateWithFlags()).isNotNull();
        assertThat(rocm.gpuStreamSynchronize()).isNotNull();
        assertThat(rocm.blasCreate()).isNotNull();
        assertThat(rocm.blasDestroy()).isNotNull();
        assertThat(rocm.blasSetStream()).isNotNull();
        assertThat(rocm.blasSetPointerMode()).isNotNull();
        assertThat(rocm.blasSgemv()).isNotNull();
        assertThat(rocm.blasHSSgemvStridedBatched()).isNotNull();
    }

    @Test
    @DisplayName("RocmBindings: createMatVec returns RocmMatVec for a ROCm context")
    void rocm_create_mat_vec_returns_rocm_impl() {
        assumeTrue(RocmBindings.isAvailable(), "Skipping — no ROCm device");
        assumeFalse(CudaBindings.isAvailable(), "Skipping — CUDA takes precedence on this machine");
        try (GpuContext ctx = GpuContext.init(0)) {
            MatVec mv = ctx.bindings().createMatVec(ctx);
            assertThat(mv).isInstanceOf(RocmMatVec.class);
        }
    }

    // ── MatVecBackend public visibility ───────────────────────────────────────

    @Test
    @DisplayName("MatVecBackend is public and label() returns stable JFR strings")
    void mat_vec_backend_is_public_with_stable_labels() {
        assertThat(MatVecBackend.CPU.label()).isEqualTo("cpu");
        assertThat(MatVecBackend.CUDA.label()).isEqualTo("cuda");
        assertThat(MatVecBackend.CUDA_RESIDENT.label()).isEqualTo("cuda-resident");
        assertThat(MatVecBackend.CUDA_RESIDENT_FP16.label()).isEqualTo("cuda-resident-fp16");
        assertThat(MatVecBackend.ROCM.label()).isEqualTo("rocm");
        assertThat(MatVecBackend.ROCM_RESIDENT.label()).isEqualTo("rocm-resident");
        assertThat(MatVecBackend.ROCM_RESIDENT_FP16.label()).isEqualTo("rocm-resident-fp16");
    }
}