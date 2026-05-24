package cab.ml.juno.node;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assumptions.assumeFalse;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * RocmBindings contract tests.
 *
 * Two scenarios:
 *   1. ROCm present  (@Tag("rocm")) — verifies every MethodHandle is non-null
 *      and that the singleton loads cleanly.
 *   2. ROCm absent (CPU-only / CUDA-only CI) — verifies isAvailable() returns
 *      false and instance() throws a descriptive IllegalStateException.
 *
 * Run on ROCm machines:
 *   mvn test -Dgroups=rocm -pl node
 * Run on CPU-only / CUDA-only:
 *   mvn test -Dgroups='!rocm' -pl node
 */
@DisplayName("RocmBindings — Panama FFI library loading (AMD ROCm)")
class RocmBindingsTest {

    // ── ROCm absent path (CPU-only / CUDA-only CI) ────────────────────────────

    @Test
    @DisplayName("isAvailable() returns false when ROCm libraries are absent")
    void isAvailable_false_when_no_rocm() {
        assumeFalse(RocmBindings.isAvailable(), "Skipping — ROCm is present");
        assertThat(RocmBindings.isAvailable()).isFalse();
    }

    @Test
    @DisplayName("instance() throws IllegalStateException when ROCm is absent")
    void instance_throws_when_unavailable() {
        assumeFalse(RocmBindings.isAvailable(), "Skipping — ROCm is present");
        assertThatThrownBy(RocmBindings::instance)
            .isInstanceOf(IllegalStateException.class)
            .hasMessageContaining("ROCm not available");
    }

    // ── ROCm present path (@Tag("rocm")) ──────────────────────────────────────

    @Test
    @Tag("rocm")
    @DisplayName("isAvailable() returns true when ROCm libraries load")
    void isAvailable_true_on_rocm() {
        assumeTrue(RocmBindings.isAvailable(), "Skipping — no ROCm device");
        assertThat(RocmBindings.isAvailable()).isTrue();
    }

    @Test
    @Tag("rocm")
    @DisplayName("instance() is a singleton — same reference on repeated calls")
    void instance_is_singleton() {
        assumeTrue(RocmBindings.isAvailable(), "Skipping — no ROCm device");
        RocmBindings a = RocmBindings.instance();
        RocmBindings b = RocmBindings.instance();
        assertThat(a).isSameAs(b);
    }

    @Test
    @Tag("rocm")
    @DisplayName("All HIP runtime MethodHandles are non-null")
    void hip_runtime_handles_non_null() {
        assumeTrue(RocmBindings.isAvailable(), "Skipping — no ROCm device");
        RocmBindings rocm = RocmBindings.instance();
        assertThat(rocm.cudaGetDeviceCount()).isNotNull();
        assertThat(rocm.cudaGetDeviceProperties()).isNotNull();
        assertThat(rocm.cudaSetDevice()).isNotNull();
        assertThat(rocm.cudaMalloc()).isNotNull();
        assertThat(rocm.cudaFree()).isNotNull();
        assertThat(rocm.cudaMallocHost()).isNotNull();
        assertThat(rocm.cudaFreeHost()).isNotNull();
        assertThat(rocm.cudaMemcpy()).isNotNull();
        assertThat(rocm.cudaMemcpyAsync()).isNotNull();
        assertThat(rocm.cudaStreamCreateWithFlags()).isNotNull();
        assertThat(rocm.cudaStreamSynchronize()).isNotNull();
        assertThat(rocm.cudaStreamDestroy()).isNotNull();
    }

    @Test
    @Tag("rocm")
    @DisplayName("All rocBLAS MethodHandles are non-null")
    void rocblas_handles_non_null() {
        assumeTrue(RocmBindings.isAvailable(), "Skipping — no ROCm device");
        RocmBindings rocm = RocmBindings.instance();
        assertThat(rocm.cublasCreate()).isNotNull();
        assertThat(rocm.cublasDestroy()).isNotNull();
        assertThat(rocm.cublasSetStream()).isNotNull();
        assertThat(rocm.cublasSetPointerMode()).isNotNull();
        assertThat(rocm.cublasSgemv()).isNotNull();
        assertThat(rocm.cublasHSSgemvStridedBatched()).isNotNull();
    }

    @Test
    @Tag("rocm")
    @DisplayName("OP_TRANSPOSE = 112 (rocblas_operation_transpose)")
    void op_transpose_is_rocblas_value() {
        assumeTrue(RocmBindings.isAvailable(), "Skipping — no ROCm device");
        assertThat(RocmBindings.instance().opTranspose()).isEqualTo(112);
    }

    @Test
    @Tag("rocm")
    @DisplayName("POINTER_MODE_HOST = 0 (rocblas_pointer_mode_host)")
    void pointer_mode_host_is_zero() {
        assumeTrue(RocmBindings.isAvailable(), "Skipping — no ROCm device");
        assertThat(RocmBindings.instance().pointerModeHost()).isEqualTo(0);
    }

    @Test
    @Tag("rocm")
    @DisplayName("hipDeviceProp_t struct size matches ROCm 7.x headers (1472 bytes)")
    void hip_device_prop_size_correct() {
        assumeTrue(RocmBindings.isAvailable(), "Skipping — no ROCm device");
        assertThat(RocmBindings.instance().devicePropBytes()).isEqualTo(1472);
    }

    @Test
    @Tag("rocm")
    @DisplayName("deviceMalloc/deviceFree round-trip does not throw")
    void device_malloc_free_roundtrip() {
        assumeTrue(RocmBindings.isAvailable(), "Skipping — no ROCm device");
        RocmBindings rocm = RocmBindings.instance();
        java.lang.foreign.MemorySegment seg = rocm.deviceMalloc(0, 1024);
        assertThat(seg).isNotNull();
        assertThat(seg.byteSize()).isEqualTo(1024);
        rocm.deviceFree(seg);
    }

    @Test
    @Tag("rocm")
    @DisplayName("GpuContext.selectBindings() returns RocmBindings on ROCm-only system")
    void gpu_context_selects_rocm_on_amd() {
        assumeTrue(RocmBindings.isAvailable(), "Skipping — no ROCm device");
        assumeFalse(CudaBindings.isAvailable(), "Skipping — CUDA also present, CUDA wins in auto mode");
        GpuBindings bindings = GpuContext.selectBindings();
        assertThat(bindings).isInstanceOf(RocmBindings.class);
        assertThat(bindings.backendLabel()).isEqualTo("rocm");
    }
}
