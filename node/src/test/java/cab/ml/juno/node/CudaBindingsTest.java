package cab.ml.juno.node;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assumptions.assumeTrue;
import static org.junit.jupiter.api.Assumptions.assumeFalse;

/**
 * CudaBindings contract tests.
 *
 * Two scenarios:
 *   1. CUDA present  (@Tag("gpu"))  — verifies every MethodHandle is non-null
 *      and that the singleton loads cleanly.
 *   2. CUDA absent (CPU-only CI)    — verifies isAvailable() returns false and
 *      instance() throws a descriptive IllegalStateException.
 *
 * Run on GPU machines:
 *   mvn test -Dgroups=gpu -pl node
 * Run on CPU-only:
 *   mvn test -Dgroups='!gpu' -pl node
 */
@DisplayName("CudaBindings — Panama FFI library loading")
class CudaBindingsTest {

    // ── CUDA absent path (CPU-only CI) ────────────────────────────────────────

    @Test
    @DisplayName("isAvailable() returns false when CUDA libraries are absent")
    void isAvailable_false_on_cpu_only() {
        assumeFalse(CudaBindings.isAvailable(), "Skipping — CUDA is present");
        assertThat(CudaBindings.isAvailable()).isFalse();
    }

    @Test
    @DisplayName("instance() throws IllegalStateException when CUDA is absent")
    void instance_throws_when_unavailable() {
        assumeFalse(CudaBindings.isAvailable(), "Skipping — CUDA is present");
        assertThatThrownBy(CudaBindings::instance)
            .isInstanceOf(IllegalStateException.class)
            .hasMessageContaining("CUDA not available");
    }

    // ── CUDA present path ─────────────────────────────────────────────────────

    @Test
    @Tag("gpu")
    @DisplayName("isAvailable() returns true when CUDA libraries load")
    void isAvailable_true_on_gpu() {
        assumeTrue(CudaBindings.isAvailable(), "Skipping — no CUDA device");
        assertThat(CudaBindings.isAvailable()).isTrue();
    }

    @Test
    @Tag("gpu")
    @DisplayName("instance() is a singleton — same reference on repeated calls")
    void instance_is_singleton() {
        assumeTrue(CudaBindings.isAvailable(), "Skipping — no CUDA device");
        CudaBindings a = CudaBindings.instance();
        CudaBindings b = CudaBindings.instance();
        assertThat(a).isSameAs(b);
    }

    @Test
    @Tag("gpu")
    @DisplayName("All cudart MethodHandles are non-null")
    void cudart_handles_non_null() {
        assumeTrue(CudaBindings.isAvailable(), "Skipping — no CUDA device");
        CudaBindings cuda = CudaBindings.instance();
        assertThat(cuda.cudaGetDeviceCount).isNotNull();
        assertThat(cuda.cudaGetDeviceProperties).isNotNull();
        assertThat(cuda.cudaSetDevice).isNotNull();
        assertThat(cuda.cudaMalloc).isNotNull();
        assertThat(cuda.cudaFree).isNotNull();
        assertThat(cuda.cudaMallocHost).isNotNull();
        assertThat(cuda.cudaFreeHost).isNotNull();
        assertThat(cuda.cudaMemcpy).isNotNull();
        assertThat(cuda.cudaMemcpyAsync).isNotNull();
        assertThat(cuda.cudaStreamCreateWithFlags).isNotNull();
        assertThat(cuda.cudaStreamSynchronize).isNotNull();
        assertThat(cuda.cudaStreamDestroy).isNotNull();
    }

    @Test
    @Tag("gpu")
    @DisplayName("All cuBLAS MethodHandles are non-null")
    void cublas_handles_non_null() {
        assumeTrue(CudaBindings.isAvailable(), "Skipping — no CUDA device");
        CudaBindings cuda = CudaBindings.instance();
        assertThat(cuda.cublasCreate).isNotNull();
        assertThat(cuda.cublasDestroy).isNotNull();
        assertThat(cuda.cublasSetStream).isNotNull();
        assertThat(cuda.cublasSetPointerMode).isNotNull();
        assertThat(cuda.cublasSgemv).isNotNull();
        assertThat(cuda.cublasHSSgemvStridedBatched).isNotNull();
    }

    @Test
    @Tag("gpu")
    @DisplayName("deviceMalloc/deviceFree round-trip does not throw")
    void deviceMalloc_free_roundtrip() {
        assumeTrue(CudaBindings.isAvailable(), "Skipping — no CUDA device");
        CudaBindings cuda = CudaBindings.instance();
        java.lang.foreign.MemorySegment seg = cuda.deviceMalloc(0, 1024);
        assertThat(seg).isNotNull();
        assertThat(seg.byteSize()).isEqualTo(1024);
        cuda.deviceFree(seg); // must not throw
    }

    @Test
    @Tag("gpu")
    @DisplayName("DEVICE_PROP_BYTES constant is positive")
    void device_prop_size_is_positive() {
        assertThat(CudaBindings.DEVICE_PROP_BYTES).isGreaterThan(0);
    }
}