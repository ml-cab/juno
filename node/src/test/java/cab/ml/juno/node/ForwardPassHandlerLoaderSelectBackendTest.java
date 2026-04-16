package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;
import static org.junit.jupiter.api.Assumptions.assumeFalse;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Tests for {@link ForwardPassHandlerLoader#selectBackend()}.
 *
 * Verifies that the no-arg {@code load()} path picks the right compute backend
 * based on the {@code JUNO_USE_GPU} system property and CUDA availability,
 * rather than unconditionally hard-coding {@link CpuMatVec}.
 *
 * All tests restore the original value of {@code JUNO_USE_GPU} after running.
 * GPU-tagged tests only execute when {@link CudaAvailability#isAvailable()} is
 * true (i.e. on AWS / CUDA nodes).
 */
@DisplayName("ForwardPassHandlerLoader — selectBackend() GPU/CPU routing")
class ForwardPassHandlerLoaderSelectBackendTest {

    private String originalGpuFlag;

    @BeforeEach
    void saveFlag() {
        originalGpuFlag = System.getProperty("JUNO_USE_GPU");
    }

    @AfterEach
    void restoreFlag() {
        if (originalGpuFlag == null) {
            System.clearProperty("JUNO_USE_GPU");
        } else {
            System.setProperty("JUNO_USE_GPU", originalGpuFlag);
        }
    }

    // ── CPU-only (always run) ─────────────────────────────────────────────────

    @Test
    @DisplayName("JUNO_USE_GPU=false → CpuMatVec regardless of hardware")
    void gpu_flag_false_yields_cpu_backend() {
        System.setProperty("JUNO_USE_GPU", "false");

        MatVec backend = ForwardPassHandlerLoader.selectBackend();

        assertThat(backend).isInstanceOf(CpuMatVec.class);
    }

    @Test
    @DisplayName("JUNO_USE_GPU absent → CpuMatVec (safe default)")
    void gpu_flag_absent_yields_cpu_backend() {
        System.clearProperty("JUNO_USE_GPU");

        MatVec backend = ForwardPassHandlerLoader.selectBackend();

        assertThat(backend).isInstanceOf(CpuMatVec.class);
    }

    @Test
    @DisplayName("JUNO_USE_GPU=true on CPU-only machine → falls back to CpuMatVec")
    void gpu_flag_true_no_cuda_falls_back_to_cpu() {
        assumeFalse(CudaAvailability.isAvailable(), "Skipping — CUDA present on this machine");
        System.setProperty("JUNO_USE_GPU", "true");

        MatVec backend = ForwardPassHandlerLoader.selectBackend();

        assertThat(backend).isInstanceOf(CpuMatVec.class);
    }

    @Test
    @DisplayName("selectBackend() never returns null")
    void select_backend_never_null() {
        System.setProperty("JUNO_USE_GPU", "false");
        assertThat(ForwardPassHandlerLoader.selectBackend()).isNotNull();

        System.clearProperty("JUNO_USE_GPU");
        assertThat(ForwardPassHandlerLoader.selectBackend()).isNotNull();
    }

    // ── GPU-tagged (runs only on CUDA nodes) ──────────────────────────────────

    @Test
    @Tag("gpu")
    @DisplayName("JUNO_USE_GPU=true on CUDA node → CudaMatVec")
    void gpu_flag_true_with_cuda_yields_cuda_backend() {
        assumeTrue(CudaAvailability.isAvailable(), "No CUDA device — skipping");
        System.setProperty("JUNO_USE_GPU", "true");

        MatVec backend = ForwardPassHandlerLoader.selectBackend();

        assertThat(backend).isInstanceOf(CudaMatVec.class);
    }

    @Test
    @Tag("gpu")
    @DisplayName("selectBackend() reuses process-wide GpuContext.shared(0)")
    void select_backend_reuses_shared_gpu_context() {
        assumeTrue(CudaAvailability.isAvailable(), "No CUDA device — skipping");
        System.setProperty("JUNO_USE_GPU", "true");

        MatVec a = ForwardPassHandlerLoader.selectBackend();
        MatVec b = ForwardPassHandlerLoader.selectBackend();

        assertThat(a).isInstanceOf(CudaMatVec.class);
        assertThat(b).isInstanceOf(CudaMatVec.class);
        assertThat(((CudaMatVec) a).gpuContext()).isSameAs(((CudaMatVec) b).gpuContext());
        assertThat(((CudaMatVec) a).gpuContext().isProcessShared()).isTrue();
    }
}