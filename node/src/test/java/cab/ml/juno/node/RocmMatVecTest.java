package cab.ml.juno.node;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.Assumptions.assumeTrue;
import static org.junit.jupiter.api.Assumptions.assumeFalse;

/**
 * RocmMatVec correctness and contract tests — requires an AMD GPU with ROCm 6+.
 *
 * <p>Inherits the full {@link MatVecBackendContractTest} suite so every contract
 * that holds for {@link CpuMatVec} must also hold for {@link RocmMatVec}.
 * Additionally tests ROCm-specific behaviour: correctness vs the CPU reference
 * implementation, concurrent safety, and the unsupported device-resident paths.
 *
 * <p>Run on ROCm machines:
 * <pre>
 *   mvn test -Dgroups=rocm -pl node
 * </pre>
 * Skip on CPU-only / CUDA-only:
 * <pre>
 *   mvn test -Dgroups='!rocm' -pl node
 * </pre>
 */
@Tag("rocm")
@DisplayName("RocmMatVec — rocblas_sgemv correctness (requires AMD ROCm)")
class RocmMatVecTest extends MatVecBackendContractTest {

    private static GpuContext ctx;
    private static RocmMatVec rocmImpl;

    @BeforeAll
    static void initRocm() {
        assumeTrue(RocmAvailability.isAvailable(),
            "Skipping RocmMatVecTest — no ROCm device available");
        ctx = GpuContext.init(0);
        rocmImpl = new RocmMatVec(ctx);
    }

    @AfterAll
    static void destroyRocm() {
        if (ctx != null) ctx.close();
    }

    /** Provide RocmMatVec to the inherited contract tests. */
    @Override
    protected MatVec impl() {
        return rocmImpl;
    }

    // ── ROCm-specific correctness tests ───────────────────────────────────────

    @Test
    @DisplayName("rocblas_sgemv matches LlamaTransformerHandler.matVec — 2048×2048 hidden dim")
    void rocblas_matches_cpu_reference_hidden_dim() {
        int rows = 2048, cols = 2048;
        float[] A = randomMatrix(rows, cols, 100);
        float[] x = randomVector(cols, 101);

        float[] cpuResult  = LlamaTransformerHandler.matVec(A, x, rows, cols);
        float[] rocmResult = rocmImpl.sgemv(A, x, rows, cols);

        assertThat(rocmResult).hasSize(rows);
        for (int i = 0; i < rows; i++)
            assertThat(rocmResult[i])
                .as("y[%d]", i)
                .isCloseTo(cpuResult[i], within(DELTA));
    }

    @Test
    @DisplayName("rocblas_sgemv matches CPU reference — 5632×2048 FFN gate (TinyLlama)")
    void rocblas_matches_cpu_reference_ffn_gate() {
        int rows = 5632, cols = 2048;
        float[] A = randomMatrix(rows, cols, 110);
        float[] x = randomVector(cols, 111);

        float[] cpuResult  = LlamaTransformerHandler.matVec(A, x, rows, cols);
        float[] rocmResult = rocmImpl.sgemv(A, x, rows, cols);

        assertThat(rocmResult).hasSize(rows);
        for (int i = 0; i < rows; i++)
            assertThat(rocmResult[i])
                .as("y[%d]", i)
                .isCloseTo(cpuResult[i], within(DELTA));
    }

    @Test
    @DisplayName("rocblas_sgemv matches CPU reference — 32000×2048 output projection")
    void rocblas_matches_cpu_reference_output_projection() {
        int rows = 32000, cols = 2048;
        float[] A = randomMatrix(rows, cols, 102);
        float[] x = randomVector(cols, 103);

        float[] cpuResult  = LlamaTransformerHandler.matVec(A, x, rows, cols);
        float[] rocmResult = rocmImpl.sgemv(A, x, rows, cols);

        for (int i = 0; i < rows; i++)
            assertThat(rocmResult[i])
                .as("y[%d]", i)
                .isCloseTo(cpuResult[i], within(DELTA));
    }

    @Test
    @DisplayName("1×1 trivial: A=[2.0], x=[3.0] → y=[6.0]")
    void trivial_1x1_known_value() {
        float[] y = rocmImpl.sgemv(new float[]{2.0f}, new float[]{3.0f}, 1, 1);
        assertThat(y).hasSize(1);
        assertThat(y[0]).isCloseTo(6.0f, within(1e-5f));
    }

    @Test
    @DisplayName("2×3 known result: y = [1+2+3, 4+5+6] = [6, 15]")
    void known_2x3_result() {
        float[] A = {1, 2, 3,  4, 5, 6};
        float[] x = {1, 1, 1};
        float[] y = rocmImpl.sgemv(A, x, 2, 3);
        assertThat(y).hasSize(2);
        assertThat(y[0]).isCloseTo(6.0f, within(1e-5f));
        assertThat(y[1]).isCloseTo(15.0f, within(1e-5f));
    }

    // ── Device-resident paths (A held on the GPU across calls) ─────────────────

    @Test
    @DisplayName("sgemv(DeviceFloatMatrix, x) rejects a null matrix")
    void device_float_matrix_rejects_null() {
        assertThatThrownBy(() -> rocmImpl.sgemv((DeviceFloatMatrix) null, new float[]{1f}))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("A must not be null");
    }

    @Test
    @DisplayName("sgemv(DeviceHalfMatrix, x) rejects a null matrix")
    void device_half_matrix_rejects_null() {
        assertThatThrownBy(() -> rocmImpl.sgemv((DeviceHalfMatrix) null, new float[]{1f}))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("A must not be null");
    }

    @Test
    @DisplayName("device-resident FP32 path matches CPU reference — 2048×2048")
    void device_resident_fp32_matches_cpu_reference() {
        int rows = 2048, cols = 2048;
        float[] A = randomMatrix(rows, cols, 200);
        float[] x = randomVector(cols, 201);

        float[] cpuResult = LlamaTransformerHandler.matVec(A, x, rows, cols);
        try (DeviceFloatMatrix dA = rocmImpl.upload(A, rows, cols)) {
            float[] rocmResult = rocmImpl.sgemv(dA, x);
            assertThat(rocmResult).hasSize(rows);
            for (int i = 0; i < rows; i++)
                assertThat(rocmResult[i])
                    .as("y[%d]", i)
                    .isCloseTo(cpuResult[i], within(DELTA));
        }
    }

    @Test
    @DisplayName("device-resident FP16 path matches CPU reference — 2048×2048 (FP16 tolerance)")
    void device_resident_fp16_matches_cpu_reference() {
        int rows = 2048, cols = 2048;
        float[] A = randomMatrix(rows, cols, 210);
        float[] x = randomVector(cols, 211);

        float[] cpuResult = LlamaTransformerHandler.matVec(A, x, rows, cols);
        try (DeviceHalfMatrix dA = rocmImpl.uploadHalf(A, rows, cols)) {
            float[] rocmResult = rocmImpl.sgemv(dA, x);
            assertThat(rocmResult).hasSize(rows);
            // FP16 weights/activations: looser tolerance than the FP32 paths.
            for (int i = 0; i < rows; i++)
                assertThat(rocmResult[i])
                    .as("y[%d]", i)
                    .isCloseTo(cpuResult[i], within(1e-3f));
        }
    }

    // ── Constructor validation ────────────────────────────────────────────────

    @Test
    @DisplayName("RocmMatVec(null) throws IllegalArgumentException")
    void constructor_rejects_null_context() {
        assertThatThrownBy(() -> new RocmMatVec(null))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("ctx must not be null");
    }

    // ── Backend identity ──────────────────────────────────────────────────────

    @Test
    @DisplayName("GpuContext.backendLabel() is 'rocm' on AMD system")
    void gpu_context_backend_label_is_rocm() {
        assumeFalse(CudaAvailability.isAvailable(), "Skipping — CUDA present, CUDA wins");
        assertThat(ctx.backendLabel()).isEqualTo("rocm");
    }

    @Test
    @DisplayName("GpuContext.createMatVec() returns RocmMatVec on AMD system")
    void create_mat_vec_returns_rocm_mat_vec() {
        assumeFalse(CudaAvailability.isAvailable(), "Skipping — CUDA present, CUDA wins");
        MatVec mv = ctx.createMatVec();
        assertThat(mv).isInstanceOf(RocmMatVec.class);
    }

    // ── Concurrent safety ─────────────────────────────────────────────────────

    @Test
    @DisplayName("Concurrent sgemv calls produce correct results (4 threads × 256×256)")
    void concurrent_calls_are_correct() throws InterruptedException {
        int rows = 256, cols = 256;
        int threads = 4;
        float[] A = randomMatrix(rows, cols, 300);
        float[] x = randomVector(cols, 301);
        float[] expected = scalarMatVec(A, x, rows, cols);

        Thread[] workers = new Thread[threads];
        float[][] results = new float[threads][];
        for (int t = 0; t < threads; t++) {
            final int idx = t;
            workers[t] = new Thread(() ->
                results[idx] = rocmImpl.sgemv(A, x, rows, cols));
            workers[t].start();
        }
        for (Thread w : workers) w.join();

        for (int t = 0; t < threads; t++) {
            for (int i = 0; i < rows; i++)
                assertThat(results[t][i])
                    .as("thread=%d y[%d]", t, i)
                    .isCloseTo(expected[i], within(DELTA));
        }
    }

    // ── Throughput sanity ─────────────────────────────────────────────────────

    @Test
    @DisplayName("ROCm path has reasonable latency for 32000×2048 output projection (< 3 s for 5 runs)")
    void rocm_path_has_reasonable_latency() {
        int rows = 32000, cols = 2048;
        float[] A = randomMatrix(rows, cols, 200);
        float[] x = randomVector(cols, 201);

        // warm up
        rocmImpl.sgemv(A, x, rows, cols);
        rocmImpl.sgemv(A, x, rows, cols);

        long start = System.nanoTime();
        for (int i = 0; i < 5; i++) rocmImpl.sgemv(A, x, rows, cols);
        long ms = (System.nanoTime() - start) / 1_000_000;

        System.out.printf("32000×2048 rocblas_sgemv — %dms (5 runs)%n", ms);
        assertThat(ms).isLessThan(3_000L);
    }
}
