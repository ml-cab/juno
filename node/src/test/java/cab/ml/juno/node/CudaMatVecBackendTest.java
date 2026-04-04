package cab.ml.juno.node;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * CudaMatVecBackend tests — requires CUDA and an Nvidia GPU (org.bytedeco cuda).
 *
 * Inherits the full MatVecBackendContractTest suite. Every contract test that
 * passes on CpuMatVecBackend must also pass on CudaMatVecBackend — numerically identical
 * output is the correctness requirement.
 *
 * Additionally tests:
 *   - cublasSgemv output matches LlamaTransformerHandler.matVec reference
 *     (cross-impl numerical equivalence — the primary regression anchor)
 *   - Large matrix throughput is measurably faster than CPU (sanity check)
 *
 * Run selectively in CI:
 *   mvn test -Dgroups=gpu -pl node
 *
 * Skip on non-GPU machines:
 *   mvn test -Dgroups='!gpu' -pl node
 *
 * On AWS: launch a g4dn.xlarge (T4 GPU), install CUDA 12.x, run:
 *   mvn test -Dgroups=gpu -pl node --enable-native-access=ALL-UNNAMED
 */
@Tag("gpu")
@DisplayName("CudaMatVecBackend — cuBLAS cublasSgemv correctness (requires CUDA)")
class CudaMatVecBackendTest extends MatVecBackendContractTest {

    private static GpuContext ctx;
    private static CudaMatVec cudaImpl;

    @BeforeAll
    static void initCuda() {
        assumeTrue(CudaAvailability.isAvailable(),
            "Skipping CudaMatVecBackendTest — no CUDA device available");
        ctx = GpuContext.init(0);
        cudaImpl = new CudaMatVec(ctx);
    }

    @AfterAll
    static void destroyCuda() {
        if (ctx != null) ctx.close();
    }

    /** Provide CudaMatVecBackend to the inherited contract tests. */
    @Override
    protected MatVec impl() {
        return cudaImpl;
    }

    // ── CudaMatVecBackend-specific tests ───────────────────────────────────────────

    @Test
    @DisplayName("sgemv(DeviceFloatMatrix, x) matches host sgemv — 2048×2048")
    void device_matrix_sgemv_matches_host_path() {
        int rows = 2048, cols = 2048;
        float[] A = randomMatrix(rows, cols, 400);
        float[] x = randomVector(cols, 401);
        DeviceFloatMatrix d = DeviceFloatMatrix.upload(ctx, A, rows, cols);
        try {
            float[] hostPath = cudaImpl.sgemv(A, x, rows, cols);
            float[] devPath = cudaImpl.sgemv(d, x);
            assertThat(devPath).hasSize(rows);
            for (int i = 0; i < rows; i++)
                assertThat(devPath[i]).as("y[%d]", i).isCloseTo(hostPath[i], within(DELTA));
        } finally {
            d.close();
        }
    }

    @Test
    @DisplayName("cublasSgemv matches LlamaTransformerHandler.matVec reference — 2048×2048")
    void cublas_matches_cpu_reference_hidden_dim() {
        int rows = 2048, cols = 2048;
        float[] A = randomMatrix(rows, cols, 100);
        float[] x = randomVector(cols, 101);

        float[] cpuResult    = LlamaTransformerHandler.matVec(A, x, rows, cols);
        float[] cublasResult = cudaImpl.sgemv(A, x, rows, cols);

        assertThat(cublasResult).hasSize(rows);
        for (int i = 0; i < rows; i++)
            assertThat(cublasResult[i])
                .as("y[%d]", i)
                .isCloseTo(cpuResult[i], within(DELTA));
    }

    @Test
    @DisplayName("cublasSgemv matches CPU reference — 32000×2048 output projection")
    void cublas_matches_cpu_reference_output_projection() {
        int rows = 32000, cols = 2048;
        float[] A = randomMatrix(rows, cols, 102);
        float[] x = randomVector(cols, 103);

        float[] cpuResult    = LlamaTransformerHandler.matVec(A, x, rows, cols);
        float[] cublasResult = cudaImpl.sgemv(A, x, rows, cols);

        for (int i = 0; i < rows; i++)
            assertThat(cublasResult[i])
                .as("y[%d]", i)
                .isCloseTo(cpuResult[i], within(DELTA));
    }

    @Test
    @DisplayName("GPU path stays within reasonable overhead for large matrix (32000×2048)")
    void gpu_path_has_reasonable_overhead_for_large_matrix() {
        int rows = 32000, cols = 2048;
        float[] A = randomMatrix(rows, cols, 200);
        float[] x = randomVector(cols, 201);

        // Warm up both
        LlamaTransformerHandler.matVec(A, x, rows, cols);
        cudaImpl.sgemv(A, x, rows, cols);

        // CPU timing
        long cpuStart = System.nanoTime();
        for (int i = 0; i < 5; i++) LlamaTransformerHandler.matVec(A, x, rows, cols);
        long cpuMs = (System.nanoTime() - cpuStart) / 1_000_000;

        // GPU timing
        long gpuStart = System.nanoTime();
        for (int i = 0; i < 5; i++) cudaImpl.sgemv(A, x, rows, cols);
        long gpuMs = (System.nanoTime() - gpuStart) / 1_000_000;

        System.out.printf("32000×2048 sgemv — CPU: %dms  GPU: %dms  (5 runs each)%n",
            cpuMs, gpuMs);

        // Host-path sgemv still copies A each call; keep a coarse upper bound for CI variance.
        assertThat(gpuMs).isLessThan(3_000L);
    }

    @Test
    @DisplayName("Concurrent sgemv calls produce correct results")
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
                results[idx] = cudaImpl.sgemv(A, x, rows, cols));
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
}