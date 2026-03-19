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
 * CublasMatVec tests — requires CUDA and an Nvidia GPU (org.bytedeco cuda).
 *
 * Inherits the full GpuMatVecContractTest suite. Every contract test that
 * passes on CpuMatVec must also pass on CublasMatVec — numerically identical
 * output is the correctness requirement.
 *
 * Additionally tests:
 *   - cublasSgemv output matches CpuForwardPassHandler.matVec reference
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
@DisplayName("CublasMatVec — cuBLAS cublasSgemv correctness (requires CUDA)")
class CublasMatVecTest extends GpuMatVecContractTest {

    private static GpuContext ctx;
    private static CublasMatVec cublasImpl;

    @BeforeAll
    static void initCuda() {
        assumeTrue(CudaAvailability.isAvailable(),
            "Skipping CublasMatVecTest — no CUDA device available");
        ctx = GpuContext.init(0);
        cublasImpl = new CublasMatVec(ctx);
    }

    @AfterAll
    static void destroyCuda() {
        if (ctx != null) ctx.close();
    }

    /** Provide CublasMatVec to the inherited contract tests. */
    @Override
    protected GpuMatVec impl() {
        return cublasImpl;
    }

    // ── CublasMatVec-specific tests ───────────────────────────────────────────

    @Test
    @DisplayName("cublasSgemv matches CpuForwardPassHandler.matVec reference — 2048×2048")
    void cublas_matches_cpu_reference_hidden_dim() {
        int rows = 2048, cols = 2048;
        float[] A = randomMatrix(rows, cols, 100);
        float[] x = randomVector(cols, 101);

        float[] cpuResult    = CpuForwardPassHandler.matVec(A, x, rows, cols);
        float[] cublasResult = cublasImpl.sgemv(A, x, rows, cols);

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

        float[] cpuResult    = CpuForwardPassHandler.matVec(A, x, rows, cols);
        float[] cublasResult = cublasImpl.sgemv(A, x, rows, cols);

        for (int i = 0; i < rows; i++)
            assertThat(cublasResult[i])
                .as("y[%d]", i)
                .isCloseTo(cpuResult[i], within(DELTA));
    }

    @Test
    @DisplayName("GPU is faster than CPU for large matrix (32000×2048)")
    void gpu_is_faster_than_cpu_for_large_matrix() {
        int rows = 32000, cols = 2048;
        float[] A = randomMatrix(rows, cols, 200);
        float[] x = randomVector(cols, 201);

        // Warm up both
        CpuForwardPassHandler.matVec(A, x, rows, cols);
        cublasImpl.sgemv(A, x, rows, cols);

        // CPU timing
        long cpuStart = System.nanoTime();
        for (int i = 0; i < 5; i++) CpuForwardPassHandler.matVec(A, x, rows, cols);
        long cpuMs = (System.nanoTime() - cpuStart) / 1_000_000;

        // GPU timing
        long gpuStart = System.nanoTime();
        for (int i = 0; i < 5; i++) cublasImpl.sgemv(A, x, rows, cols);
        long gpuMs = (System.nanoTime() - gpuStart) / 1_000_000;

        System.out.printf("32000×2048 sgemv — CPU: %dms  GPU: %dms  (5 runs each)%n",
            cpuMs, gpuMs);

        // GPU should be at least 2x faster on a real Nvidia card for this size.
        // On a T4, expect 10-50x for large projections.
        assertThat(gpuMs).isLessThan(cpuMs);
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
                results[idx] = cublasImpl.sgemv(A, x, rows, cols));
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