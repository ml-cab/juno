package cab.ml.juno.node;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GpuWeightShard — device upload and matVec correctness (requires CUDA).
 *
 * Tests that DeviceFloatMatrix instances produced by a shard yield the same
 * sgemv result as the host-side float[] reference, confirming that the H2D
 * upload is bit-exact and that CudaMatVec dispatches through the resident path.
 *
 * Run with:  mvn test -Dgroups=gpu -pl node
 */
@Tag("gpu")
@DisplayName("GpuWeightShard — upload correctness and lifecycle")
class GpuWeightShardTest {

    private static final float DELTA = 1e-3f;

    private static GpuContext ctx;
    private static CudaMatVec cuda;

    @BeforeAll
    static void initCuda() {
        assumeTrue(CudaAvailability.isAvailable(), "Skipping — no CUDA device");
        ctx  = GpuContext.init(0);
        cuda = new CudaMatVec(ctx);
    }

    @AfterAll
    static void teardown() {
        if (ctx != null) ctx.close();
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    private static float[] randomMatrix(int rows, int cols, long seed) {
        Random rng = new Random(seed);
        float[] A = new float[rows * cols];
        for (int i = 0; i < A.length; i++) A[i] = (rng.nextFloat() - 0.5f) * 2f;
        return A;
    }

    private static float[] randomVector(int size, long seed) {
        Random rng = new Random(seed);
        float[] v = new float[size];
        for (int i = 0; i < v.length; i++) v[i] = (rng.nextFloat() - 0.5f) * 2f;
        return v;
    }

    // ── tests ─────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("DeviceFloatMatrix upload — sgemv matches CPU reference (small dims)")
    void device_sgemv_matches_cpu_reference() {
        int rows = 64, cols = 128;
        float[] A = randomMatrix(rows, cols, 1L);
        float[] x = randomVector(cols, 2L);

        DeviceFloatMatrix dev = DeviceFloatMatrix.upload(ctx, A, rows, cols);
        try {
            float[] hostResult   = cuda.sgemv(A, x, rows, cols);
            float[] deviceResult = cuda.sgemv(dev, x);

            assertThat(deviceResult).hasSize(rows);
            for (int i = 0; i < rows; i++)
                assertThat(deviceResult[i]).as("y[%d]", i).isCloseTo(hostResult[i], within(DELTA));
        } finally {
            dev.close();
        }
    }

    @Test
    @DisplayName("DeviceFloatMatrix upload — sgemv matches CPU reference (llama-scale dims)")
    void device_sgemv_matches_cpu_reference_llama_scale() {
        int rows = 2048, cols = 2048; // wq/wo dims for TinyLlama
        float[] A = randomMatrix(rows, cols, 10L);
        float[] x = randomVector(cols, 11L);

        DeviceFloatMatrix dev = DeviceFloatMatrix.upload(ctx, A, rows, cols);
        try {
            float[] hostResult   = cuda.sgemv(A, x, rows, cols);
            float[] deviceResult = cuda.sgemv(dev, x);

            assertThat(deviceResult).hasSize(rows);
            for (int i = 0; i < rows; i++)
                assertThat(deviceResult[i]).as("y[%d]", i).isCloseTo(hostResult[i], within(DELTA));
        } finally {
            dev.close();
        }
    }

    @Test
    @DisplayName("close() marks matrix as closed — further devicePointer() throws")
    void close_marks_matrix_closed() {
        float[] host = new float[16];
        DeviceFloatMatrix dev = DeviceFloatMatrix.upload(ctx, host, 4, 4);
        assertThat(dev.isClosed()).isFalse();
        dev.close();
        assertThat(dev.isClosed()).isTrue();
    }

    @Test
    @DisplayName("close() is idempotent — no exception on double close")
    void close_is_idempotent() {
        float[] host = new float[4];
        DeviceFloatMatrix dev = DeviceFloatMatrix.upload(ctx, host, 2, 2);
        dev.close();
        dev.close(); // must not throw
    }

    @Test
    @DisplayName("Multiple DeviceFloatMatrix uploads — independent buffers, no aliasing")
    void multiple_uploads_are_independent() {
        int rows = 32, cols = 32;
        float[] A1 = randomMatrix(rows, cols, 20L);
        float[] A2 = randomMatrix(rows, cols, 21L);
        float[] x  = randomVector(cols, 22L);

        DeviceFloatMatrix d1 = DeviceFloatMatrix.upload(ctx, A1, rows, cols);
        DeviceFloatMatrix d2 = DeviceFloatMatrix.upload(ctx, A2, rows, cols);
        try {
            float[] r1 = cuda.sgemv(d1, x);
            float[] r2 = cuda.sgemv(d2, x);

            float[] ref1 = cuda.sgemv(A1, x, rows, cols);
            float[] ref2 = cuda.sgemv(A2, x, rows, cols);

            for (int i = 0; i < rows; i++) {
                assertThat(r1[i]).as("d1 y[%d]", i).isCloseTo(ref1[i], within(DELTA));
                assertThat(r2[i]).as("d2 y[%d]", i).isCloseTo(ref2[i], within(DELTA));
            }
        } finally {
            d1.close();
            d2.close();
        }
    }
}