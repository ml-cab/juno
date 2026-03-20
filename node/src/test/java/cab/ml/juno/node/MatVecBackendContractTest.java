package cab.ml.juno.node;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

/**
 * MatVecBackend contract tests.
 *
 * Runs against CpuMatVecBackend so the full suite passes on any machine without a GPU.
 * The same suite is inherited by CudaMatVecBackendTest (see that class) which
 * re-runs all these cases against the real JCublas implementation on AWS.
 *
 * Rules verified:
 *   - Output shape is always float[rows]
 *   - Input arrays are not mutated
 *   - Numerically equivalent to the scalar reference across all shapes
 *   - Edge cases: single row, single col, zeros matrix, zeros vector
 *   - Throws on dimension mismatch
 */
@DisplayName("MatVecBackend contract — CpuMatVecBackend reference")
class MatVecBackendContractTest {

    /** Override in subclasses to test a different MatVecBackend implementation. */
    protected MatVecBackend impl() {
        return CpuMatVecBackend.INSTANCE;
    }

    protected static final float DELTA = 1e-4f;

    // ── Scalar reference ──────────────────────────────────────────────────────

    static float[] scalarMatVec(float[] A, float[] x, int rows, int cols) {
        float[] y = new float[rows];
        for (int r = 0; r < rows; r++) {
            float acc = 0f;
            int base = r * cols;
            for (int c = 0; c < cols; c++) acc += A[base + c] * x[c];
            y[r] = acc;
        }
        return y;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    static float[] randomVector(int n, long seed) {
        Random rng = new Random(seed);
        float[] v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float) (rng.nextGaussian() * 0.02);
        return v;
    }

    static float[] randomMatrix(int rows, int cols, long seed) {
        return randomVector(rows * cols, seed);
    }

    void assertMatchesScalar(float[] A, float[] x, int rows, int cols) {
        float[] expected = scalarMatVec(A, x, rows, cols);
        float[] actual   = impl().sgemv(A, x, rows, cols);

        assertThat(actual).hasSize(rows);
        for (int i = 0; i < rows; i++)
            assertThat(actual[i]).as("y[%d]", i).isCloseTo(expected[i], within(DELTA));
    }

    // ── Shape and output contract ─────────────────────────────────────────────

    @Test
    @DisplayName("Output length equals rows")
    void output_length_is_rows() {
        float[] A = randomMatrix(7, 5, 1);
        float[] x = randomVector(5, 2);
        assertThat(impl().sgemv(A, x, 7, 5)).hasSize(7);
    }

    @Test
    @DisplayName("Returns a new array — does not return A or x")
    void returns_new_array() {
        float[] A = randomMatrix(4, 4, 3);
        float[] x = randomVector(4, 4);
        float[] y = impl().sgemv(A, x, 4, 4);
        assertThat(y).isNotSameAs(A).isNotSameAs(x);
    }

    @Test
    @DisplayName("Input array A is not mutated")
    void A_not_mutated() {
        float[] A    = randomMatrix(8, 8, 5);
        float[] copy = A.clone();
        float[] x    = randomVector(8, 6);
        impl().sgemv(A, x, 8, 8);
        assertThat(A).isEqualTo(copy);
    }

    @Test
    @DisplayName("Input vector x is not mutated")
    void x_not_mutated() {
        float[] A    = randomMatrix(8, 8, 7);
        float[] x    = randomVector(8, 8);
        float[] copy = x.clone();
        impl().sgemv(A, x, 8, 8);
        assertThat(x).isEqualTo(copy);
    }

    // ── Known-value correctness ───────────────────────────────────────────────

    @Test
    @DisplayName("2×3 identity-like: y = [3, 5]")
    void identity_like_2x3() {
        float[] A = { 1, 0, 0,  0, 1, 0 };
        float[] x = { 3, 5, 7 };
        float[] y = impl().sgemv(A, x, 2, 3);
        assertThat(y[0]).isCloseTo(3f, within(1e-6f));
        assertThat(y[1]).isCloseTo(5f, within(1e-6f));
    }

    @Test
    @DisplayName("All-zeros matrix → all-zeros output")
    void zeros_matrix_gives_zeros() {
        float[] A = new float[64 * 64];
        float[] x = randomVector(64, 10);
        for (float v : impl().sgemv(A, x, 64, 64))
            assertThat(v).isEqualTo(0f);
    }

    @Test
    @DisplayName("All-zeros vector → all-zeros output")
    void zeros_vector_gives_zeros() {
        float[] A = randomMatrix(64, 64, 11);
        float[] x = new float[64];
        for (float v : impl().sgemv(A, x, 64, 64))
            assertThat(v).isEqualTo(0f);
    }

    // ── Shape coverage — all LLaMA matrix shapes ──────────────────────────────

    @Test
    @DisplayName("32×32 random matches scalar")
    void shape_32x32() {
        assertMatchesScalar(randomMatrix(32, 32, 20), randomVector(32, 21), 32, 32);
    }

    @Test
    @DisplayName("256×256 random matches scalar")
    void shape_256x256() {
        assertMatchesScalar(randomMatrix(256, 256, 22), randomVector(256, 23), 256, 256);
    }

    @Test
    @DisplayName("2048×2048 — TinyLlama hidden dim — matches scalar")
    void shape_2048x2048() {
        assertMatchesScalar(
            randomMatrix(2048, 2048, 30), randomVector(2048, 31), 2048, 2048);
    }

    @Test
    @DisplayName("5632×2048 — TinyLlama FFN gate — matches scalar")
    void shape_ffn_gate() {
        assertMatchesScalar(
            randomMatrix(5632, 2048, 40), randomVector(2048, 41), 5632, 2048);
    }

    @Test
    @DisplayName("2048×5632 — TinyLlama FFN down — matches scalar")
    void shape_ffn_down() {
        assertMatchesScalar(
            randomMatrix(2048, 5632, 50), randomVector(5632, 51), 2048, 5632);
    }

    @Test
    @DisplayName("32000×2048 — output projection — matches scalar")
    void shape_output_projection() {
        assertMatchesScalar(
            randomMatrix(32000, 2048, 60), randomVector(2048, 61), 32000, 2048);
    }

    @Test
    @DisplayName("1-row edge case matches scalar")
    void shape_single_row() {
        assertMatchesScalar(randomMatrix(1, 512, 70), randomVector(512, 71), 1, 512);
    }

    @Test
    @DisplayName("1-col edge case matches scalar")
    void shape_single_col() {
        assertMatchesScalar(randomMatrix(128, 1, 80), randomVector(1, 81), 128, 1);
    }

    // ── Error cases ───────────────────────────────────────────────────────────

    @Test
    @DisplayName("A.length != rows*cols → IllegalArgumentException")
    void wrong_A_length_throws() {
        assertThatThrownBy(() ->
            impl().sgemv(new float[10], new float[4], 3, 4))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("rows*cols");
    }

    @Test
    @DisplayName("x.length != cols → IllegalArgumentException")
    void wrong_x_length_throws() {
        assertThatThrownBy(() ->
            impl().sgemv(new float[12], new float[5], 3, 4))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("cols");
    }
}