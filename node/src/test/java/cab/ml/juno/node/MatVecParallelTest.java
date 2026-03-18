package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Correctness tests for CpuForwardPassHandler.matVec().
 *
 * These tests act as a regression anchor: any change to the matVec
 * implementation (scalar → parallel streams, SIMD, etc.) must not change
 * numerical output beyond floating-point rounding (1e-4 tolerance).
 *
 * Run before and after parallelising matVec to confirm the optimisation is
 * numerically identical to the reference scalar implementation.
 */
@DisplayName("CpuForwardPassHandler — matVec correctness")
class MatVecParallelTest {

	// ── Reference scalar implementation (copied from pre-optimisation code) ──

	/** Reference: original single-threaded scalar implementation. */
	private static float[] scalarMatVec(float[] A, float[] x, int rows, int cols) {
		float[] y = new float[rows];
		for (int r = 0; r < rows; r++) {
			float acc = 0f;
			int base = r * cols;
			for (int c = 0; c < cols; c++)
				acc += A[base + c] * x[c];
			y[r] = acc;
		}
		return y;
	}

	// ── Helpers ───────────────────────────────────────────────────────────────

	private static float[] randomVector(int n, long seed) {
		Random rng = new Random(seed);
		float[] v = new float[n];
		for (int i = 0; i < n; i++)
			v[i] = (float) (rng.nextGaussian() * 0.02);
		return v;
	}

	private static float[] randomMatrix(int rows, int cols, long seed) {
		return randomVector(rows * cols, seed);
	}

	private void assertMatvecMatch(float[] A, float[] x, int rows, int cols) {
		float[] expected = scalarMatVec(A, x, rows, cols);
		float[] actual = CpuForwardPassHandler.matVec(A, x, rows, cols);

		assertThat(actual).hasSize(rows);
		for (int i = 0; i < rows; i++) {
			assertThat(actual[i]).as("y[%d]", i).isCloseTo(expected[i], within(1e-4f));
		}
	}

	// ── Tests ─────────────────────────────────────────────────────────────────

	@Test
	@DisplayName("2×3 identity-like matrix produces correct dot products")
	void matVec_2x3_correctness() {
		// A = [[1,0,0],[0,1,0]] x = [3,5,7] → y = [3, 5]
		float[] A = { 1, 0, 0, 0, 1, 0 };
		float[] x = { 3, 5, 7 };

		float[] y = CpuForwardPassHandler.matVec(A, x, 2, 3);

		assertThat(y).hasSize(2);
		assertThat(y[0]).isCloseTo(3f, within(1e-6f));
		assertThat(y[1]).isCloseTo(5f, within(1e-6f));
	}

	@Test
	@DisplayName("All-zeros matrix produces all-zeros output")
	void matVec_zeros_matrix() {
		float[] A = new float[64 * 64];
		float[] x = randomVector(64, 1);

		float[] y = CpuForwardPassHandler.matVec(A, x, 64, 64);

		for (float v : y)
			assertThat(v).isEqualTo(0f);
	}

	@Test
	@DisplayName("All-zeros input vector produces all-zeros output")
	void matVec_zeros_vector() {
		float[] A = randomMatrix(64, 64, 2);
		float[] x = new float[64];

		float[] y = CpuForwardPassHandler.matVec(A, x, 64, 64);

		for (float v : y)
			assertThat(v).isEqualTo(0f);
	}

	@Test
	@DisplayName("Small random matrix (32×32) matches scalar reference")
	void matVec_32x32_matches_scalar() {
		int rows = 32, cols = 32;
		assertMatvecMatch(randomMatrix(rows, cols, 10), randomVector(cols, 11), rows, cols);
	}

	@Test
	@DisplayName("Medium random matrix (256×256) matches scalar reference")
	void matVec_256x256_matches_scalar() {
		int rows = 256, cols = 256;
		assertMatvecMatch(randomMatrix(rows, cols, 20), randomVector(cols, 21), rows, cols);
	}

	@Test
	@DisplayName("Square matrix matching TinyLlama hidden dim (2048×2048) matches scalar")
	void matVec_2048x2048_matches_scalar() {
		int rows = 2048, cols = 2048;
		assertMatvecMatch(randomMatrix(rows, cols, 30), randomVector(cols, 31), rows, cols);
	}

	@Test
	@DisplayName("Non-square matrix (5632×2048 — TinyLlama FFN gate) matches scalar")
	void matVec_ffn_gate_shape_matches_scalar() {
		// TinyLlama: intermediateSize=5632, hiddenDim=2048
		int rows = 5632, cols = 2048;
		assertMatvecMatch(randomMatrix(rows, cols, 40), randomVector(cols, 41), rows, cols);
	}

	@Test
	@DisplayName("Non-square matrix (2048×5632 — TinyLlama FFN down) matches scalar")
	void matVec_ffn_down_shape_matches_scalar() {
		int rows = 2048, cols = 5632;
		assertMatvecMatch(randomMatrix(rows, cols, 50), randomVector(cols, 51), rows, cols);
	}

	@Test
	@DisplayName("Output projection (32000×2048 — vocab logits) matches scalar")
	void matVec_output_projection_matches_scalar() {
		// TinyLlama: vocabSize=32000, hiddenDim=2048
		int rows = 32000, cols = 2048;
		assertMatvecMatch(randomMatrix(rows, cols, 60), randomVector(cols, 61), rows, cols);
	}

	@Test
	@DisplayName("1-row edge case (single dot product) matches scalar")
	void matVec_single_row_matches_scalar() {
		int rows = 1, cols = 512;
		assertMatvecMatch(randomMatrix(rows, cols, 70), randomVector(cols, 71), rows, cols);
	}

	@Test
	@DisplayName("1-column edge case matches scalar")
	void matVec_single_col_matches_scalar() {
		int rows = 128, cols = 1;
		assertMatvecMatch(randomMatrix(rows, cols, 80), randomVector(cols, 81), rows, cols);
	}
}