package cab.ml.juno.node;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

/**
 * Resident transpose ({@code W^T * g}) contract for {@link GpuMatVec}.
 *
 * <p>Subclasses supply a live CUDA or ROCm {@link GpuMatVec}. The adjoint gate
 * {@code dot(W*x, g) == dot(x, W^T*g)} must pass before handler integration.
 */
abstract class GpuMatVecTransposeContractTest {

	/** FP32 target from PLAN-LoRA-Tier4 and existing MatVecBackendContractTest. */
	protected static final float FP32_TOL = 1e-4f;

	/** FP16 mixed path target from PLAN-LoRA-Tier4. */
	protected static final float FP16_TOL = 2e-3f;

	protected abstract GpuMatVec impl();

	protected abstract GpuContext ctx();

	protected boolean halfResidentAvailable() {
		return impl().supportsHalfResident();
	}

	// ── Scalar references ─────────────────────────────────────────────────────

	static float[] scalarMatVec(float[] W, float[] x, int rows, int cols) {
		float[] y = new float[rows];
		for (int r = 0; r < rows; r++) {
			float acc = 0f;
			int base = r * cols;
			for (int c = 0; c < cols; c++)
				acc += W[base + c] * x[c];
			y[r] = acc;
		}
		return y;
	}

	/** z = W^T * g for row-major W[rows×cols]; g length rows → z length cols. */
	static float[] scalarTransposeMatVec(float[] W, float[] g, int rows, int cols) {
		float[] z = new float[cols];
		for (int r = 0; r < rows; r++) {
			int base = r * cols;
			float gr = g[r];
			for (int c = 0; c < cols; c++)
				z[c] += W[base + c] * gr;
		}
		return z;
	}

	static float dot(float[] a, float[] b) {
		float s = 0f;
		for (int i = 0; i < a.length; i++)
			s += a[i] * b[i];
		return s;
	}

	static float[] randomVector(int n, long seed) {
		Random rng = new Random(seed);
		float[] v = new float[n];
		for (int i = 0; i < n; i++)
			v[i] = (float) (rng.nextGaussian() * 0.02);
		return v;
	}

	static float[] randomMatrix(int rows, int cols, long seed) {
		return randomVector(rows * cols, seed);
	}

	void assertClose(float[] expected, float[] actual, float tol, String label) {
		assertThat(actual).as(label).hasSize(expected.length);
		for (int i = 0; i < expected.length; i++)
			assertThat(actual[i]).as("%s[%d]", label, i).isCloseTo(expected[i], within(tol));
	}

	// ── FP32 adjoint and dense reference ──────────────────────────────────────

	@Test
	@DisplayName("FP32 adjoint: dot(W*x,g) == dot(x,W^T*g) — square 64×64")
	void fp32_adjoint_square() {
		assertFp32Adjoint(64, 64, 10);
	}

	@Test
	@DisplayName("FP32 adjoint — TinyLlama Q/K/V/O style 2048×2048")
	void fp32_adjoint_hidden() {
		assertFp32Adjoint(128, 128, 11); // smaller than full H for CI speed; layout identical
	}

	@Test
	@DisplayName("FP32 adjoint — rectangular FFN down 64×256 (H×I style)")
	void fp32_adjoint_ffn_down() {
		assertFp32Adjoint(64, 256, 12);
	}

	@Test
	@DisplayName("FP32 adjoint — rectangular FFN gate/up 256×64 (I×H style)")
	void fp32_adjoint_ffn_up() {
		assertFp32Adjoint(256, 64, 13);
	}

	@Test
	@DisplayName("FP32 adjoint — GQA K/V style 32×64 (kvDim×H)")
	void fp32_adjoint_gqa() {
		assertFp32Adjoint(32, 64, 14);
	}

	@Test
	@DisplayName("FP32 sgemvTranspose matches scalar dense reference")
	void fp32_matches_scalar_reference() {
		int rows = 48, cols = 32;
		float[] W = randomMatrix(rows, cols, 20);
		float[] g = randomVector(rows, 21);
		DeviceFloatMatrix dW = impl().upload(W, rows, cols);
		try {
			float[] gpu = impl().sgemvTranspose(dW, g);
			float[] cpu = scalarTransposeMatVec(W, g, rows, cols);
			assertClose(cpu, gpu, FP32_TOL, "W^T*g");
		} finally {
			dW.close();
		}
	}

	@Test
	@DisplayName("FP32 forward+transpose round-trip matches CPU dense pair")
	void fp32_forward_and_transpose_match_cpu() {
		int rows = 40, cols = 24;
		float[] W = randomMatrix(rows, cols, 30);
		float[] x = randomVector(cols, 31);
		float[] g = randomVector(rows, 32);
		DeviceFloatMatrix dW = impl().upload(W, rows, cols);
		try {
			assertClose(scalarMatVec(W, x, rows, cols), impl().sgemv(dW, x), FP32_TOL, "W*x");
			assertClose(scalarTransposeMatVec(W, g, rows, cols), impl().sgemvTranspose(dW, g), FP32_TOL, "W^T*g");
		} finally {
			dW.close();
		}
	}

	// ── FP16 ──────────────────────────────────────────────────────────────────

	@Test
	@DisplayName("FP16 adjoint identity when half-resident is supported")
	void fp16_adjoint_when_supported() {
		org.junit.jupiter.api.Assumptions.assumeTrue(halfResidentAvailable(),
				"Skipping — FP16 resident GEMV unsupported on this device");
		int rows = 64, cols = 48;
		float[] W = randomMatrix(rows, cols, 40);
		float[] x = randomVector(cols, 41);
		float[] g = randomVector(rows, 42);
		DeviceHalfMatrix dW = impl().uploadHalf(W, rows, cols);
		try {
			float[] Wx = impl().sgemv(dW, x);
			float[] WTg = impl().sgemvTranspose(dW, g);
			assertThat(dot(Wx, g)).as("adjoint").isCloseTo(dot(x, WTg), within(FP16_TOL));
		} finally {
			dW.close();
		}
	}

	@Test
	@DisplayName("FP16 sgemvTranspose close to dense FP32 reference")
	void fp16_close_to_fp32_dense() {
		org.junit.jupiter.api.Assumptions.assumeTrue(halfResidentAvailable(),
				"Skipping — FP16 resident GEMV unsupported on this device");
		int rows = 64, cols = 48;
		float[] W = randomMatrix(rows, cols, 50);
		float[] g = randomVector(rows, 51);
		DeviceHalfMatrix dW = impl().uploadHalf(W, rows, cols);
		try {
			float[] gpu = impl().sgemvTranspose(dW, g);
			float[] cpu = scalarTransposeMatVec(W, g, rows, cols);
			// Compare against dense FP32; weights were rounded to FP16 on upload.
			assertClose(cpu, gpu, 8e-2f, "FP16 W^T*g vs FP32 ref");
		} finally {
			dW.close();
		}
	}

	// ── Error contracts ───────────────────────────────────────────────────────

	@Test
	@DisplayName("sgemvTranspose rejects null matrix")
	void rejects_null_matrix() {
		assertThatThrownBy(() -> impl().sgemvTranspose((DeviceFloatMatrix) null, new float[1]))
				.isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	@DisplayName("sgemvTranspose rejects closed DeviceFloatMatrix")
	void rejects_closed_float_matrix() {
		DeviceFloatMatrix dW = impl().upload(randomMatrix(4, 3, 60), 4, 3);
		dW.close();
		assertThatThrownBy(() -> impl().sgemvTranspose(dW, new float[4]))
				.isInstanceOf(IllegalStateException.class)
				.hasMessageContaining("closed");
	}

	@Test
	@DisplayName("sgemvTranspose rejects g.length != rows")
	void rejects_wrong_g_length() {
		DeviceFloatMatrix dW = impl().upload(randomMatrix(5, 3, 61), 5, 3);
		try {
			assertThatThrownBy(() -> impl().sgemvTranspose(dW, new float[3]))
					.isInstanceOf(IllegalArgumentException.class)
					.hasMessageContaining("g.length");
		} finally {
			dW.close();
		}
	}

	private void assertFp32Adjoint(int rows, int cols, long seed) {
		float[] W = randomMatrix(rows, cols, seed);
		float[] x = randomVector(cols, seed + 1);
		float[] g = randomVector(rows, seed + 2);
		DeviceFloatMatrix dW = impl().upload(W, rows, cols);
		try {
			float[] Wx = impl().sgemv(dW, x);
			float[] WTg = impl().sgemvTranspose(dW, g);
			assertThat(Wx).hasSize(rows);
			assertThat(WTg).hasSize(cols);
			assertThat(dot(Wx, g)).as("dot(W*x,g) == dot(x,W^T*g)").isCloseTo(dot(x, WTg), within(FP32_TOL));
		} finally {
			dW.close();
		}
	}
}
