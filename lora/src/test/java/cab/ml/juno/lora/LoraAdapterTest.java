package cab.ml.juno.lora;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.util.Random;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link LoraAdapter}.
 *
 * <h2>Key test: numerical gradient check</h2> The backward pass is verified
 * against finite-difference estimates of dL/dA, dL/dB, and dL/dx. This is the
 * gold-standard test for any backprop implementation — if the analytical
 * gradients match the finite-difference gradients, the chain rule was applied
 * correctly.
 *
 * <h2>Gradient check methodology</h2> For each parameter θ:
 * 
 * <pre>
 *   dL/dθ ≈ (L(θ + h) - L(θ - h)) / (2h),   h = 1e-3
 * </pre>
 * 
 * We compare this numerical estimate against the analytical gradient from
 * {@link LoraAdapter#backward(float[], float[])}. Tolerances are generous
 * (1e-3) because the finite-difference estimate has O(h²) truncation error and
 * the parameters are float32.
 *
 * <h2>What to watch during testing</h2>
 * <ul>
 * <li>If {@code gradA} check passes but {@code gradB} fails: the B
 * outer-product term {@code scale × gradDelta ⊗ h^T} has a sign or dimension
 * error.
 * <li>If {@code gradX} check fails: the transpose A^T × gradH path has an index
 * transposition error (common bug: {@code a[r*inDim + j]} vs
 * {@code a[j*rank + r]}).
 * <li>If all gradients are zero: {@link LoraAdapter#zeroGrad()} was called
 * accidentally before reading the results.
 * <li>If gradients are 2× expected: the same (gradDelta, x) pair was passed to
 * backward twice without zeroGrad().
 * </ul>
 */
@DisplayName("LoraAdapter")
class LoraAdapterTest {

	private static final float FD_H = 1e-3f; // finite-difference step
	private static final double TOL = 2e-3; // relative tolerance for grad check
	private static final float ABS_TOL = 1e-4f; // absolute tolerance for near-zero grads

	private Random rng;

	@BeforeEach
	void setUp() {
		rng = new Random(42);
	}

	// ── Construction ──────────────────────────────────────────────────────────

	@Test
	@DisplayName("B is zero at construction — delta is zero for any input")
	void b_zero_init_means_zero_delta() {
		LoraAdapter lora = new LoraAdapter(4, 8, 16, 4f, rng);
		float[] x = randomVec(8, 1);
		float[] delta = lora.forward(x);
		for (float d : delta)
			assertThat(d).isEqualTo(0f);
	}

	@Test
	@DisplayName("Forward output has correct shape [outDim]")
	void forward_output_shape() {
		LoraAdapter lora = new LoraAdapter(4, 16, 32, 4f, rng);
		assertThat(lora.forward(randomVec(16, 2))).hasSize(32);
	}

	@Test
	@DisplayName("Backward gradX has correct shape [inDim]")
	void backward_gradX_shape() {
		LoraAdapter lora = new LoraAdapter(4, 16, 32, 4f, rng);
		float[] gradX = lora.backward(randomVec(32, 3), randomVec(16, 4));
		assertThat(gradX).hasSize(16);
	}

	@Test
	@DisplayName("zeroGrad clears all accumulators")
	void zero_grad_clears_accumulators() {
		LoraAdapter lora = new LoraAdapter(4, 8, 16, 4f, rng);
		lora.backward(randomVec(16, 5), randomVec(8, 6));
		lora.zeroGrad();
		for (float g : lora.gradA())
			assertThat(g).isEqualTo(0f);
		for (float g : lora.gradB())
			assertThat(g).isEqualTo(0f);
	}

	@Test
	@DisplayName("Backward gradients accumulate over multiple calls")
	void gradients_accumulate() {
		LoraAdapter lora = new LoraAdapter(4, 8, 16, 4f, rng);
		float[] gradDelta = randomVec(16, 7);
		float[] x = randomVec(8, 8);
		lora.backward(gradDelta, x);
		float[] gradA1 = lora.gradA().clone();

		// second call without zeroGrad — gradients should double
		lora.backward(gradDelta, x);
		for (int i = 0; i < gradA1.length; i++)
			assertThat(lora.gradA()[i]).isCloseTo(gradA1[i] * 2, within(ABS_TOL));
	}

	// ── Gradient checks (the critical tests) ─────────────────────────────────

	@Nested
	@DisplayName("Numerical gradient check — rank=4, in=8, out=16, alpha=4")
	class GradientCheck_Small {

		private static final int RANK = 4, IN = 8, OUT = 16;
		private static final float ALPHA = 4f;

		@Test
		@DisplayName("dL/dA matches finite difference")
		void gradA_matches_fd() {
			LoraAdapter lora = makeNonZero(RANK, IN, OUT, ALPHA);
			float[] x = randomVec(IN, 10);
			float[] gradDelta = randomVec(OUT, 11);

			checkGradA(lora, x, gradDelta, TOL);
		}

		@Test
		@DisplayName("dL/dB matches finite difference")
		void gradB_matches_fd() {
			LoraAdapter lora = makeNonZero(RANK, IN, OUT, ALPHA);
			float[] x = randomVec(IN, 12);
			float[] gradDelta = randomVec(OUT, 13);

			checkGradB(lora, x, gradDelta, TOL);
		}

		@Test
		@DisplayName("dL/dX matches finite difference")
		void gradX_matches_fd() {
			LoraAdapter lora = makeNonZero(RANK, IN, OUT, ALPHA);
			float[] x = randomVec(IN, 14);
			float[] gradDelta = randomVec(OUT, 15);

			checkGradX(lora, x, gradDelta, TOL);
		}
	}

	@Nested
	@DisplayName("Numerical gradient check — rank=8, in=32, out=64, alpha=8")
	class GradientCheck_Medium {

		@Test
		@DisplayName("dL/dA matches finite difference")
		void gradA_matches_fd() {
			LoraAdapter lora = makeNonZero(8, 32, 64, 8f);
			checkGradA(lora, randomVec(32, 20), randomVec(64, 21), TOL);
		}

		@Test
		@DisplayName("dL/dB matches finite difference")
		void gradB_matches_fd() {
			LoraAdapter lora = makeNonZero(8, 32, 64, 8f);
			checkGradB(lora, randomVec(32, 22), randomVec(64, 23), TOL);
		}

		@Test
		@DisplayName("dL/dX matches finite difference")
		void gradX_matches_fd() {
			LoraAdapter lora = makeNonZero(8, 32, 64, 8f);
			checkGradX(lora, randomVec(32, 24), randomVec(64, 25), TOL);
		}
	}

	// ── Scale invariant ───────────────────────────────────────────────────────

	@Test
	@DisplayName("scale = alpha/rank is applied to both forward delta and backward gradient")
	void scale_applied_consistently() {
		// Use rank=4, alpha=8 → scale=2
		LoraAdapter loraScale2 = makeNonZero(4, 8, 16, 8f); // scale = 2
		LoraAdapter loraScale1 = LoraAdapter.fromWeights(4, 8, 16, 4f, // scale = 1
				loraScale2.a().clone(), loraScale2.b().clone());

		float[] x = randomVec(8, 30);
		float[] gradDelta = randomVec(16, 31);

		float[] delta2 = loraScale2.forward(x);
		float[] delta1 = loraScale1.forward(x);
		for (int i = 0; i < delta2.length; i++)
			assertThat(delta2[i]).isCloseTo(delta1[i] * 2f, within(1e-5f));

		loraScale2.backward(gradDelta, x);
		loraScale1.backward(gradDelta, x);
		for (int i = 0; i < loraScale2.gradA().length; i++)
			assertThat(loraScale2.gradA()[i]).isCloseTo(loraScale1.gradA()[i] * 2f, within(1e-5f));
	}

	// ── Deterministic train-only dropout ──────────────────────────────────────

	@Nested
	@DisplayName("Train dropout")
	class TrainDropout {

		@Test
		@DisplayName("dropout rate 0 matches inference forward/backward")
		void dropout_zero_equivalence() {
			LoraAdapter a = makeNonZero(4, 8, 16, 4f);
			LoraAdapter b = LoraAdapter.fromWeights(4, 8, 16, 4f, a.a().clone(), a.b().clone());
			float[] x = randomVec(8, 40);
			float[] g = randomVec(16, 41);

			float[] d0 = a.forward(x);
			float[] d1 = b.forwardTrain(x, 0f, 1L, 1, 0, 0, 0, 0);
			assertThat(d1).containsExactly(d0);

			float[] gx0 = a.backward(g, x);
			float[] gx1 = b.backwardTrain(g, x, 0f, 1L, 1, 0, 0, 0, 0);
			assertThat(gx1).containsExactly(gx0);
			assertThat(b.gradA()).containsExactly(a.gradA());
			assertThat(b.gradB()).containsExactly(a.gradB());
		}

		@Test
		@DisplayName("same seed regenerates identical masks; different seeds diverge")
		void same_and_different_seeds() {
			LoraAdapter a = makeNonZero(4, 8, 16, 4f);
			float[] x = randomVec(8, 42);
			float rate = 0.3f;
			float[] d1 = a.forwardTrain(x, rate, 99L, 2, 1, 3, 0, 0);
			float[] d2 = a.forwardTrain(x, rate, 99L, 2, 1, 3, 0, 0);
			float[] d3 = a.forwardTrain(x, rate, 100L, 2, 1, 3, 0, 0);
			assertThat(d2).containsExactly(d1);
			boolean differ = false;
			for (int i = 0; i < d1.length; i++)
				if (d1[i] != d3[i]) {
					differ = true;
					break;
				}
			assertThat(differ).isTrue();
		}

		@Test
		@DisplayName("different chunk ordinals produce different masks at the same token position")
		void different_chunks_differ() {
			LoraAdapter a = makeNonZero(4, 8, 16, 4f);
			float[] x = randomVec(8, 43);
			float[] d0 = a.forwardTrain(x, 0.4f, 7L, 1, 0, 5, 0, 0);
			float[] d1 = a.forwardTrain(x, 0.4f, 7L, 1, 1, 5, 0, 0);
			boolean differ = false;
			for (int i = 0; i < d0.length; i++)
				if (d0[i] != d1[i]) {
					differ = true;
					break;
				}
			assertThat(differ).isTrue();
		}

		@Test
		@DisplayName("kept coordinates match inverted-scale expectation over masks")
		void mask_expectation() {
			float rate = 0.25f;
			int kept = 0;
			int n = 2000;
			for (int j = 0; j < n; j++)
				if (LoraDropout.keep(123L, 1, 0, 0, 0, 0, j, rate))
					kept++;
			double frac = kept / (double) n;
			assertThat(frac).isCloseTo(1.0 - rate, within(0.05));
			assertThat(LoraDropout.invertedScale(rate)).isCloseTo(1f / (1f - rate), within(1e-6f));
		}

		@Test
		@DisplayName("fixed-mask finite differences match A/B/X training grads")
		void fixed_mask_finite_differences() {
			LoraAdapter lora = makeNonZero(4, 8, 16, 4f);
			float[] x = randomVec(8, 50);
			float[] gradDelta = randomVec(16, 51);
			float rate = 0.35f;
			long seed = 55L;
			int update = 3, chunk = 2, pos = 4, layer = 1, proj = 0;

			lora.zeroGrad();
			float[] analyticX = lora.backwardTrain(gradDelta, x, rate, seed, update, chunk, pos, layer, proj);
			float[] analyticA = lora.gradA().clone();
			float[] analyticB = lora.gradB().clone();

			for (int i = 0; i < lora.a().length; i++) {
				float orig = lora.a()[i];
				lora.a()[i] = orig + FD_H;
				float hi = dot(gradDelta, lora.forwardTrain(x, rate, seed, update, chunk, pos, layer, proj));
				lora.a()[i] = orig - FD_H;
				float lo = dot(gradDelta, lora.forwardTrain(x, rate, seed, update, chunk, pos, layer, proj));
				lora.a()[i] = orig;
				assertCloseEnough("gradA[" + i + "]", analyticA[i], (hi - lo) / (2 * FD_H), TOL);
			}
			for (int i = 0; i < lora.b().length; i++) {
				float orig = lora.b()[i];
				lora.b()[i] = orig + FD_H;
				float hi = dot(gradDelta, lora.forwardTrain(x, rate, seed, update, chunk, pos, layer, proj));
				lora.b()[i] = orig - FD_H;
				float lo = dot(gradDelta, lora.forwardTrain(x, rate, seed, update, chunk, pos, layer, proj));
				lora.b()[i] = orig;
				assertCloseEnough("gradB[" + i + "]", analyticB[i], (hi - lo) / (2 * FD_H), TOL);
			}
			for (int i = 0; i < x.length; i++) {
				float orig = x[i];
				x[i] = orig + FD_H;
				float hi = dot(gradDelta, lora.forwardTrain(x, rate, seed, update, chunk, pos, layer, proj));
				x[i] = orig - FD_H;
				float lo = dot(gradDelta, lora.forwardTrain(x, rate, seed, update, chunk, pos, layer, proj));
				x[i] = orig;
				assertCloseEnough("gradX[" + i + "]", analyticX[i], (hi - lo) / (2 * FD_H), TOL);
			}
		}

		@Test
		@DisplayName("dropped coordinates have zero input gradient")
		void dropped_coordinate_gradX_zero() {
			LoraAdapter lora = makeNonZero(4, 8, 16, 4f);
			float[] x = randomVec(8, 60);
			float[] g = randomVec(16, 61);
			float rate = 0.5f;
			long seed = 9L;
			float[] gradX = lora.backwardTrain(g, x, rate, seed, 1, 0, 0, 0, 0);
			for (int j = 0; j < x.length; j++) {
				if (!LoraDropout.keep(seed, 1, 0, 0, 0, 0, j, rate))
					assertThat(gradX[j]).isEqualTo(0f);
			}
		}

		@Test
		@DisplayName("invalid dropout rates are rejected")
		void invalid_rates() {
			assertThatThrownBy(() -> LoraDropout.validateRate(-0.1f))
					.isInstanceOf(IllegalArgumentException.class);
			assertThatThrownBy(() -> LoraDropout.validateRate(1f)).isInstanceOf(IllegalArgumentException.class);
			assertThatThrownBy(() -> LoraDropout.validateRate(Float.NaN))
					.isInstanceOf(IllegalArgumentException.class);
		}
	}

	// ── Helpers ───────────────────────────────────────────────────────────────

	/**
	 * Make an adapter with random A and non-zero B so the gradient check exercises
	 * non-trivial paths through B.
	 */
	private LoraAdapter makeNonZero(int rank, int inDim, int outDim, float alpha) {
		LoraAdapter lora = new LoraAdapter(rank, inDim, outDim, alpha, rng);
		// B starts at zero; initialise with small random values so the backward
		// path through B is exercised
		Random b = new Random(99);
		for (int i = 0; i < lora.b().length; i++)
			lora.b()[i] = (float) (b.nextGaussian() * 0.02);
		return lora;
	}

	/**
	 * Finite-difference gradient check for A. Loss = sum(gradDelta * forward(x))
	 * (linear surrogate loss).
	 */
	private void checkGradA(LoraAdapter lora, float[] x, float[] gradDelta, double tol) {
		lora.zeroGrad();
		lora.backward(gradDelta, x);
		float[] analytic = lora.gradA().clone();

		for (int i = 0; i < lora.a().length; i++) {
			float orig = lora.a()[i];
			lora.a()[i] = orig + FD_H;
			float lossHi = dot(gradDelta, lora.forward(x));
			lora.a()[i] = orig - FD_H;
			float lossLo = dot(gradDelta, lora.forward(x));
			lora.a()[i] = orig;

			float fd = (lossHi - lossLo) / (2 * FD_H);
			assertCloseEnough("gradA[" + i + "]", analytic[i], fd, tol);
		}
	}

	/** Finite-difference gradient check for B. */
	private void checkGradB(LoraAdapter lora, float[] x, float[] gradDelta, double tol) {
		lora.zeroGrad();
		lora.backward(gradDelta, x);
		float[] analytic = lora.gradB().clone();

		for (int i = 0; i < lora.b().length; i++) {
			float orig = lora.b()[i];
			lora.b()[i] = orig + FD_H;
			float lossHi = dot(gradDelta, lora.forward(x));
			lora.b()[i] = orig - FD_H;
			float lossLo = dot(gradDelta, lora.forward(x));
			lora.b()[i] = orig;

			float fd = (lossHi - lossLo) / (2 * FD_H);
			assertCloseEnough("gradB[" + i + "]", analytic[i], fd, tol);
		}
	}

	/** Finite-difference gradient check for x. */
	private void checkGradX(LoraAdapter lora, float[] x, float[] gradDelta, double tol) {
		lora.zeroGrad();
		float[] analytic = lora.backward(gradDelta, x);

		for (int i = 0; i < x.length; i++) {
			float orig = x[i];
			x[i] = orig + FD_H;
			float lossHi = dot(gradDelta, lora.forward(x));
			x[i] = orig - FD_H;
			float lossLo = dot(gradDelta, lora.forward(x));
			x[i] = orig;

			float fd = (lossHi - lossLo) / (2 * FD_H);
			assertCloseEnough("gradX[" + i + "]", analytic[i], fd, tol);
		}
	}

	private static void assertCloseEnough(String label, float analytic, float fd, double tol) {
		double absErr = Math.abs(analytic - fd);
		// Both values are near zero: absolute agreement is all we can expect from
		// float32 finite differences at this scale. Relative tolerance is meaningless.
		if (absErr < 5e-4f)
			return;
		float scale = Math.max(Math.abs(analytic), Math.abs(fd));
		if (scale < 1e-6f)
			return; // both essentially zero — skip
		double relErr = absErr / scale;
		assertThat(relErr).as("%s: analytic=%.6f  fd=%.6f  relErr=%.4f > tol=%.4f", label, analytic, fd, relErr, tol)
				.isLessThanOrEqualTo(tol);
	}

	private static float dot(float[] a, float[] b) {
		float sum = 0f;
		for (int i = 0; i < a.length; i++)
			sum += a[i] * b[i];
		return sum;
	}

	private float[] randomVec(int n, long seed) {
		Random r = new Random(seed);
		float[] v = new float[n];
		for (int i = 0; i < n; i++)
			v[i] = (float) (r.nextGaussian() * 0.3);
		return v;
	}
}