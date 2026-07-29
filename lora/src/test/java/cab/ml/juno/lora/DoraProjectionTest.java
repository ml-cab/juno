package cab.ml.juno.lora;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

@DisplayName("DoraProjection")
class DoraProjectionTest {

	private static final float FD_H = 1e-3f;
	private static final double TOL = 3e-3;

	@Test
	@DisplayName("B=0 with magnitude=row-norms reproduces base Wx")
	void b0_identity() {
		int in = 6, out = 4;
		float[] w = randomMatrix(out, in, 1);
		LoraAdapter lora = new LoraAdapter(
				LoraAdapterConfig.of(2, 2f, LoraScaling.STANDARD, LoraInitialization.LEGACY_NORMAL, LoraMode.DORA), in,
				out, new Random(2));
		assertThat(lora.b()).containsOnly(0f);
		DoraMagnitude mag = DoraProjection.magnitudeFromBaseRows(w, out, in);
		DoraProjection dora = new DoraProjection(w, lora, mag);
		float[] x = randomVec(in, 3);
		float[] expected = matvec(w, out, in, x);
		assertThat(dora.forward(x)).containsExactly(expected, within(1e-5f));
	}

	@Test
	@DisplayName("coefficients apply per output row")
	void row_axis_scaling() {
		int in = 3, out = 2;
		float[] w = new float[] { 1f, 0f, 0f, 0f, 1f, 0f };
		LoraAdapter lora = LoraAdapter.fromWeights(
				LoraAdapterConfig.of(1, 1f, LoraScaling.STANDARD, LoraInitialization.LEGACY_NORMAL, LoraMode.DORA), in,
				out, new float[] { 0f, 0f, 0f }, new float[] { 0f, 0f });
		DoraMagnitude mag = DoraMagnitude.fromValues(new float[] { 2f, 3f }); // norms are 1,1
		DoraProjection dora = new DoraProjection(w, lora, mag);
		float[] y = dora.forward(new float[] { 1f, 1f, 0f });
		assertThat(y[0]).isCloseTo(2f, within(1e-5f));
		assertThat(y[1]).isCloseTo(3f, within(1e-5f));
	}

	@Test
	@DisplayName("dense reference forward matches manual formula")
	void dense_reference_forward() {
		int in = 5, out = 3, rank = 2;
		float[] w = randomMatrix(out, in, 10);
		LoraAdapter lora = makeDora(rank, in, out, 4f, 11);
		DoraMagnitude mag = DoraMagnitude.fromValues(new float[] { 1.1f, 0.7f, 1.3f });
		DoraProjection dora = new DoraProjection(w, lora, mag);
		dora.refresh();
		float[] x = randomVec(in, 12);

		float[] direction = new float[out];
		float[] wx = matvec(w, out, in, x);
		float[] delta = lora.forward(x);
		for (int i = 0; i < out; i++)
			direction[i] = wx[i] + delta[i];
		float[] expected = new float[out];
		for (int i = 0; i < out; i++)
			expected[i] = dora.coefficients()[i] * direction[i];
		assertThat(dora.forward(x)).containsExactly(expected, within(1e-5f));
	}

	@Test
	@DisplayName("epsilon floors zero-norm rows")
	void epsilon_floor() {
		float[] w = new float[4]; // 2x2 zeros
		LoraAdapter lora = LoraAdapter.fromWeights(
				LoraAdapterConfig.of(1, 1f, LoraScaling.STANDARD, LoraInitialization.LEGACY_NORMAL, LoraMode.DORA), 2, 2,
				new float[] { 0f, 0f }, new float[] { 0f, 0f });
		DoraMagnitude mag = DoraMagnitude.fromValues(new float[] { 1f, 1f });
		DoraProjection dora = new DoraProjection(w, lora, mag);
		dora.refresh();
		assertThat(dora.coefficients()[0]).isCloseTo(1f / DoraProjection.EPS, within(1e-3f));
	}

	@Test
	@DisplayName("markDirty forces norm refresh after A/B mutation")
	void cache_invalidation() {
		int in = 4, out = 3;
		float[] w = randomMatrix(out, in, 20);
		LoraAdapter lora = makeDora(2, in, out, 2f, 21);
		DoraMagnitude mag = DoraProjection.magnitudeFromBaseRows(w, out, in);
		DoraProjection dora = new DoraProjection(w, lora, mag);
		float[] c0 = dora.coefficients().clone();
		for (int i = 0; i < lora.b().length; i++)
			lora.b()[i] = 0.5f;
		assertThat(dora.coefficients()).containsExactly(c0); // stale until dirty
		dora.markDirty();
		assertThat(dora.dirty()).isTrue();
		float[] c1 = dora.coefficients();
		boolean changed = false;
		for (int i = 0; i < c0.length; i++)
			if (c0[i] != c1[i])
				changed = true;
		assertThat(changed).isTrue();
	}

	@Test
	@DisplayName("rejects non-DoRA adapters")
	void rejects_lora_mode() {
		float[] w = new float[8];
		LoraAdapter lora = new LoraAdapter(2, 4, 2, 2f, new Random(1));
		assertThatThrownBy(() -> new DoraProjection(w, lora, new DoraMagnitude(2)))
				.isInstanceOf(IllegalArgumentException.class);
	}

	@Nested
	@DisplayName("Detached-norm finite differences")
	class FiniteDifferences {

		@Test
		@DisplayName("dL/d magnitude, A, B, X match FD")
		void grads_match_fd() {
			int in = 5, out = 4, rank = 2;
			float[] w = randomMatrix(out, in, 30);
			LoraAdapter lora = makeDora(rank, in, out, 4f, 31);
			DoraMagnitude mag = DoraMagnitude.fromValues(new float[] { 0.9f, 1.1f, 0.8f, 1.2f });
			DoraProjection dora = new DoraProjection(w, lora, mag);
			float[] x = randomVec(in, 32);
			float[] gradOut = randomVec(out, 33);

			dora.forward(x);
			lora.zeroGrad();
			mag.zeroGrad();
			float[] analyticX = dora.backward(gradOut, x);
			float[] analyticMag = mag.grad().clone();
			float[] analyticA = lora.gradA().clone();
			float[] analyticB = lora.gradB().clone();
			float[] frozenCoeff = dora.coefficients().clone();

			for (int i = 0; i < mag.length(); i++) {
				float orig = mag.values()[i];
				mag.values()[i] = orig + FD_H;
				dora.markDirty();
				float hi = dot(gradOut, dora.forward(x));
				mag.values()[i] = orig - FD_H;
				dora.markDirty();
				float lo = dot(gradOut, dora.forward(x));
				mag.values()[i] = orig;
				dora.markDirty();
				dora.refresh();
				assertClose("mag[" + i + "]", analyticMag[i], (hi - lo) / (2 * FD_H));
			}
			// A/B FD must freeze norms (detached): recompute only direction output.
			for (int i = 0; i < lora.a().length; i++) {
				float orig = lora.a()[i];
				lora.a()[i] = orig + FD_H;
				float hi = dot(gradOut, frozenForward(w, lora, frozenCoeff, x));
				lora.a()[i] = orig - FD_H;
				float lo = dot(gradOut, frozenForward(w, lora, frozenCoeff, x));
				lora.a()[i] = orig;
				assertClose("A[" + i + "]", analyticA[i], (hi - lo) / (2 * FD_H));
			}
			for (int i = 0; i < lora.b().length; i++) {
				float orig = lora.b()[i];
				lora.b()[i] = orig + FD_H;
				float hi = dot(gradOut, frozenForward(w, lora, frozenCoeff, x));
				lora.b()[i] = orig - FD_H;
				float lo = dot(gradOut, frozenForward(w, lora, frozenCoeff, x));
				lora.b()[i] = orig;
				assertClose("B[" + i + "]", analyticB[i], (hi - lo) / (2 * FD_H));
			}
			dora.markDirty();
			dora.refresh();
			for (int i = 0; i < x.length; i++) {
				float orig = x[i];
				x[i] = orig + FD_H;
				float hi = dot(gradOut, dora.forward(x));
				x[i] = orig - FD_H;
				float lo = dot(gradOut, dora.forward(x));
				x[i] = orig;
				assertClose("X[" + i + "]", analyticX[i], (hi - lo) / (2 * FD_H));
			}
		}

		/** Forward with frozen coefficients (detached-norm FD oracle for A/B). */
		private static float[] frozenForward(float[] w, LoraAdapter lora, float[] coeff, float[] x) {
			float[] y = matvec(w, lora.outDim, lora.inDim, x);
			float[] delta = lora.forward(x);
			for (int i = 0; i < y.length; i++)
				y[i] = coeff[i] * (y[i] + delta[i]);
			return y;
		}
	}

	private static LoraAdapter makeDora(int rank, int in, int out, float alpha, long seed) {
		LoraAdapter a = new LoraAdapter(
				LoraAdapterConfig.of(rank, alpha, LoraScaling.STANDARD, LoraInitialization.LEGACY_NORMAL, LoraMode.DORA),
				in, out, new Random(seed));
		Random r = new Random(seed + 1);
		for (int i = 0; i < a.b().length; i++)
			a.b()[i] = (float) (r.nextGaussian() * 0.05);
		return a;
	}

	private static void assertClose(String label, float analytic, float fd) {
		double absErr = Math.abs(analytic - fd);
		if (absErr < 5e-4f)
			return;
		float scale = Math.max(Math.abs(analytic), Math.abs(fd));
		if (scale < 1e-6f)
			return;
		assertThat(absErr / scale).as("%s analytic=%s fd=%s", label, analytic, fd).isLessThanOrEqualTo(TOL);
	}

	private static float[] matvec(float[] w, int out, int in, float[] x) {
		float[] y = new float[out];
		for (int r = 0; r < out; r++) {
			float acc = 0f;
			int base = r * in;
			for (int c = 0; c < in; c++)
				acc += w[base + c] * x[c];
			y[r] = acc;
		}
		return y;
	}

	private static float[] randomMatrix(int out, int in, long seed) {
		Random r = new Random(seed);
		float[] w = new float[out * in];
		for (int i = 0; i < w.length; i++)
			w[i] = (float) (r.nextGaussian() * 0.3);
		return w;
	}

	private static float[] randomVec(int n, long seed) {
		Random r = new Random(seed);
		float[] v = new float[n];
		for (int i = 0; i < n; i++)
			v[i] = (float) (r.nextGaussian() * 0.3);
		return v;
	}

	private static float dot(float[] a, float[] b) {
		float s = 0f;
		for (int i = 0; i < a.length; i++)
			s += a[i] * b[i];
		return s;
	}
}
