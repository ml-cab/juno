package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Test;

class Phi2RopeTest {

	@Test
	void pos0IsIdentity() {
		// At position 0, angle = 0 for every dim, so cos=1, sin=0 and the
		// vector must be returned unchanged.
		float[] x = { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f };
		float[] expected = x.clone();

		Phi2Rope.ropePartial(x, 0, 1, 8, 8, 10000f);

		assertThat(x).containsExactly(expected);
	}

	@Test
	void dimensionsBeyondRopeDimAreUntouched() {
		// headDim=8, ropeDim=4 (partial rope, matching phi2.rope.dimension_count
		// being half of headDim in the real model). Dims [4..7] must pass through
		// unchanged for any position.
		float[] x = { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f };

		Phi2Rope.ropePartial(x, 5, 1, 8, 4, 10000f);

		assertThat(x[4]).isEqualTo(5f);
		assertThat(x[5]).isEqualTo(6f);
		assertThat(x[6]).isEqualTo(7f);
		assertThat(x[7]).isEqualTo(8f);
	}

	@Test
	void rotationPreservesNormOfRotatedSubvector() {
		// Rotation is an orthogonal transform: the L2 norm of the rotated
		// ropeDim-wide slice must be preserved for any position.
		float[] x = { 0.3f, -1.7f, 2.2f, 0.05f, -0.9f, 4.1f };
		double normBefore = l2(x, 0, 6);

		Phi2Rope.ropePartial(x, 37, 1, 6, 6, 10000f);

		double normAfter = l2(x, 0, 6);
		assertThat(normAfter).isCloseTo(normBefore, within(1e-4));
	}

	@Test
	void usesSplitHalfPairingNotAdjacentPairing() {
		// headDim=4, ropeDim=4, nHeads=1, pos=1, theta=10000.
		// half = ropeDim/2 = 2, so NeoX pairs are (x[0],x[2]) and (x[1],x[3]).
		// freq(i=0) = theta^0 = 1        -> angle = 1
		// freq(i=1) = theta^(-2/4)=1/100 -> angle = 0.01
		float[] x = { 1f, 0f, 0f, 1f };

		Phi2Rope.ropePartial(x, 1, 1, 4, 4, 10000f);

		double cos1 = Math.cos(1.0);
		double sin1 = Math.sin(1.0);
		double cos001 = Math.cos(0.01);
		double sin001 = Math.sin(0.01);
		// pair (x0=x[0]=1, x1=x[2]=0): x[0]=cos1, x[2]=sin1
		assertThat(x[0]).isCloseTo((float) cos1, within(1e-5f));
		assertThat(x[2]).isCloseTo((float) sin1, within(1e-5f));
		// pair (x0=x[1]=0, x1=x[3]=1): x[1]=-sin(0.01), x[3]=cos(0.01)
		assertThat(x[1]).isCloseTo((float) -sin001, within(1e-5f));
		assertThat(x[3]).isCloseTo((float) cos001, within(1e-5f));
	}

	@Test
	void multipleHeadsAreIndependent() {
		float[] x = { 1f, 0f, 0f, 1f,   1f, 0f, 0f, 1f };
		Phi2Rope.ropePartial(x, 1, 2, 4, 4, 10000f);
		// Both heads must rotate identically since inputs are identical.
		assertThat(x[0]).isEqualTo(x[4]);
		assertThat(x[1]).isEqualTo(x[5]);
		assertThat(x[2]).isEqualTo(x[6]);
		assertThat(x[3]).isEqualTo(x[7]);
	}

	private static double l2(float[] v, int off, int len) {
		double sq = 0;
		for (int i = off; i < off + len; i++) sq += (double) v[i] * v[i];
		return Math.sqrt(sq);
	}
}