package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link VectorQuantKernels#dot}.
 *
 * <p>
 * These exercise the SIMD dot-product entry point directly (rather than only
 * indirectly through the {@code sgemm*WeightStationary} kernels covered by
 * {@link PhiQuantizedMatVecTest}), covering block sizes actually used by the
 * kernels (Q8_0: 32, Q4_K/Q5_K: 256) plus lengths that are not a multiple of
 * common SIMD lane widths, to prove the scalar tail loop is correct
 * regardless of {@link VectorQuantKernels#AVAILABLE}.
 */
@DisplayName("VectorQuantKernels.dot")
class VectorQuantKernelsTest {

	private static float dotReference(float[] dq, int dqOffset, float[] xp, int xOffset, int len) {
		float acc = 0f;
		for (int i = 0; i < len; i++) {
			acc += dq[dqOffset + i] * xp[xOffset + i];
		}
		return acc;
	}

	@Test
	@DisplayName("Vector API availability probe does not throw and is a stable value")
	void available_isStableAndDoesNotThrow() {
		// AVAILABLE is computed once at class-init; just prove it is readable
		// and consistent within a single JVM run. True or false are both
		// valid depending on whether --add-modules jdk.incubator.vector was
		// passed to this JVM, because the kernel must be correct either way.
		boolean first = VectorQuantKernels.AVAILABLE;
		boolean second = VectorQuantKernels.AVAILABLE;
		assertThat(first).isEqualTo(second);
	}

	@Test
	@DisplayName("Q8_0 block size (32): matches scalar reference")
	void dot_blockSize32_matchesScalar() {
		java.util.Random rnd = new java.util.Random(1);
		float[] dq = randomArray(32, rnd);
		float[] xp = randomArray(32, rnd);

		float actual = VectorQuantKernels.dot(dq, 0, xp, 0, 32);
		float expected = dotReference(dq, 0, xp, 0, 32);

		assertThat(actual).isCloseTo(expected, within(1e-3f));
	}

	@Test
	@DisplayName("Q4_K/Q5_K block size (256): matches scalar reference")
	void dot_blockSize256_matchesScalar() {
		java.util.Random rnd = new java.util.Random(2);
		float[] dq = randomArray(256, rnd);
		float[] xp = randomArray(256, rnd);

		float actual = VectorQuantKernels.dot(dq, 0, xp, 0, 256);
		float expected = dotReference(dq, 0, xp, 0, 256);

		assertThat(actual).isCloseTo(expected, within(1e-2f));
	}

	@Test
	@DisplayName("Non-lane-aligned lengths exercise the scalar tail loop correctly")
	void dot_oddLengths_matchScalar() {
		java.util.Random rnd = new java.util.Random(3);
		for (int len : new int[] { 1, 3, 5, 7, 9, 13, 17, 31, 33, 63, 65 }) {
			float[] dq = randomArray(len, rnd);
			float[] xp = randomArray(len, rnd);

			float actual = VectorQuantKernels.dot(dq, 0, xp, 0, len);
			float expected = dotReference(dq, 0, xp, 0, len);

			assertThat(actual).as("len=%d", len).isCloseTo(expected, within(1e-3f));
		}
	}

	@Test
	@DisplayName("Non-zero offsets into larger backing arrays are respected")
	void dot_withOffsets_readsCorrectSlice() {
		java.util.Random rnd = new java.util.Random(4);
		float[] dq = randomArray(64, rnd); // dequant scratch is never offset in practice, but the API allows it
		float[] xp = randomArray(1024, rnd); // simulates a batched input row, offset by block index
		int dqOffset = 0;
		int xOffset = 256; // e.g. the 8th 32-wide Q8_0 block within one input row

		float actual = VectorQuantKernels.dot(dq, dqOffset, xp, xOffset, 32);
		float expected = dotReference(dq, dqOffset, xp, xOffset, 32);

		assertThat(actual).isCloseTo(expected, within(1e-3f));
	}

	@Test
	@DisplayName("Zero length returns zero without throwing")
	void dot_zeroLength_returnsZero() {
		float[] dq = new float[] { 1f, 2f, 3f };
		float[] xp = new float[] { 1f, 2f, 3f };
		assertThat(VectorQuantKernels.dot(dq, 0, xp, 0, 0)).isZero();
	}

	private static float[] randomArray(int len, java.util.Random rnd) {
		float[] a = new float[len];
		for (int i = 0; i < len; i++) {
			a[i] = (rnd.nextFloat() - 0.5f) * 2f;
		}
		return a;
	}
}