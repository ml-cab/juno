package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link VectorQuantKernels}: the SIMD dot-product
 * accumulation entry point ({@link VectorQuantKernels#dot}) and the
 * vectorized Q8_0 dequantization entry point
 * ({@link VectorQuantKernels#dequantizeQ8_0}).
 *
 * <p>
 * These exercise both entry points directly (rather than only indirectly
 * through the {@code sgemm*WeightStationary} kernels covered by
 * {@link PhiQuantizedMatVecTest}), covering block sizes actually used by the
 * kernels (Q8_0: 32, Q4_K/Q5_K: 256), lengths that are not a multiple of
 * common SIMD lane widths (for {@code dot}'s scalar tail loop), and the full
 * signed-byte range (for the Q8_0 dequant widen-and-scale conversion).
 */
@DisplayName("VectorQuantKernels")
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

	@Test
	@DisplayName("Q8_0 dequant availability probe does not throw and is a stable value")
	void q8_0DequantAvailable_isStableAndDoesNotThrow() {
		// Same reasoning as available_isStableAndDoesNotThrow above: either
		// value is valid, this just proves the self-check ran and is stable.
		boolean first = VectorQuantKernels.Q8_0_DEQUANT_AVAILABLE;
		boolean second = VectorQuantKernels.Q8_0_DEQUANT_AVAILABLE;
		assertThat(first).isEqualTo(second);
		// If the SIMD path claims availability, it can only be because its
		// own internal self-test already passed (see probeQ8_0Dequant());
		// AVAILABLE is therefore implied.
		if (first) {
			assertThat(VectorQuantKernels.AVAILABLE).isTrue();
		}
	}

	@Test
	@DisplayName("Q8_0 dequant: full signed-byte range matches scalar reference, when SIMD path is available")
	void dequantizeQ8_0_fullByteRange_matchesScalar() {
		org.junit.jupiter.api.Assumptions.assumeTrue(VectorQuantKernels.Q8_0_DEQUANT_AVAILABLE,
				"SIMD Q8_0 dequant not available on this JVM; scalar-fallback behavior is covered separately");

		byte[] raw = new byte[34];
		for (int i = 0; i < 32; i++) {
			raw[2 + i] = (byte) (i * 8 - 128); // sweeps the full signed-byte range, including -128
		}
		float scale = 0.0625f;

		float[] dq = new float[32];
		boolean handled = VectorQuantKernels.dequantizeQ8_0(raw, 2, scale, dq);
		assertThat(handled).isTrue();

		for (int i = 0; i < 32; i++) {
			float expected = scale * raw[2 + i];
			assertThat(dq[i]).as("lane=%d", i).isCloseTo(expected, within(1e-6f));
		}
	}

	@Test
	@DisplayName("Q8_0 dequant output feeds correctly into dot(), matching an end-to-end scalar reference")
	void dequantizeQ8_0_thenDot_matchesEndToEndScalar() {
		org.junit.jupiter.api.Assumptions.assumeTrue(VectorQuantKernels.Q8_0_DEQUANT_AVAILABLE,
				"SIMD Q8_0 dequant not available on this JVM; scalar-fallback behavior is covered separately");

		java.util.Random rnd = new java.util.Random(5);
		byte[] raw = new byte[34];
		for (int i = 0; i < 32; i++) {
			raw[2 + i] = (byte) (rnd.nextInt(256) - 128);
		}
		float scale = 0.017578125f; // 9/512, arbitrary non-trivial scale
		float[] xp = randomArray(32, rnd);

		float[] dqSimd = new float[32];
		assertThat(VectorQuantKernels.dequantizeQ8_0(raw, 2, scale, dqSimd)).isTrue();
		float actual = VectorQuantKernels.dot(dqSimd, 0, xp, 0, 32);

		float[] dqScalar = new float[32];
		for (int i = 0; i < 32; i++) {
			dqScalar[i] = scale * raw[2 + i];
		}
		float expected = dotReference(dqScalar, 0, xp, 0, 32);

		assertThat(actual).isCloseTo(expected, within(1e-3f));
	}

	@Test
	@DisplayName("Q8_0 dequant reports unavailable cleanly when the SIMD path is not usable")
	void dequantizeQ8_0_whenUnavailable_returnsFalseAndDoesNotThrow() {
		org.junit.jupiter.api.Assumptions.assumeFalse(VectorQuantKernels.Q8_0_DEQUANT_AVAILABLE,
				"SIMD Q8_0 dequant is available on this JVM; the available-path behavior is covered separately");

		byte[] raw = new byte[34];
		float[] dq = new float[32];
		boolean handled = VectorQuantKernels.dequantizeQ8_0(raw, 2, 1f, dq);

		assertThat(handled).isFalse();
	}

	private static float[] randomArray(int len, java.util.Random rnd) {
		float[] a = new float[len];
		for (int i = 0; i < len; i++) {
			a[i] = (rnd.nextFloat() - 0.5f) * 2f;
		}
		return a;
	}
}