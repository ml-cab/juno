package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.util.Arrays;
import java.util.Random;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import cab.ml.juno.node.ActivationCodec;
import cab.ml.juno.node.ActivationDtype;

class ActivationCodecTest {

	/**
	 * Gaussian activations matching a typical post-LayerNorm distribution (σ ≈
	 * 0.5).
	 */
	private static final float[] TYPICAL;

	static {
		Random rng = new Random(42);
		TYPICAL = new float[2048];
		for (int i = 0; i < TYPICAL.length; i++) {
			TYPICAL[i] = (float) (rng.nextGaussian() * 0.5);
		}
	}

	// ── FLOAT32 ───────────────────────────────────────────────────────────────

	@Test
	void float32_roundtrip_is_bitwise_lossless() {
		byte[] encoded = ActivationCodec.encode(TYPICAL, ActivationDtype.FLOAT32);
		float[] decoded = ActivationCodec.decode(encoded, ActivationDtype.FLOAT32);

		assertThat(decoded).containsExactly(TYPICAL);
	}

	@Test
	void float32_encoded_size_is_4_bytes_per_element() {
		float[] floats = new float[512];
		assertThat(ActivationCodec.encode(floats, ActivationDtype.FLOAT32)).hasSize(512 * 4);
	}

	// ── FLOAT16 ───────────────────────────────────────────────────────────────

	@Test
	void float16_roundtrip_has_bounded_error_for_typical_activations() {
		byte[] encoded = ActivationCodec.encode(TYPICAL, ActivationDtype.FLOAT16);
		float[] decoded = ActivationCodec.decode(encoded, ActivationDtype.FLOAT16);

		assertThat(decoded).hasSize(TYPICAL.length);

		for (int i = 0; i < TYPICAL.length; i++) {
			assertThat(Math.abs(decoded[i] - TYPICAL[i]))
					.as("element %d: orig=%.6f decoded=%.6f", i, TYPICAL[i], decoded[i]).isLessThanOrEqualTo(0.002f); // FP16
																														// ≈
																														// 3
																														// decimal
																														// digits
																														// of
																														// precision
		}
	}

	@Test
	void float16_encoded_size_is_exactly_half_of_float32() {
		float[] floats = new float[2048];
		int f32 = ActivationCodec.encode(floats, ActivationDtype.FLOAT32).length;
		int f16 = ActivationCodec.encode(floats, ActivationDtype.FLOAT16).length;

		assertThat(f16).isEqualTo(f32 / 2);
	}

	@Test
	void float16_handles_zero_array() {
		float[] zeros = new float[256];
		float[] decoded = ActivationCodec.decode(ActivationCodec.encode(zeros, ActivationDtype.FLOAT16),
				ActivationDtype.FLOAT16);

		assertThat(decoded).containsOnly(0.0f);
	}

	@Test
	void float16_handles_positive_and_negative_values() {
		float[] floats = { 1.0f, -1.0f, 0.5f, -0.5f, 0.125f, -0.125f };
		float[] decoded = ActivationCodec.decode(ActivationCodec.encode(floats, ActivationDtype.FLOAT16),
				ActivationDtype.FLOAT16);

		for (int i = 0; i < floats.length; i++) {
			assertThat(decoded[i]).isCloseTo(floats[i], within(0.001f));
		}
	}

	@Test
	void float16_overflow_becomes_infinity() {
		// float16 max is 65504; values beyond saturate to ±∞
		float huge = 1e8f;
		short half = ActivationCodec.floatToHalf(huge);
		float back = ActivationCodec.halfToFloat(half);

		assertThat(Float.isInfinite(back)).isTrue();
	}

	@Test
	void float16_preserves_zero_and_negative_zero() {
		assertThat(ActivationCodec.halfToFloat(ActivationCodec.floatToHalf(0.0f))).isEqualTo(0.0f);
		// negative-zero: both decode to ±0, bit pattern differs but value equals 0
		float negZero = ActivationCodec.halfToFloat(ActivationCodec.floatToHalf(-0.0f));
		assertThat(negZero == 0.0f || negZero == -0.0f).isTrue();
	}

	@Test
	void float16_small_values_underflow_to_zero_gracefully() {
		float tiny = 1e-8f; // well below FP16 min normal (6.1e-5)
		short half = ActivationCodec.floatToHalf(tiny);
		float back = ActivationCodec.halfToFloat(half);

		// Must not throw; value becomes 0 (underflow) — acceptable
		assertThat(back).isGreaterThanOrEqualTo(0.0f);
		assertThat(back).isLessThanOrEqualTo(tiny * 10);
	}

	// ── INT8 ──────────────────────────────────────────────────────────────────

	@Test
	void int8_roundtrip_has_bounded_error_for_typical_activations() {
		byte[] encoded = ActivationCodec.encode(TYPICAL, ActivationDtype.INT8);
		float[] decoded = ActivationCodec.decode(encoded, ActivationDtype.INT8);

		assertThat(decoded).hasSize(TYPICAL.length);

		// Allowed absolute error = 1.5 × quantisation step (scale / 127 × 127 = scale)
		float maxAbs = 0f;
		for (float f : TYPICAL)
			maxAbs = Math.max(maxAbs, Math.abs(f));
		float maxAllowedError = (maxAbs / 127f) * 1.5f;

		for (int i = 0; i < TYPICAL.length; i++) {
			assertThat(Math.abs(decoded[i] - TYPICAL[i]))
					.as("element %d: orig=%.6f decoded=%.6f", i, TYPICAL[i], decoded[i])
					.isLessThanOrEqualTo(maxAllowedError);
		}
	}

	@Test
	void int8_encoded_size_is_4_byte_scale_plus_1_byte_per_element() {
		float[] floats = new float[512];
		assertThat(ActivationCodec.encode(floats, ActivationDtype.INT8)).hasSize(4 + 512);
	}

	@Test
	void int8_gives_approximately_4x_size_reduction_vs_float32() {
		float[] floats = new float[1024];
		int f32 = ActivationCodec.encode(floats, ActivationDtype.FLOAT32).length; // 4096
		int i8 = ActivationCodec.encode(floats, ActivationDtype.INT8).length; // 1028

		assertThat((double) f32 / i8).isGreaterThan(3.9);
	}

	@Test
	void int8_handles_all_zero_array_without_divide_by_zero() {
		float[] zeros = new float[256];
		float[] decoded = ActivationCodec.decode(ActivationCodec.encode(zeros, ActivationDtype.INT8),
				ActivationDtype.INT8);

		assertThat(decoded).containsOnly(0.0f);
	}

	@Test
	void int8_preserves_sign_of_each_element() {
		float[] floats = { 1.0f, -1.0f, 2.0f, -2.0f, 0.5f, -0.5f };
		float[] decoded = ActivationCodec.decode(ActivationCodec.encode(floats, ActivationDtype.INT8),
				ActivationDtype.INT8);

		for (int i = 0; i < floats.length; i++) {
			assertThat(Math.signum(decoded[i])).as("sign of element %d", i).isEqualTo(Math.signum(floats[i]));
		}
	}

	// ── Empty / edge cases ────────────────────────────────────────────────────

	@Test
	void decode_null_returns_empty_array_for_all_dtypes() {
		assertThat(ActivationCodec.decode(null, ActivationDtype.FLOAT32)).isEmpty();
		assertThat(ActivationCodec.decode(null, ActivationDtype.FLOAT16)).isEmpty();
		assertThat(ActivationCodec.decode(null, ActivationDtype.INT8)).isEmpty();
	}

	@Test
	void decode_empty_bytes_returns_empty_array_for_all_dtypes() {
		byte[] empty = new byte[0];
		assertThat(ActivationCodec.decode(empty, ActivationDtype.FLOAT32)).isEmpty();
		assertThat(ActivationCodec.decode(empty, ActivationDtype.FLOAT16)).isEmpty();
		assertThat(ActivationCodec.decode(empty, ActivationDtype.INT8)).isEmpty();
	}

	@ParameterizedTest(name = "single-element roundtrip [{0}]")
	@EnumSource(ActivationDtype.class)
	void single_element_roundtrip(ActivationDtype dtype) {
		float[] single = { 42.0f };
		float[] decoded = ActivationCodec.decode(ActivationCodec.encode(single, dtype), dtype);

		assertThat(decoded).hasSize(1);
		// INT8 has higher relative error for large values; allow up to 1% of value
		float tolerance = dtype == ActivationDtype.INT8 ? 42.0f * 0.01f : 0.02f;
		assertThat(decoded[0]).isCloseTo(42.0f, within(tolerance));
	}

	// ── Compression ratio sanity ──────────────────────────────────────────────

	@Test
	void compression_ratios_match_expected_sizes() {
		int n = 8192; // typical hidden_dim
		float[] floats = new float[n];
		Arrays.fill(floats, 0.5f);

		int f32 = ActivationCodec.encode(floats, ActivationDtype.FLOAT32).length;
		int f16 = ActivationCodec.encode(floats, ActivationDtype.FLOAT16).length;
		int i8 = ActivationCodec.encode(floats, ActivationDtype.INT8).length;

		assertThat(f32).isEqualTo(n * 4); // 32768 bytes
		assertThat(f16).isEqualTo(n * 2); // 16384 bytes — 2× saving
		assertThat(i8).isEqualTo(4 + n); // 8196 bytes — ≈4× saving
	}
}