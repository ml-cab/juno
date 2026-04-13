/*
 * Copyright 2026 Dmytro Soloviov (soulaway)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package cab.ml.juno.node;

/**
 * Configurable activation codec — static switch between big-endian (BE) and
 * little-endian (LE) wire formats.
 *
 * Used exclusively at the gRPC transport boundary: compress before writing
 * {@code ForwardRequest.activation}, decompress after reading
 * {@code ForwardResponse.activation}.
 *
 * The rest of the pipeline (ForwardRequest record, ForwardResult record,
 * ForwardPassHandler) always works with float[] — byte order is purely a
 * network-wire concern and is invisible to business logic.
 *
 * Byte order is selected once at JVM startup via the system property
 * {@code juno.byteOrder}:
 * <pre>
 *   -Djuno.byteOrder=BE   big-endian    (default — validated on real hardware)
 *   -Djuno.byteOrder=LE   little-endian (native x86 order)
 * </pre>
 *
 * All encode/decode calls delegate to either {@link ActivationBECodec} or
 * {@link ActivationLECodec}. Both implementations are thread-safe and stateless.
 *
 * Thread-safe: all methods are stateless.
 */
public final class ActivationCodec {

	private ActivationCodec() {
	} // utility class — no instances

	// ── Byte-order selection (resolved once at class-load time) ───────────────

	/**
	 * {@code true} → big-endian (ActivationBECodec), {@code false} → little-endian
	 * (ActivationLECodec). Driven by {@code -Djuno.byteOrder=BE|LE}; defaults to
	 * {@code BE} which has been validated by hard testing on production hardware.
	 */
	private static final boolean USE_BE;

	static {
		String prop = System.getProperty("juno.byteOrder", "BE").trim().toUpperCase();
		USE_BE = !"LE".equals(prop); // anything other than explicit "LE" → BE
		System.out.println("[ActivationCodec] byteOrder=" + (USE_BE ? "BE" : "LE")
				+ "  (override with -Djuno.byteOrder=LE|BE)");
	}

	/**
	 * Returns the active byte-order label: {@code "BE"} or {@code "LE"}.
	 * Useful for logging and health-endpoint responses.
	 */
	public static String byteOrder() {
		return USE_BE ? "BE" : "LE";
	}

	// ── Public API ────────────────────────────────────────────────────────────

	/**
	 * Encode {@code floats} into bytes using the requested dtype.
	 * Byte order is determined by {@code juno.byteOrder} at startup.
	 *
	 * @param floats source activation tensor (may be empty, must not be null)
	 * @param dtype  target wire format
	 * @return compressed byte representation; length depends on dtype
	 */
	public static byte[] encode(float[] floats, ActivationDtype dtype) {
		return USE_BE
				? ActivationBECodec.encode(floats, dtype)
				: ActivationLECodec.encode(floats, dtype);
	}

	/**
	 * Decode {@code bytes} back to a float[]. Must be called with the same dtype
	 * and byte-order that was used for encoding.
	 *
	 * @param bytes encoded bytes (from {@link #encode}); null or empty returns
	 *              float[0]
	 * @param dtype the format used during encoding
	 * @return reconstructed float array
	 */
	public static float[] decode(byte[] bytes, ActivationDtype dtype) {
		return USE_BE
				? ActivationBECodec.decode(bytes, dtype)
				: ActivationLECodec.decode(bytes, dtype);
	}

	// ── F32 raw bytes matVec ──────────────────────────────────────────────────

	/**
	 * Matrix-vector multiply directly on raw float32 bytes.
	 * Delegates to the active byte-order implementation.
	 */
	public static float[] matVecF32raw(byte[] raw, float[] x, int rowStart, int rowEnd, int cols) {
		return USE_BE
				? ActivationBECodec.matVecF32raw(raw, x, rowStart, rowEnd, cols)
				: ActivationLECodec.matVecF32raw(raw, x, rowStart, rowEnd, cols);
	}

	// ── Package-private helpers (forwarded to active impl, used by tests) ─────

	/**
	 * Convert IEEE 754 single-precision float to half-precision (16-bit).
	 * Delegates to the active byte-order implementation.
	 * Package-private for unit testing.
	 */
	static short floatToHalf(float f) {
		return USE_BE ? ActivationBECodec.floatToHalf(f) : ActivationLECodec.floatToHalf(f);
	}

	/**
	 * Convert IEEE 754 half-precision (16-bit) to single-precision float.
	 * Delegates to the active byte-order implementation.
	 * Package-private for unit testing.
	 */
	static float halfToFloat(short half) {
		return USE_BE ? ActivationBECodec.halfToFloat(half) : ActivationLECodec.halfToFloat(half);
	}
}