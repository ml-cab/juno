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

import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * SIMD entry point for the weight-stationary CPU quantized matmul kernels
 * ({@link LlamaTransformerHandler#sgemmQ4KWeightStationary},
 * {@link LlamaTransformerHandler#sgemmQ5KWeightStationary},
 * {@link LlamaTransformerHandler#sgemmQ8_0WeightStationary}).
 *
 * <p>
 * Scope: this class vectorizes only the dot-product accumulation phase:
 * {@code sum(dq[i] * xp[xBase+i])} over a dequantized block. This is the
 * naturally SIMD-friendly half of each kernel (contiguous multiply-add, no
 * bit manipulation). It is also the dominant cost: for a batch of B input
 * rows, the accumulation phase does {@code BLOCK_SIZE * B} multiply-adds per
 * block, versus {@code BLOCK_SIZE} for dequantization, done once per block
 * regardless of B. For the batch sizes this project actually runs
 * (B in the hundreds during prefill), accumulation dominates by orders of
 * magnitude.
 *
 * <p>
 * Deliberately out of scope: vectorizing the dequantization phase itself
 * (nibble/bit unpacking for Q4_K/Q5_K, or the int8-to-float widen for
 * Q8_0). The Vector API's byte/int/float shape-conversion lanes need care
 * that is easy to get subtly wrong in code that cannot be exercised against
 * real hardware before review, and silent numeric corruption in a
 * quantized-weight kernel is exactly the failure mode this codebase's own
 * standing rule (unit tests first, for bit-manipulation-heavy code) exists
 * to guard against. Left as a follow-up once this phase is validated on
 * real hardware. See {@code juno-documentation}'s performance notes.
 *
 * <p>
 * Dequantization is now partially in scope, as a follow-up to the
 * accumulate-only version of this class: {@link #dequantizeQ8_0} vectorizes
 * the Q8_0 int8-to-float widen-and-scale step, the simplest of the three
 * quant formats (a flat scale plus 32 raw signed bytes, no nibble or
 * high-bit-plane unpacking). Q4_K/Q5_K dequantization remains scalar; their
 * irregular bit-shift/mask patterns are a different, harder problem and are
 * left for a later pass once this one is validated on real hardware.
 *
 * <p>
 * Because the byte-to-int-to-float shape conversion this uses is a less
 * common corner of the Vector API than the plain same-shape arithmetic in
 * {@link #dot}, and because this code cannot be exercised on real hardware
 * before review, {@link #dequantizeQ8_0} is gated by its own dedicated probe
 * ({@code Simd.probeDequantQ8_0()}) that dequantizes a known 32-byte pattern
 * covering the full signed-byte range and checks every output lane against
 * the scalar-computed expected value. The SIMD dequant path is only enabled
 * if that self-check passes, so every call transparently reports "not
 * available" otherwise and the caller falls back to its existing scalar
 * dequant loop. This is a stronger runtime guarantee than {@link #AVAILABLE}
 * alone, because that flag only proves the module loaded, not that this
 * specific conversion is correct on this JVM/hardware combination, and it
 * exists specifically because this path could not be unit-tested against
 * real hardware ahead of time.
 *
 * <p>
 * Runtime availability: {@code jdk.incubator.vector} is still an incubating
 * JDK module as of JDK 25/26 (JEP 508 / JEP 529), because it requires
 * {@code --add-modules jdk.incubator.vector} at both compile and run time,
 * and is not guaranteed present on every deployment target (e.g. a JVM
 * launched without that flag, or a future JDK that removes the module before
 * finalizing it). All references to {@code jdk.incubator.vector} types live
 * in the nested {@link Simd} class so the class-loading failure that would
 * result from a missing module is confined to the first attempt to load
 * {@link Simd}: caught once, here, at class-init time, rather than
 * failing verification of this outer class or any of its callers.
 */
public final class VectorQuantKernels {

	private static final Logger log = Logger.getLogger(VectorQuantKernels.class.getName());

	/**
	 * True if {@code jdk.incubator.vector} loaded and a trivial vector op
	 * executed successfully on this JVM. Checked once at class-init; callers
	 * do not need to re-check it themselves. {@link #dot} always does the
	 * correct thing (vectorized or scalar) regardless of this flag's value.
	 */
	static final boolean AVAILABLE = probe();

	/**
	 * True if {@link #AVAILABLE} and, in addition, the vectorized Q8_0
	 * dequantization path ({@code Simd.dequantizeQ8_0Block}) produced correct
	 * output against a known test pattern covering the full signed-byte
	 * range. See the class javadoc for why this gets a separate, stronger
	 * self-check than {@link #AVAILABLE}.
	 */
	static final boolean Q8_0_DEQUANT_AVAILABLE = probeQ8_0Dequant();

	private VectorQuantKernels() {
	}

	private static boolean probe() {
		try {
			float sum = Simd.probe();
			return sum == Simd.PROBE_EXPECTED;
		} catch (Throwable t) {
			// Covers NoClassDefFoundError (module not added with --add-modules),
			// UnsupportedClassVersionError, and any other linkage failure on
			// exotic targets. Falls back to the scalar path below: same
			// numerical result, just without the speedup.
			log.log(Level.INFO, "jdk.incubator.vector unavailable, using scalar quantized-matmul kernels: "
					+ t.getClass().getSimpleName() + (t.getMessage() != null ? ": " + t.getMessage() : ""));
			return false;
		}
	}

	private static boolean probeQ8_0Dequant() {
		if (!AVAILABLE) {
			return false;
		}
		try {
			return Simd.probeDequantQ8_0();
		} catch (Throwable t) {
			log.log(Level.INFO, "SIMD Q8_0 dequantization unavailable, using scalar dequant: "
					+ t.getClass().getSimpleName() + (t.getMessage() != null ? ": " + t.getMessage() : ""));
			return false;
		}
	}

	/**
	 * {@code sum(dq[dqOffset..dqOffset+len) * xp[xOffset..xOffset+len))}.
	 *
	 * <p>
	 * Used by every {@code sgemm*WeightStationary} kernel's inner loop, once
	 * per (block, batch-row) pair, with {@code dq} the block just
	 * dequantized (256 elements for Q4_K/Q5_K, 32 for Q8_0) and {@code xp}
	 * one row of the batched input matrix.
	 *
	 * <p>
	 * Reduction order differs from the plain scalar loop once vectorized
	 * (SIMD-lane-width partial sums, combined at the end), so results are not
	 * bit-exact against the scalar reference. This is the same as every
	 * other batching change in this kernel family. Compare with
	 * relative+absolute tolerance, not exact equality.
	 */
	static float dot(float[] dq, int dqOffset, float[] xp, int xOffset, int len) {
		if (AVAILABLE) {
			try {
				return Simd.dot(dq, dqOffset, xp, xOffset, len);
			} catch (Throwable t) {
				// Should not happen once AVAILABLE is true (the probe already
				// exercised the same code path), but never let a kernel crash
				// a running inference request over an unexpected SIMD failure.
				log.log(Level.WARNING, "Vector API dot-product failed after successful probe, "
						+ "falling back to scalar for this call", t);
			}
		}
		return dotScalar(dq, dqOffset, xp, xOffset, len);
	}

	private static float dotScalar(float[] dq, int dqOffset, float[] xp, int xOffset, int len) {
		float acc = 0f;
		for (int i = 0; i < len; i++) {
			acc += dq[dqOffset + i] * xp[xOffset + i];
		}
		return acc;
	}

	/**
	 * Vectorized Q8_0 dequantization: fills {@code dq[0..32)} with
	 * {@code scale * raw[byteOffset+i]} for a 32-byte Q8_0 block.
	 *
	 * <p>
	 * Returns {@code false} (writing nothing to {@code dq}) if the SIMD path
	 * is not available or self-verified on this JVM. See
	 * {@link #Q8_0_DEQUANT_AVAILABLE}. Callers must fall back to their own
	 * scalar dequant loop when this returns {@code false}:
	 *
	 * <pre>{@code
	 * if (!VectorQuantKernels.dequantizeQ8_0(raw, bo + 2, sc, dq)) {
	 *     for (int i = 0; i < 32; i++) dq[i] = sc * raw[bo + 2 + i];
	 * }
	 * }</pre>
	 */
	static boolean dequantizeQ8_0(byte[] raw, int byteOffset, float scale, float[] dq) {
		if (Q8_0_DEQUANT_AVAILABLE) {
			try {
				Simd.dequantizeQ8_0Block(raw, byteOffset, scale, dq);
				return true;
			} catch (Throwable t) {
				// Should not happen once Q8_0_DEQUANT_AVAILABLE is true (the
				// probe already exercised this exact code path), but never
				// let a kernel crash a running request over an unexpected
				// SIMD failure.
				log.log(Level.WARNING, "Vector API Q8_0 dequantization failed after successful probe, "
						+ "falling back to scalar for this call", t);
			}
		}
		return false;
	}

	/**
	 * One-line human-readable summary of the actual SIMD width in use, for a
	 * single startup log line. Exists because "AVAILABLE=true" alone does not
	 * say whether the JVM picked a narrow or wide vector shape on this CPU,
	 * and that width materially affects how much speedup to expect. Safe to
	 * call regardless of {@link #AVAILABLE}: never throws, isolates any
	 * {@code jdk.incubator.vector} failure the same way {@link #dot} does.
	 */
	public static String diagnosticSummary() {
		if (!AVAILABLE) {
			return "SIMD unavailable: jdk.incubator.vector did not load "
					+ "(missing --add-modules jdk.incubator.vector, or an unsupported target). "
					+ "Quantized-matmul kernels are running the scalar fallback.";
		}
		try {
			return Simd.diagnosticSummary();
		} catch (Throwable t) {
			return "SIMD available=true but reading the diagnostic failed: "
					+ t.getClass().getSimpleName() + ". Kernels still run correctly via the scalar fallback.";
		}
	}

	/**
	 * All {@code jdk.incubator.vector} references are confined to this nested
	 * class. It is only class-loaded the first time {@link #probe()} or
	 * {@link #dot} reaches into it, so a JVM without the module installed (or
	 * started without {@code --add-modules jdk.incubator.vector}) never pays
	 * a verification cost for code it cannot run, and {@link VectorQuantKernels}
	 * itself always loads cleanly.
	 */
	private static final class Simd {

		private static final jdk.incubator.vector.VectorSpecies<Float> SPECIES =
				jdk.incubator.vector.FloatVector.SPECIES_PREFERRED;

		private static final float PROBE_EXPECTED = 3f;

		/** Trivial vector op used only to prove the module actually works. */
		static float probe() {
			jdk.incubator.vector.FloatVector v = jdk.incubator.vector.FloatVector.broadcast(SPECIES, 1f);
			v = v.add(jdk.incubator.vector.FloatVector.broadcast(SPECIES, 2f));
			return v.lane(0);
		}

		static String diagnosticSummary() {
			int lanes = SPECIES.length();
			int bits = SPECIES.vectorShape().vectorBitSize();
			// 256-bit = AVX2 (8 float lanes), 512-bit = AVX-512 (16 lanes),
			// 128-bit = SSE/NEON (4 lanes). On hybrid Intel P/E-core parts
			// (e.g. Alder Lake and later), AVX-512 is fused off even though
			// the P-cores support it in silicon, so 256-bit here is the
			// expected/correct result on that class of CPU, not a fallback.
			return "SIMD available: FloatVector.SPECIES_PREFERRED=" + bits + "-bit (" + lanes
					+ " float lanes per instruction)";
		}

		static float dot(float[] dq, int dqOffset, float[] xp, int xOffset, int len) {
			int lanes = SPECIES.length();
			int upper = SPECIES.loopBound(len);

			jdk.incubator.vector.FloatVector acc = jdk.incubator.vector.FloatVector.zero(SPECIES);
			int i = 0;
			for (; i < upper; i += lanes) {
				jdk.incubator.vector.FloatVector va =
						jdk.incubator.vector.FloatVector.fromArray(SPECIES, dq, dqOffset + i);
				jdk.incubator.vector.FloatVector vb =
						jdk.incubator.vector.FloatVector.fromArray(SPECIES, xp, xOffset + i);
				acc = va.fma(vb, acc);
			}
			float sum = acc.reduceLanes(jdk.incubator.vector.VectorOperators.ADD);

			// Scalar tail for any remainder below one full lane width. The
			// batching structure keeps len at 32 (Q8_0) or 256 (Q4_K/Q5_K),
			// which is a multiple of every realistic SPECIES length (4/8/16),
			// so in practice this loop rarely executes. Kept for
			// correctness on any vector width.
			for (; i < len; i++) {
				sum += dq[dqOffset + i] * xp[xOffset + i];
			}
			return sum;
		}

		// Q8_0 dequantization.
		// Fixed 8-lane species across byte/int/float rather than deriving
		// from SPECIES_PREFERRED: 64-bit byte / 256-bit int / 256-bit float
		// all have exactly 8 lanes, which keeps the widening conversion a
		// clean 1:1 lane mapping (no fan-out across multiple source vectors)
		// and evenly tiles a 32-element Q8_0 block in exactly 4 iterations
		// with no remainder. Fixed-shape species are always usable per the
		// Vector API spec even on hardware narrower than 256 bits (the JVM
		// falls back to a correct, if not fully accelerated, implementation),
		// so this does not need its own hardware-width fallback the way
		// SPECIES_PREFERRED-based code would.
		private static final jdk.incubator.vector.VectorSpecies<Byte> Q8_0_BYTE_SPECIES =
				jdk.incubator.vector.ByteVector.SPECIES_64;
		private static final jdk.incubator.vector.VectorSpecies<Integer> Q8_0_INT_SPECIES =
				jdk.incubator.vector.IntVector.SPECIES_256;
		private static final jdk.incubator.vector.VectorSpecies<Float> Q8_0_FLOAT_SPECIES =
				jdk.incubator.vector.FloatVector.SPECIES_256;

		static void dequantizeQ8_0Block(byte[] raw, int byteOffset, float scale, float[] dq) {
			int lanes = Q8_0_BYTE_SPECIES.length(); // 8
			for (int i = 0; i < 32; i += lanes) {
				jdk.incubator.vector.ByteVector bv =
						jdk.incubator.vector.ByteVector.fromArray(Q8_0_BYTE_SPECIES, raw, byteOffset + i);
				// B2I: widening byte->int, sign-extends (matches Java's own
				// byte->int promotion, which is what the scalar reference
				// relies on for negative quantized values).
				jdk.incubator.vector.IntVector iv = (jdk.incubator.vector.IntVector) bv
						.convertShape(jdk.incubator.vector.VectorOperators.B2I, Q8_0_INT_SPECIES, 0);
				// I2F: widening int->float, exact for the int8 range.
				jdk.incubator.vector.FloatVector fv = (jdk.incubator.vector.FloatVector) iv
						.convertShape(jdk.incubator.vector.VectorOperators.I2F, Q8_0_FLOAT_SPECIES, 0);
				fv = fv.mul(scale);
				fv.intoArray(dq, i);
			}
		}

		/**
		 * Dequantizes a known 32-byte pattern (covering both signed-byte
		 * boundary values and a spread of ordinary ones) and checks every
		 * output lane against the scalar-computed expected value. Only if
		 * this passes does {@link VectorQuantKernels#dequantizeQ8_0} ever
		 * dispatch to {@link #dequantizeQ8_0Block}.
		 */
		static boolean probeDequantQ8_0() {
			byte[] testValues = { -128, -127, -100, -64, -33, -32, -31, -1, 0, 1, 31, 32, 33, 63, 64, 100, 126, 127,
					-50, -25, -10, -5, 5, 10, 25, 50, 75, -75, -110, 110, -15, 15 };
			if (testValues.length != 32) {
				return false; // defensive; keeps the probe self-consistent if ever edited
			}
			byte[] raw = new byte[34];
			System.arraycopy(testValues, 0, raw, 2, 32);
			float scale = 0.037109375f; // arbitrary non-trivial scale (19/512); avoids masking bugs that scale=1 would hide

			float[] dq = new float[32];
			dequantizeQ8_0Block(raw, 2, scale, dq);

			for (int i = 0; i < 32; i++) {
				float expected = scale * testValues[i];
				if (Math.abs(dq[i] - expected) > 1e-6f) {
					return false;
				}
			}
			return true;
		}
	}
}