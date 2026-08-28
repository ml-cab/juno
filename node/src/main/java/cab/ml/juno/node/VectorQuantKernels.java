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
final class VectorQuantKernels {

	private static final Logger log = Logger.getLogger(VectorQuantKernels.class.getName());

	/**
	 * True if {@code jdk.incubator.vector} loaded and a trivial vector op
	 * executed successfully on this JVM. Checked once at class-init; callers
	 * do not need to re-check it themselves. {@link #dot} always does the
	 * correct thing (vectorized or scalar) regardless of this flag's value.
	 */
	static final boolean AVAILABLE = probe();

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
	}
}