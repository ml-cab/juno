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
 * Reconstruction / projected-merge metrics for Tier-5 quantized merge paths.
 *
 * <p>All fields are finite. Saturation is the fraction of elements that hit the
 * extreme representable code after encode (caller-supplied count).
 */
public record QuantizedMergeMetrics(
		double rmse,
		double maxAbsError,
		long changedBlocks,
		long totalBlocks,
		double saturationRate,
		double deltaRetention) {

	public QuantizedMergeMetrics {
		if (!(rmse >= 0) || Double.isNaN(rmse) || Double.isInfinite(rmse)) {
			throw new IllegalArgumentException("rmse must be finite and non-negative");
		}
		if (!(maxAbsError >= 0) || Double.isNaN(maxAbsError) || Double.isInfinite(maxAbsError)) {
			throw new IllegalArgumentException("maxAbsError must be finite and non-negative");
		}
		if (changedBlocks < 0 || totalBlocks < 0 || changedBlocks > totalBlocks) {
			throw new IllegalArgumentException(
					"changedBlocks/totalBlocks invalid: " + changedBlocks + "/" + totalBlocks);
		}
		if (!(saturationRate >= 0) || saturationRate > 1.0
				|| Double.isNaN(saturationRate) || Double.isInfinite(saturationRate)) {
			throw new IllegalArgumentException("saturationRate must be in [0,1]");
		}
		if (Double.isNaN(deltaRetention) || Double.isInfinite(deltaRetention)) {
			throw new IllegalArgumentException("deltaRetention must be finite");
		}
	}

	/**
	 * Compare two equal-length float arrays (e.g. original vs decode(encode(...))).
	 *
	 * @param reference reference floats
	 * @param actual    reconstructed floats
	 * @return RMSE and max abs error; block/saturation/delta fields are zero placeholders
	 */
	public static QuantizedMergeMetrics ofReconstruction(float[] reference, float[] actual) {
		if (reference == null || actual == null) {
			throw new IllegalArgumentException("reference and actual must be non-null");
		}
		if (reference.length != actual.length) {
			throw new IllegalArgumentException(
					"length mismatch: " + reference.length + " vs " + actual.length);
		}
		if (reference.length == 0) {
			throw new IllegalArgumentException("empty arrays");
		}
		double sumSq = 0;
		double maxAbs = 0;
		for (int i = 0; i < reference.length; i++) {
			double e = (double) reference[i] - actual[i];
			sumSq += e * e;
			double ae = Math.abs(e);
			if (ae > maxAbs) {
				maxAbs = ae;
			}
		}
		double rmse = Math.sqrt(sumSq / reference.length);
		return new QuantizedMergeMetrics(rmse, maxAbs, 0, 0, 0.0, 0.0);
	}

	/**
	 * Delta-retention projection: {@code 1 - ‖Δ_target − Δ_retained‖ / ‖Δ_target‖}.
	 *
	 * <p>When {@code ‖Δ_target‖} is zero, returns {@code 1.0} (vacuous perfect retention).
	 */
	public static double deltaRetention(float[] targetDelta, float[] retainedDelta) {
		if (targetDelta == null || retainedDelta == null) {
			throw new IllegalArgumentException("deltas must be non-null");
		}
		if (targetDelta.length != retainedDelta.length) {
			throw new IllegalArgumentException("delta length mismatch");
		}
		double targetNorm = 0;
		double errNorm = 0;
		for (int i = 0; i < targetDelta.length; i++) {
			double t = targetDelta[i];
			double r = retainedDelta[i];
			targetNorm += t * t;
			double e = t - r;
			errNorm += e * e;
		}
		if (targetNorm == 0.0) {
			return 1.0;
		}
		return 1.0 - Math.sqrt(errNorm) / Math.sqrt(targetNorm);
	}
}
