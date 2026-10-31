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
package cab.ml.juno.kit;


import java.time.Duration;
import java.util.List;

/**
 * Run-time estimates: each probe's reference cost, scaled by the calibration probe's reading against the
 * reference host's, plus download time at the measured rate, shown as a range.
 */
final class Estimator {

	/** The range shown around the point estimate. */
	static final double LOW = 0.85;
	static final double HIGH = 1.20;
	/** A calibration reading outside these multiples of the reference host's is clamped. */
	static final double MIN_SCALE = 0.25;
	static final double MAX_SCALE = 8.0;
	/** Reference seconds that must have run before the observed pace replaces the calibrated scale. */
	static final double OBSERVED_AFTER_SECONDS = 60;

	record Range(Duration low, Duration point, Duration high) {

		static Range of(double seconds) {
			return new Range(Duration.ofSeconds(Math.round(seconds * LOW)), Duration.ofSeconds(Math.round(seconds)),
					Duration.ofSeconds(Math.round(seconds * HIGH)));
		}
	}

	private Estimator() {
	}

	/** This host's calibration time over the reference host's; 1.0 when either is missing. */
	static double scale(double calibrationMs, double referenceCalibrationMs) {
		if (!(calibrationMs > 0) || !(referenceCalibrationMs > 0))
			return 1.0;
		return Math.max(MIN_SCALE, Math.min(MAX_SCALE, calibrationMs / referenceCalibrationMs));
	}

	static Range estimate(List<Probe> probes, double scale, long downloadBytes, double bytesPerSecond) {
		double seconds = referenceSeconds(probes) * scale;
		if (downloadBytes > 0 && bytesPerSecond > 0)
			seconds += downloadBytes / bytesPerSecond;
		return Range.of(seconds);
	}

	/**
	 * Time left for {@code left}: at the observed pace of the finished probes once they account for
	 * {@link #OBSERVED_AFTER_SECONDS} of reference time, at the calibrated scale before that.
	 */
	static Range remaining(List<Probe> done, List<Duration> actual, List<Probe> left, double calibratedScale) {
		double ref = referenceSeconds(done);
		double scale = calibratedScale;
		if (ref >= OBSERVED_AFTER_SECONDS) {
			double took = 0;
			for (Duration d : actual)
				took += d.toMillis() / 1000.0;
			scale = took / ref;
		}
		return Range.of(referenceSeconds(left) * scale);
	}

	private static double referenceSeconds(List<Probe> probes) {
		double s = 0;
		for (Probe p : probes)
			s += p.refCostSeconds();
		return s;
	}
}
