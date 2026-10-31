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


/**
 * The runner's record of one probe: its status, how long it took, how many attempts the noise rule asked
 * for, the share of CPU other processes used while it ran, the most device memory its own processes held,
 * and whether another program used the GPU during or just before it.
 */
record ProbeOutcome(String id, String status, long wallMillis, int attempts, String note, double otherCpuShare,
		long ownGpuPeakMiB, boolean gpuBusy) {

	/** A check probe whose checks all passed. */
	static final String PASS = "PASS";
	/** A check probe with at least one failed check. */
	static final String FAIL = "FAIL";
	/** A reading probe whose readings are within the noise rule. */
	static final String DONE = "DONE";
	/** A reading probe still noisy, or still under load, after its re-runs. */
	static final String NOISY = "NOISY";
	/** The child JVM crashed, exited non-zero or wrote no report. */
	static final String ERROR = "ERROR";
	static final String TIMEOUT = "TIMEOUT";
}
