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


import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * The plan's outlier rule, applied to a reading probe: a row is not scorable when its own repetitions
 * disagree by more than 15% of their median. A reading taken while other processes used a material share
 * of the CPU, or while another program used the GPU, is treated the same way. Any of these is re-run up to
 * twice and then marked.
 */
final class NoiseRule {

	static final double MAX_SPREAD = 0.15;
	/** Other processes' share of the machine's CPU time above which a reading counts as taken under load. */
	static final double MAX_OTHER_CPU_SHARE = 0.05;
	/** The first attempt plus two re-runs. */
	static final int MAX_ATTEMPTS = 3;

	enum Verdict {
		OK, RERUN, NOISY
	}

	private NoiseRule() {
	}

	/** (max - min) / median; 0 for fewer than two values. */
	static double spread(List<Double> values) {
		if (values.size() < 2)
			return 0;
		List<Double> s = new ArrayList<>(values);
		Collections.sort(s);
		double median = s.get(s.size() / 2);
		return median == 0 ? 0 : (s.get(s.size() - 1) - s.get(0)) / Math.abs(median);
	}

	static boolean noisy(List<Double> values) {
		return spread(values) > MAX_SPREAD + 1e-12;
	}

	/**
	 * What to do with {@code report}, taken on attempt {@code attempt} (1-based); {@code gpuBusy} when another
	 * program used the GPU during or just before it.
	 */
	static Verdict verdict(ProbeReport report, double otherCpuShare, boolean gpuBusy, int attempt) {
		if (report.kind() != ProbeKind.READING)
			return Verdict.OK;
		boolean bad = otherCpuShare > MAX_OTHER_CPU_SHARE || gpuBusy;
		for (ProbeReport.Reading r : report.readings())
			bad |= r.scored() && noisy(r.values());
		if (!bad)
			return Verdict.OK;
		return attempt < MAX_ATTEMPTS ? Verdict.RERUN : Verdict.NOISY;
	}

	/** Why {@code report} was marked, for the summary; empty when it was not. */
	static String reason(ProbeReport report, double otherCpuShare, boolean gpuBusy) {
		StringBuilder sb = new StringBuilder();
		for (ProbeReport.Reading r : report.readings())
			if (r.scored() && noisy(r.values()))
				sb.append(String.format(java.util.Locale.ROOT, "%s spread %.0f%% > 15%%; ", r.name(),
						spread(r.values()) * 100));
		if (otherCpuShare > MAX_OTHER_CPU_SHARE)
			sb.append(String.format(java.util.Locale.ROOT, "other processes used %.0f%% of the CPU; ",
					otherCpuShare * 100));
		if (gpuBusy)
			sb.append("another program used the GPU; ");
		return sb.toString().trim();
	}
}
