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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import cab.ml.juno.node.CudaAvailability;
import cab.ml.juno.node.GpuKernelTimings;

/**
 * The longest single GPU kernels Juno launches at the sweep models' shapes ({@link GpuKernelTimings}: prefill
 * attention, the tiled K-quant GEMM, the LM head), each launch timed alone by CUDA events, against the watchdog
 * a display driver holds a single kernel to (two seconds by default on Windows). The prefill paths split a long
 * window's attention and GEMM into launches of bounded work, and each reading is the heaviest of them. Two checks
 * hold the slowest launch to {@code limitMs} (500 ms, a 4x margin): at a {@code tokens}-row window, and at the
 * longest window, {@code longestTokens} rows (the positions the device KV holds; skipped when not above
 * {@code tokens}). Each kernel's launches are a reading. Arguments: {@code tokens} (default 2048),
 * {@code longestTokens} (default {@link GpuKernelTimings#LONGEST_WINDOW}), {@code reps} (timed launches per
 * kernel, default 5), {@code limitMs}.
 *
 * <p>The readings are not scored by the noise rule: the check takes the slowest launch, which a noisy
 * repetition can only make stricter.
 */
final class GpuKernelTimingProbe implements KitProbe {

	static final int WATCHDOG_MS = 2000;

	@Override
	public ProbeReport run(String id, Map<String, String> args) {
		int tokens = Integer.parseInt(args.getOrDefault("tokens", "2048"));
		int longestTokens = Integer.parseInt(
				args.getOrDefault("longestTokens", String.valueOf(GpuKernelTimings.LONGEST_WINDOW)));
		int reps = Integer.parseInt(args.getOrDefault("reps", "5"));
		double limitMs = Double.parseDouble(args.getOrDefault("limitMs", "500"));
		String bound = " <= " + fmt(limitMs) + " ms (the " + WATCHDOG_MS / 1000 + " s display-driver watchdog / "
				+ fmt(WATCHDOG_MS / limitMs) + ")";
		String name = "longest single kernel at a " + tokens + "-token window" + bound;
		boolean longest = longestTokens > tokens;
		String longestName = "longest single kernel at a " + longestTokens + "-token window, the longest prefill window,"
				+ bound;
		Map<String, Object> facts = new LinkedHashMap<>();
		facts.put("tokens", tokens);
		if (longest)
			facts.put("longestTokens", longestTokens);
		facts.put("limitMs", limitMs);
		List<GpuKernelTimings.Timing> timings;
		List<GpuKernelTimings.Timing> longestTimings = List.of();
		try {
			timings = GpuKernelTimings.measure(tokens, reps);
			if (longest)
				longestTimings = GpuKernelTimings.measureWindow(longestTokens, reps);
		} catch (RuntimeException e) {
			List<ProbeReport.Check> none = new ArrayList<>();
			none.add(new ProbeReport.Check(name, false, "no reading: " + e.getMessage()));
			if (longest)
				none.add(new ProbeReport.Check(longestName, false, "no reading: " + e.getMessage()));
			return new ProbeReport(id, ProbeKind.CHECK, false, none, List.of(), facts);
		}
		facts.put("device", CudaAvailability.deviceName(0));
		List<ProbeReport.Reading> readings = new ArrayList<>();
		List<ProbeReport.Check> checks = new ArrayList<>();
		checks.add(check(name, timings, reps, limitMs, readings, facts, "longestKernel", "longestMs"));
		if (longest)
			checks.add(check(longestName, longestTimings, reps, limitMs, readings, facts, "longestWindowKernel",
					"longestWindowMs"));
		boolean met = checks.stream().allMatch(ProbeReport.Check::passed);
		return new ProbeReport(id, ProbeKind.CHECK, met, checks, readings, facts);
	}

	/** Adds each timing as a reading, and holds the slowest launch among them to {@code limitMs}. */
	private static ProbeReport.Check check(String name, List<GpuKernelTimings.Timing> timings, int reps,
			double limitMs, List<ProbeReport.Reading> readings, Map<String, Object> facts, String kernelFact,
			String msFact) {
		GpuKernelTimings.Timing slowest = null;
		for (GpuKernelTimings.Timing t : timings) {
			readings.add(new ProbeReport.Reading(t.name(), "ms", t.millis(), false, false));
			if (slowest == null || t.maxMillis() > slowest.maxMillis())
				slowest = t;
		}
		boolean met = slowest.maxMillis() <= limitMs;
		facts.put(kernelFact, slowest.name());
		facts.put(msFact, slowest.maxMillis());
		String detail = slowest.name() + ": slowest of " + reps + " launches " + fmt(slowest.maxMillis())
				+ " ms, median " + fmt(slowest.medianMillis()) + " ms";
		return new ProbeReport.Check(name, met, detail);
	}

	private static String fmt(double v) {
		return v == Math.rint(v) ? String.valueOf((long) v) : String.format(Locale.ROOT, "%.2f", v);
	}
}
