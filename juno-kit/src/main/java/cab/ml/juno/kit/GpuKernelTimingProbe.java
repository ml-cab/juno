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
 * a display driver holds a single kernel to (two seconds by default on Windows). The check holds the slowest
 * launch of all to {@code limitMs} (500 ms, a 4x margin); each kernel's launches are a reading. Arguments:
 * {@code tokens} (the prefill window, default 2048), {@code reps} (timed launches per kernel, default 5),
 * {@code limitMs}.
 *
 * <p>The readings are not scored by the noise rule: the check takes the slowest launch, which a noisy
 * repetition can only make stricter.
 */
final class GpuKernelTimingProbe implements KitProbe {

	static final int WATCHDOG_MS = 2000;

	@Override
	public ProbeReport run(String id, Map<String, String> args) {
		int tokens = Integer.parseInt(args.getOrDefault("tokens", "2048"));
		int reps = Integer.parseInt(args.getOrDefault("reps", "5"));
		double limitMs = Double.parseDouble(args.getOrDefault("limitMs", "500"));
		String name = "longest single kernel at a " + tokens + "-token window <= " + fmt(limitMs) + " ms (the "
				+ WATCHDOG_MS / 1000 + " s display-driver watchdog / " + fmt(WATCHDOG_MS / limitMs) + ")";
		Map<String, Object> facts = new LinkedHashMap<>();
		facts.put("tokens", tokens);
		facts.put("limitMs", limitMs);
		List<GpuKernelTimings.Timing> timings;
		try {
			timings = GpuKernelTimings.measure(tokens, reps);
		} catch (RuntimeException e) {
			return new ProbeReport(id, ProbeKind.CHECK, false,
					List.of(new ProbeReport.Check(name, false, "no reading: " + e.getMessage())), List.of(), facts);
		}
		facts.put("device", CudaAvailability.deviceName(0));
		List<ProbeReport.Reading> readings = new ArrayList<>();
		GpuKernelTimings.Timing longest = null;
		for (GpuKernelTimings.Timing t : timings) {
			readings.add(new ProbeReport.Reading(t.name(), "ms", t.millis(), false, false));
			if (longest == null || t.maxMillis() > longest.maxMillis())
				longest = t;
		}
		boolean met = longest.maxMillis() <= limitMs;
		facts.put("longestKernel", longest.name());
		facts.put("longestMs", longest.maxMillis());
		String detail = longest.name() + ": slowest of " + reps + " launches " + fmt(longest.maxMillis()) + " ms, median "
				+ fmt(longest.medianMillis()) + " ms";
		return new ProbeReport(id, ProbeKind.CHECK, met, List.of(new ProbeReport.Check(name, met, detail)), readings,
				facts);
	}

	private static String fmt(double v) {
		return v == Math.rint(v) ? String.valueOf((long) v) : String.format(Locale.ROOT, "%.2f", v);
	}
}
