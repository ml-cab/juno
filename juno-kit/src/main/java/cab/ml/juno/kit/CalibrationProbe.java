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


import java.util.Arrays;
import java.util.List;
import java.util.Map;

import cab.ml.juno.node.CpuDecodeKernelVariantMicrobench;
import cab.ml.juno.node.CpuThreads;

/**
 * Times the shipped Q4_K decode kernel on synthetic weights at every thread. Its median over the
 * reference host's scales the catalog's reference costs into this machine's estimate.
 */
final class CalibrationProbe implements KitProbe {

	static final String READING = "q4k_decode_ms_per_call";

	@Override
	public ProbeReport run(String id, Map<String, String> args) {
		int rows = KitProbe.intArg(args, "rows", 4096);
		int cols = KitProbe.intArg(args, "cols", 4096);
		int reps = KitProbe.intArg(args, "reps", 5);
		double[] ms = CpuDecodeKernelVariantMicrobench.productionMsPerCall(12, rows, cols, reps,
				KitProbe.longArg(args, "targetMs", 1000), KitProbe.longArg(args, "warmupMs", 3000));
		return new ProbeReport(id, ProbeKind.READING, null, List.of(),
				List.of(new ProbeReport.Reading(READING, "ms", Arrays.stream(ms).boxed().toList(), false, true)),
				Map.of("shape", "Q4_K " + rows + "x" + cols, "threads", CpuThreads.threads()));
	}
}
