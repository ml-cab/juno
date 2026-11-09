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


import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Entry point of a probe's JVM: {@code ProbeMain <probe class> <probe id> <report.json> [key=value ...]}.
 * Probes are named in a switch rather than loaded by reflection, so renaming or removing one breaks the
 * build instead of a tester's run.
 */
public final class ProbeMain {

	/** As the report path: print the report on stdout, on one line after {@link #STDOUT_PREFIX}. */
	static final String STDOUT = "-";
	static final String STDOUT_PREFIX = "JUNO-KIT-REPORT ";

	private ProbeMain() {
	}

	public static void main(String[] args) {
		if (args.length < 3) {
			System.err.println("usage: ProbeMain <probe class> <probe id> <report.json> [key=value ...]");
			System.exit(2);
		}
		try {
			ProbeReport report = probe(args[0]).run(args[1], parseArgs(args, 3));
			if (args[2].equals(STDOUT))
				System.out.println(STDOUT_PREFIX + Json.MAPPER.writer()
						.without(com.fasterxml.jackson.databind.SerializationFeature.INDENT_OUTPUT)
						.writeValueAsString(report));
			else
				Json.write(Path.of(args[2]), report);
			System.out.flush();
			// Probes start pools, schedulers and gRPC servers; exiting here ends the JVM with them.
			System.exit(0);
		} catch (Throwable t) {
			t.printStackTrace();
			System.exit(1);
		}
	}

	static KitProbe probe(String name) {
		return switch (name) {
		case "EnvProbe" -> new EnvProbe();
		case "CalibrationProbe" -> new CalibrationProbe();
		case "KernelWidthProbe" -> new KernelWidthProbe();
		case "ThroughputProbe" -> new ThroughputProbe();
		case "LiveChecksProbe" -> new LiveChecksProbe();
		case "GpuResidencyProbe" -> new GpuResidencyProbe();
		case "PackedKquantProbe" -> new PackedKquantProbe();
		case "GpuKernelTimingProbe" -> new GpuKernelTimingProbe();
		case "UnitTestsProbe" -> new UnitTestsProbe();
		case "ConsistencyProbe" -> new ConsistencyProbe();
		case "GrammarProbe" -> new GrammarProbe();
		case "ToolsProbe" -> new ToolsProbe();
		case "ContextPolicyProbe" -> new ContextPolicyProbe();
		case "LongPromptProbe" -> new LongPromptProbe();
		case "CpuHotPathProbe" -> new CpuHotPathProbe();
		case "QualityProbe" -> new QualityProbe();
		case "ParallelProbe" -> new ParallelProbe();
		case "ScheduleProbe" -> new ScheduleProbe();
		case "MixedPrefillProbe" -> new MixedPrefillProbe();
		case "PrefillBatchProbe" -> new PrefillBatchProbe();
		case "LoraProbe" -> new LoraProbe();
		case "VisionProbe" -> new VisionProbe();
		case "LauncherProbe" -> new LauncherProbe();
		default -> throw new IllegalArgumentException("unknown probe class " + name);
		};
	}

	static Map<String, String> parseArgs(String[] args, int from) {
		Map<String, String> m = new LinkedHashMap<>();
		for (int i = from; i < args.length; i++) {
			int eq = args[i].indexOf('=');
			if (eq <= 0)
				throw new IllegalArgumentException("probe argument is not key=value: " + args[i]);
			m.put(args[i].substring(0, eq), args[i].substring(eq + 1));
		}
		return m;
	}
}
