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


import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * The CPU integer kernels at 128 bits against forced 256 bits, on one thread and on every thread, on the
 * shipped decode kernel at Mistral 7B's FFN shapes, interleaved A B A B A B so drift lands on both widths.
 * Runs only where the JVM prefers 256-bit integer vectors; its reading is what the default width is
 * revisited on.
 */
final class KernelWidthProbe implements KitProbe {

	static final String DEFAULT_SHAPES = "12:14336:4096,14:4096:14336";

	@Override
	public ProbeReport run(String id, Map<String, String> args) throws Exception {
		int alternations = KitProbe.intArg(args, "alternations", 3);
		int reps = KitProbe.intArg(args, "reps", 1);
		long targetMs = KitProbe.longArg(args, "targetMs", 1000);
		long warmupMs = KitProbe.longArg(args, "warmupMs", 3000);
		String threadList = args.getOrDefault("threads", "1," + Runtime.getRuntime().availableProcessors());
		List<ProbeReport.Reading> readings = new ArrayList<>();
		Map<String, Object> facts = new LinkedHashMap<>();
		for (String shape : args.getOrDefault("shapes", DEFAULT_SHAPES).split(",")) {
			String[] s = shape.split(":");
			String label = (s[0].equals("12") ? "Q4_K " : "Q6_K ") + s[1] + "x" + s[2];
			for (String t : threadList.split(",")) {
				List<Double> w128 = new ArrayList<>();
				List<Double> w256 = new ArrayList<>();
				for (int a = 0; a < alternations; a++) {
					w128.add(time(128, t.trim(), s, reps, targetMs, warmupMs));
					w256.add(time(256, t.trim(), s, reps, targetMs, warmupMs));
				}
				String key = label + " threads " + t.trim();
				ProbeReport.Reading r128 = new ProbeReport.Reading(key + " 128-bit", "ms/call", w128, false, true);
				ProbeReport.Reading r256 = new ProbeReport.Reading(key + " 256-bit", "ms/call", w256, false, true);
				readings.add(r128);
				readings.add(r256);
				facts.put(key + " 256-bit over 128-bit time", r256.median() / r128.median());
			}
		}
		return new ProbeReport(id, ProbeKind.READING, null, List.of(), readings, facts);
	}

	private static double time(int width, String threads, String[] shape, int reps, long targetMs, long warmupMs)
			throws IOException, InterruptedException {
		List<String> cmd = JvmLauncher.command("2g", false,
				List.of("-Djuno.cpu.integerWidth=" + width, "-Djuno.threads=" + threads),
				KernelTimingMain.class.getName(),
				List.of(shape[0], shape[1], shape[2], String.valueOf(reps), String.valueOf(targetMs),
						String.valueOf(warmupMs)));
		Process p = new ProcessBuilder(cmd).redirectErrorStream(true).start();
		p.getOutputStream().close();
		String out = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
		if (!p.waitFor(10, TimeUnit.MINUTES)) {
			p.destroyForcibly();
			throw new IOException("kernel timing at " + width + " bits timed out");
		}
		for (String line : out.split("\n"))
			if (line.startsWith("ms_per_call="))
				return Double.parseDouble(line.substring("ms_per_call=".length()).trim());
		throw new IOException("kernel timing at " + width + " bits printed no result (exit " + p.exitValue() + "):\n"
				+ out);
	}
}
