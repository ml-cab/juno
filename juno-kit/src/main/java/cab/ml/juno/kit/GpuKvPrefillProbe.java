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
 * Prefill time with a quantized device KV on one model, CUDA backend: the long prompt prefilled through the serving
 * path with the handler on the GPU, once with {@code f16} KV and once with {@code type} (default {@code q8_0}), each in
 * a JVM of its own ({@link KvPrefillRunMain}), since the cache type is read when the handler is built; on the GPU the
 * type also sets the device copy of the KV that GPU attention reads and the prefill writes. Two widths by default: the
 * whole prompt in one window and 128-token windows ({@code --prefill-batch}). Readings per width: each side's prefill
 * milliseconds per repetition, and the quantized side's median over the {@code f16} side's.
 *
 * <p>A reading, not a check: the reference host scores the ratio pinned (at most 1.10x); a tester's host records it.
 * A run whose prefill was not the device's (GPU attention or the device prefill region did not open), or whose device
 * copy is not stored in {@code type}'s formats, fails the probe rather than giving a reading of another path.
 */
final class GpuKvPrefillProbe implements KitProbe {

	static final String DEFAULT_TYPE = "q8_0";
	static final String DEFAULT_WINDOWS = "2048,128";

	/** One {@link KvPrefillRunMain} line: milliseconds per width (as printed), and what the run had open. */
	record Run(Map<String, List<Double>> ms, boolean gpuAttention, boolean prefillRegion, String kv, String deviceKv) {
	}

	@Override
	public ProbeReport run(String id, Map<String, String> args) throws Exception {
		String model = args.get("model");
		int tokens = KitProbe.intArg(args, "tokens", 2048);
		int reps = KitProbe.intArg(args, "reps", 3);
		String windows = args.getOrDefault("windows", DEFAULT_WINDOWS);
		String type = args.getOrDefault("type", DEFAULT_TYPE);
		String heap = args.getOrDefault("childHeap", "4g");
		Run f16 = launch(model, tokens, windows, reps, heap, "f16");
		Run quant = launch(model, tokens, windows, reps, heap, type);
		requireDevicePrefill(f16, "f16");
		requireDevicePrefill(quant, type);
		Map<String, Object> facts = new LinkedHashMap<>();
		facts.put("tokens", tokens);
		facts.put("reps", reps);
		facts.put("f16Kv", f16.kv());
		facts.put("quantizedKv", quant.kv());
		facts.put("deviceKv", quant.deviceKv());
		return new ProbeReport(id, ProbeKind.READING, null, List.of(), readings(f16, quant, type), facts);
	}

	/**
	 * Fails unless {@code run} prefilled on the device: GPU attention and the device prefill region open, and for a
	 * quantized {@code type} the handler's notice naming its formats ({@link GpuKvQualityProbe#storedCheck}).
	 */
	static void requireDevicePrefill(Run run, String type) {
		if (!run.gpuAttention())
			throw new IllegalStateException(type + " run: GPU attention did not open, so the prefill was not the device's");
		if (!run.prefillRegion())
			throw new IllegalStateException(type + " run: the device prefill region did not open, so the prefill was"
					+ " not the device's");
		if (!"f16".equals(type)) {
			ProbeReport.Check stored = GpuKvQualityProbe.storedCheck(type, run.deviceKv());
			if (!stored.passed())
				throw new IllegalStateException(type + " run: " + stored.detail());
		}
	}

	/** Per width, in the runs' order: both sides' milliseconds (scored) and the quantized median over the f16 one. */
	static List<ProbeReport.Reading> readings(Run f16, Run quant, String type) {
		List<ProbeReport.Reading> out = new ArrayList<>();
		for (Map.Entry<String, List<Double>> e : f16.ms().entrySet()) {
			String width = "windows of " + e.getKey() + " tokens";
			List<Double> q = quant.ms().get(e.getKey());
			if (q == null)
				throw new IllegalStateException("the " + type + " run has no prefill of " + width);
			ProbeReport.Reading a = new ProbeReport.Reading("prefill ms, f16 device KV, " + width, "ms", e.getValue(),
					false, true);
			ProbeReport.Reading b = new ProbeReport.Reading("prefill ms, " + type + " device KV, " + width, "ms", q,
					false, true);
			out.add(a);
			out.add(b);
			out.add(new ProbeReport.Reading(type + " over f16 prefill time, " + width, "x",
					List.of(b.median() / a.median()), false, false));
		}
		return out;
	}

	private static Run launch(String model, int tokens, String windows, int reps, String heap, String type)
			throws IOException, InterruptedException {
		List<String> cmd = JvmLauncher.command(heap, false, KvQualityProbe.cacheType(type),
				KvPrefillRunMain.class.getName(), List.of(model, String.valueOf(tokens), windows, String.valueOf(reps)));
		Process p = new ProcessBuilder(cmd).redirectErrorStream(true).start();
		p.getOutputStream().close();
		String out = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
		if (!p.waitFor(1, TimeUnit.HOURS)) {
			p.destroyForcibly();
			throw new IOException("the " + type + " prefill run timed out");
		}
		System.out.println("== " + type + " device KV (exit " + p.exitValue() + ")\n" + out);
		return parse(out, type, p.exitValue());
	}

	/** The run's result line among its output, or an error naming the type and exit code. */
	static Run parse(String output, String type, int exit) throws IOException {
		for (String line : output.split("\n"))
			if (line.startsWith(KvPrefillRunMain.PREFIX))
				return Json.MAPPER.readValue(line.substring(KvPrefillRunMain.PREFIX.length()), Run.class);
		throw new IOException("the " + type + " prefill run printed no result (exit " + exit + ")");
	}
}
