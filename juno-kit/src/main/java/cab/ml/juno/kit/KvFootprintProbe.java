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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * The host KV one request holds on one model, as a reading: a 2,048-token prompt and 64 generated tokens, the handler
 * told the request's length as the server tells it, at the default {@code f16} KV (or {@code type}), in a JVM of its own
 * ({@link KvFootprintRunMain}) with a fixed maximum heap, so the collector's region size is the same on every host. On
 * the GPU by default ({@code gpu=false} runs the CPU backend). Readings: the request's host KV in MiB, the bytes it
 * takes per position, and the live heap in MiB after load and with the request held.
 *
 * <p>A reading, not a check: the reference host scores Phi-3.5-mini's host KV against the build before real FP16
 * storage and exact growth (at most 0.35x) and its live heap (at most 0.75x). A run whose host KV is not the request's
 * positions at the selected types' row size fails the probe: the KV grew past the request's length.
 */
final class KvFootprintProbe implements KitProbe {

	static final String DEFAULT_TYPE = "f16";

	/** One {@link KvFootprintRunMain} line. */
	record Run(int positions, long hostKvBytes, long hostKvBytesPerPosition, long liveHeapAfterLoadBytes,
			long liveHeapWithRequestBytes, long maxHeapBytes, String kv, boolean gpuAttention) {
	}

	@Override
	public ProbeReport run(String id, Map<String, String> args) throws Exception {
		String model = args.get("model");
		int tokens = KitProbe.intArg(args, "tokens", 2048);
		int generated = KitProbe.intArg(args, "generated", 64);
		boolean gpu = Boolean.parseBoolean(args.getOrDefault("gpu", "true"));
		String type = args.getOrDefault("type", DEFAULT_TYPE);
		String heap = args.getOrDefault("childHeap", "12g");
		Run run = launch(model, tokens, generated, gpu, heap, type);
		requireExactGrowth(run);
		Map<String, Object> facts = new LinkedHashMap<>();
		facts.put("tokens", tokens);
		facts.put("generated", generated);
		facts.put("kv", run.kv());
		facts.put("backend", gpu ? "cuda" : "cpu");
		facts.put("gpuAttention", run.gpuAttention());
		facts.put("maxHeapMiB", run.maxHeapBytes() >> 20);
		return new ProbeReport(id, ProbeKind.READING, null, List.of(), readings(run), facts);
	}

	/** Fails unless the request's host KV is exactly its positions at the row size the handler reports. */
	static void requireExactGrowth(Run run) {
		if (run.hostKvBytesPerPosition() <= 0)
			throw new IllegalStateException("the handler reports no host KV bytes per position");
		long want = (long) run.positions() * run.hostKvBytesPerPosition();
		if (run.hostKvBytes() != want)
			throw new IllegalStateException("the request's host KV is " + run.hostKvBytes() + " bytes, not "
					+ run.positions() + " positions x " + run.hostKvBytesPerPosition() + " = " + want
					+ ": it grew past the request's length");
	}

	static List<ProbeReport.Reading> readings(Run run) {
		double mib = 1 << 20;
		return List.of(
				new ProbeReport.Reading("host KV of the request, " + run.positions() + " positions", "MiB",
						List.of(run.hostKvBytes() / mib), false, false),
				new ProbeReport.Reading("host KV bytes per position", "bytes",
						List.of((double) run.hostKvBytesPerPosition()), false, false),
				new ProbeReport.Reading("live heap after load", "MiB", List.of(run.liveHeapAfterLoadBytes() / mib),
						false, false),
				new ProbeReport.Reading("live heap with the request held", "MiB",
						List.of(run.liveHeapWithRequestBytes() / mib), false, false));
	}

	private static Run launch(String model, int tokens, int generated, boolean gpu, String heap, String type)
			throws IOException, InterruptedException {
		List<String> cmd = JvmLauncher.command(heap, false, KvQualityProbe.cacheType(type),
				KvFootprintRunMain.class.getName(),
				List.of(model, String.valueOf(tokens), String.valueOf(generated), String.valueOf(gpu)));
		Process p = new ProcessBuilder(cmd).redirectErrorStream(true).start();
		p.getOutputStream().close();
		String out = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
		if (!p.waitFor(1, TimeUnit.HOURS)) {
			p.destroyForcibly();
			throw new IOException("the footprint run timed out");
		}
		System.out.println("== " + type + " host KV footprint (exit " + p.exitValue() + ")\n" + out);
		return parse(out, p.exitValue());
	}

	/** The run's result line among its output, or an error naming the exit code. */
	static Run parse(String output, int exit) throws IOException {
		for (String line : output.split("\n"))
			if (line.startsWith(KvFootprintRunMain.PREFIX))
				return Json.MAPPER.readValue(line.substring(KvFootprintRunMain.PREFIX.length()), Run.class);
		throw new IOException("the footprint run printed no result (exit " + exit + ")");
	}
}
