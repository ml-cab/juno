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

/**
 * Quality of a quantized device KV on one model, CUDA backend: the fixed text of {@link QualityProbe}, teacher-forced
 * through the production verify path with the handler on the GPU, once with {@code f16} KV and once per entry of
 * {@code types}, each in a JVM of its own ({@link QualityRunMain} with {@value QualityRunMain#GPU_PROPERTY}), since the
 * cache type is read when the handler is built. On the GPU the types also set the device copy of the KV that GPU
 * attention reads, so this measures that copy. Per type, two checks: the NLL at most 1.02x the {@code f16} one, and the
 * handler's notice saying the device copy is stored in the selected formats (a handler that kept an FP16 copy, which
 * it announces with a warning, fails it). The share of the first 512 positions where both runs' argmax agree is
 * recorded. The default types are {@code q8_0} and {@code q8_0/q4_0} (8-bit keys, 4-bit values), the settings held to
 * the device KV's quality bars.
 */
final class GpuKvQualityProbe implements KitProbe {

	static final String DEFAULT_TYPES = "q8_0,q8_0/q4_0";

	/** The start of the INFO line a handler logs when GPU attention's KV copy holds the quantized formats. */
	static final String STORED = "GPU attention's KV copy stored as ";

	/** Text both of the handler's device KV notices carry: the INFO line above and the FP16 warning. */
	private static final String DEVICE_KV = "GPU attention";

	@Override
	public ProbeReport run(String id, Map<String, String> args) throws Exception {
		String model = args.get("model");
		int tokens = KitProbe.intArg(args, "tokens", 2048);
		String heap = args.getOrDefault("childHeap", "4g");
		List<String> types = List.of(args.getOrDefault("types", DEFAULT_TYPES).split(","));
		QualityProbe.Score f16 = QualityProbe.score(model, tokens, heap, props("f16"), "f16 device KV");
		int positions = Math.min(QualityProbe.ARGMAX_POSITIONS, tokens);
		List<ProbeReport.Reading> readings = new ArrayList<>();
		List<ProbeReport.Check> checks = new ArrayList<>();
		Map<String, Object> facts = new LinkedHashMap<>();
		readings.add(new ProbeReport.Reading("mean NLL, f16 device KV", "nats", List.of(f16.meanNll()), false, false));
		facts.put("tokens", tokens);
		facts.put("f16Seconds", f16.seconds());
		facts.put("kernels", f16.kernels());
		Map<String, Object> seconds = new LinkedHashMap<>();
		Map<String, Object> runKv = new LinkedHashMap<>();
		for (String type : types) {
			QualityProbe.Score quant = QualityProbe.score(model, tokens, heap, props(type), type + " device KV");
			int agree = 0;
			for (int t = 0; t < positions; t++)
				if (quant.argmax()[t] == f16.argmax()[t])
					agree++;
			double ratio = quant.meanNll() / f16.meanNll();
			readings.add(new ProbeReport.Reading("mean NLL, " + type + " device KV", "nats", List.of(quant.meanNll()),
					false, false));
			readings.add(new ProbeReport.Reading(type + " argmax agreement, first " + positions + " positions", "%",
					List.of(100.0 * agree / positions), true, false));
			checks.add(new ProbeReport.Check(type + " device KV NLL at most 1.02x the f16 device KV NLL",
					ratio <= KvQualityProbe.MAX_RATIO, String.format(Locale.ROOT, "ratio %.5f (%.5f over %.5f)", ratio,
							quant.meanNll(), f16.meanNll())));
			checks.add(storedCheck(type, quant.deviceKv()));
			seconds.put(type, quant.seconds());
			runKv.put(type, quant.kv());
		}
		facts.put("quantizedSeconds", seconds);
		facts.put("quantizedRunKv", runKv);
		boolean passed = checks.stream().allMatch(ProbeReport.Check::passed);
		return new ProbeReport(id, ProbeKind.CHECK, passed, checks, readings, facts);
	}

	/**
	 * Passes when the run's device KV notice says the copy holds {@code type}'s formats ({@code "K q8_0, V q4_0"});
	 * {@code notice} is the handler's line, or {@code null} when it logged none.
	 */
	static ProbeReport.Check storedCheck(String type, String notice) {
		int slash = type.indexOf('/');
		String k = slash < 0 ? type : type.substring(0, slash);
		String v = slash < 0 ? type : type.substring(slash + 1);
		String want = STORED + "K " + k + ", V " + v;
		boolean stored = notice != null && notice.contains(want);
		return new ProbeReport.Check(type + " device KV stored in the selected formats", stored,
				notice == null ? "the handler logged no device KV notice, so GPU attention did not open" : notice);
	}

	/** The handler's notice about GPU attention's KV copy among {@code messages}, or {@code null}. */
	static String deviceKvNotice(List<String> messages) {
		for (String m : messages)
			if (m.contains(DEVICE_KV) && (m.contains(STORED) || m.contains("keeps its own copy")))
				return m;
		return null;
	}

	private static List<String> props(String type) {
		List<String> props = new ArrayList<>(KvQualityProbe.cacheType(type));
		props.add("-D" + QualityRunMain.GPU_PROPERTY + "=true");
		return props;
	}
}
