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

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Quality of a quantized host KV on one model, CPU backend: the fixed text of {@link QualityProbe}, teacher-forced
 * through the production verify path, once with an {@code f16} host KV and once with {@code type}, each in a JVM of its
 * own ({@link QualityRunMain}), since the cache type is read when the handler is built. {@code type} is one type for K
 * and V or a {@code k/v} pair; the default, {@code q8_0/q4_0} (8-bit keys, 4-bit values), is the 4-bit setting held to
 * the quality bar, since 4-bit keys lose quality on some models and 4-bit values do not. The quantized NLL may be at
 * most 1.02x the {@code f16} one. On the CPU backend attention reads the host rows, so this measures the stored types
 * themselves. The share of the first 512 positions where both runs' argmax agree is recorded.
 */
final class KvQualityProbe implements KitProbe {

	static final double MAX_RATIO = 1.02;

	/** 8-bit keys with 4-bit values. */
	static final String DEFAULT_TYPE = "q8_0/q4_0";

	@Override
	public ProbeReport run(String id, Map<String, String> args) throws Exception {
		String model = args.get("model");
		int tokens = KitProbe.intArg(args, "tokens", 2048);
		String heap = args.getOrDefault("childHeap", "4g");
		String type = args.getOrDefault("type", DEFAULT_TYPE);
		QualityProbe.Score f16 = QualityProbe.score(model, tokens, heap, cacheType("f16"), "f16 KV");
		QualityProbe.Score quant = QualityProbe.score(model, tokens, heap, cacheType(type), type + " KV");
		int positions = Math.min(QualityProbe.ARGMAX_POSITIONS, tokens);
		int agree = 0;
		for (int t = 0; t < positions; t++)
			if (quant.argmax()[t] == f16.argmax()[t])
				agree++;
		double ratio = quant.meanNll() / f16.meanNll();
		List<ProbeReport.Reading> readings = List.of(
				new ProbeReport.Reading("mean NLL, f16 KV", "nats", List.of(f16.meanNll()), false, false),
				new ProbeReport.Reading("mean NLL, " + type + " KV", "nats", List.of(quant.meanNll()), false, false),
				new ProbeReport.Reading("argmax agreement, first " + positions + " positions", "%",
						List.of(100.0 * agree / positions), true, false));
		ProbeReport.Check check = new ProbeReport.Check(type + " KV NLL at most 1.02x the f16 KV NLL",
				ratio <= MAX_RATIO, String.format(Locale.ROOT, "ratio %.5f (%.5f over %.5f)", ratio, quant.meanNll(),
						f16.meanNll()));
		Map<String, Object> facts = new LinkedHashMap<>();
		facts.put("tokens", tokens);
		facts.put("f16Seconds", f16.seconds());
		facts.put("quantizedSeconds", quant.seconds());
		facts.put("f16RunKv", f16.kv());
		facts.put("quantizedRunKv", quant.kv());
		facts.put("kernels", quant.kernels());
		return new ProbeReport(id, ProbeKind.CHECK, check.passed(), List.of(check), readings, facts);
	}

	/** {@code type} is one type for K and V, or {@code k/v}. */
	static List<String> cacheType(String type) {
		int slash = type.indexOf('/');
		String k = slash < 0 ? type : type.substring(0, slash);
		String v = slash < 0 ? type : type.substring(slash + 1);
		return List.of("-DJUNO_CACHE_TYPE_K=" + k, "-DJUNO_CACHE_TYPE_V=" + v);
	}
}
