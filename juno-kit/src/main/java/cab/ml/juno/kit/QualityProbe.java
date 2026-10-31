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
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * The model-quality NLL test on one model, CPU backend: the fixed text (the public-domain opening of the
 * Declaration of Independence and a frozen excerpt of the how-to reference), tokenized by the model's own
 * tokenizer and cut to {@code tokens + 1} ids, teacher-forced through the production verify path in windows
 * of 128, once on the integer kernels and once on the float-activation kernels, each in a JVM of its own
 * ({@link QualityRunMain}), since the kernel setting is read when the JVM starts. The mean NLL of each is a
 * reading {@code ingest} compares with the reference run's; the integer one may be at most 1.01x the float
 * one, the test's threshold. The share of the first 512 positions where both settings' argmax agree is
 * recorded.
 */
final class QualityProbe implements KitProbe {

	static final int ARGMAX_POSITIONS = 512;
	static final double MAX_RATIO = 1.01;

	@Override
	public ProbeReport run(String id, Map<String, String> args) throws Exception {
		String model = args.get("model");
		int tokens = KitProbe.intArg(args, "tokens", 2048);
		String heap = args.getOrDefault("childHeap", "4g");
		Score integer = score(model, tokens, heap, true);
		Score floatAct = score(model, tokens, heap, false);
		int positions = Math.min(ARGMAX_POSITIONS, tokens);
		int agree = 0;
		for (int t = 0; t < positions; t++)
			if (integer.argmax()[t] == floatAct.argmax()[t])
				agree++;
		double ratio = integer.meanNll() / floatAct.meanNll();
		List<ProbeReport.Reading> readings = List.of(
				new ProbeReport.Reading("mean NLL, integer kernels", "nats", List.of(integer.meanNll()), false, false),
				new ProbeReport.Reading("mean NLL, float-activation kernels", "nats", List.of(floatAct.meanNll()),
						false, false),
				new ProbeReport.Reading("argmax agreement, first " + positions + " positions", "%",
						List.of(100.0 * agree / positions), true, false));
		ProbeReport.Check check = new ProbeReport.Check("integer-kernel NLL at most 1.01x the float-activation NLL",
				ratio <= MAX_RATIO, String.format(Locale.ROOT, "ratio %.5f (%.5f over %.5f)", ratio, integer.meanNll(),
						floatAct.meanNll()));
		Map<String, Object> facts = new LinkedHashMap<>();
		facts.put("tokens", tokens);
		facts.put("integerSeconds", integer.seconds());
		facts.put("floatSeconds", floatAct.seconds());
		facts.put("integerRunKernels", integer.kernels());
		facts.put("floatRunKernels", floatAct.kernels());
		return new ProbeReport(id, ProbeKind.CHECK, check.passed(), List.of(check), readings, facts);
	}

	/** One kernel setting's score, in a JVM of its own: the setting is read once, when the JVM starts. */
	private static Score score(String model, int tokens, String heap, boolean integer)
			throws IOException, InterruptedException {
		List<String> props = integer ? List.of() : List.of("-Djuno.cpu.integerKernels=false");
		List<String> cmd = JvmLauncher.command(heap, false, props, QualityRunMain.class.getName(),
				List.of(model, String.valueOf(tokens)));
		Process p = new ProcessBuilder(cmd).redirectErrorStream(true).start();
		p.getOutputStream().close();
		String out = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
		if (!p.waitFor(2, TimeUnit.HOURS)) {
			p.destroyForcibly();
			throw new IOException("the NLL run timed out");
		}
		System.out.println("== " + (integer ? "integer" : "float-activation") + " kernels (exit " + p.exitValue()
				+ ")\n" + out);
		for (String line : out.split("\n"))
			if (line.startsWith(QualityRunMain.PREFIX))
				return Json.MAPPER.readValue(line.substring(QualityRunMain.PREFIX.length()), Score.class);
		throw new IOException("the NLL run printed no result (exit " + p.exitValue() + ")");
	}

	/** {@code argmax} holds the first {@link #ARGMAX_POSITIONS} positions only. */
	record Score(double meanNll, int[] argmax, double seconds, String kernels) {
	}

	static String text() throws IOException {
		try (InputStream in = QualityProbe.class.getResourceAsStream("model-quality-nll.txt")) {
			if (in == null)
				throw new IOException("model-quality-nll.txt is missing from the kit");
			return new String(in.readAllBytes(), StandardCharsets.UTF_8);
		}
	}
}
