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
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;

/**
 * Session save and restore across a restart, as a check: for each KV type (default {@code f16} and {@code q8_0}), a
 * JVM runs a session's first turn, saves it and runs its second turn uninterrupted; a second JVM restores the session
 * from the same directory and runs the same second turn ({@link SessionRunMain}). Passes when, for every type, the
 * restore reads back the saved token count, the second turn is served from the restored KV (a prefix hit, so only the
 * new turn is prefilled), and it equals the uninterrupted second turn token for token. On the CPU backend by default;
 * {@code gpu=true} runs CUDA.
 */
final class SessionProbe implements KitProbe {

	/** One {@link SessionRunMain} line. */
	record Run(int tokens, long bytes, boolean prefixHit, List<Integer> secondTurn, String secondTurnText) {
	}

	@Override
	public ProbeReport run(String id, Map<String, String> args) throws Exception {
		String model = args.get("model");
		boolean gpu = Boolean.parseBoolean(args.getOrDefault("gpu", "false"));
		String heap = args.getOrDefault("childHeap", "4g");
		List<ProbeReport.Check> checks = new ArrayList<>();
		Map<String, Object> facts = new LinkedHashMap<>();
		facts.put("backend", gpu ? "cuda" : "cpu");
		for (String type : args.getOrDefault("types", "f16,q8_0").split(",")) {
			Path dir = Files.createTempDirectory("juno-kit-session");
			try {
				Run saved = launch(model, dir, "save", gpu, heap, type.strip());
				Run restored = launch(model, dir, "restore", gpu, heap, type.strip());
				checks.addAll(checks(type.strip(), saved, restored));
				facts.put(type.strip() + ".tokens", saved.tokens());
				facts.put(type.strip() + ".bytes", saved.bytes());
			} finally {
				deleteTree(dir);
			}
		}
		boolean passed = checks.stream().allMatch(ProbeReport.Check::passed);
		return new ProbeReport(id, ProbeKind.CHECK, passed, checks, List.of(), facts);
	}

	/** The three checks of one type: the token count read back, the prefix hit, and the second turn. */
	static List<ProbeReport.Check> checks(String type, Run saved, Run restored) {
		return List.of(
				new ProbeReport.Check(type + ": the restore reads back the saved session",
						restored.tokens() == saved.tokens() && restored.bytes() == saved.bytes(),
						"saved " + saved.tokens() + " tokens, " + saved.bytes() + " bytes; restored " + restored.tokens()
								+ " tokens, " + restored.bytes() + " bytes"),
				new ProbeReport.Check(type + ": the second turn starts from the restored KV", restored.prefixHit(),
						restored.prefixHit() ? "prefix hit" : "no prefix hit: the turn was prefilled from the start"),
				new ProbeReport.Check(type + ": the second turn after the restart equals the uninterrupted one",
						restored.secondTurn().equals(saved.secondTurn()),
						"uninterrupted \"" + saved.secondTurnText() + "\"; after the restart \"" + restored.secondTurnText()
								+ "\""));
	}

	private static Run launch(String model, Path dir, String mode, boolean gpu, String heap, String type)
			throws IOException, InterruptedException {
		List<String> cmd = JvmLauncher.command(heap, false, KvQualityProbe.cacheType(type),
				SessionRunMain.class.getName(), List.of(model, dir.toString(), mode, String.valueOf(gpu)));
		Process p = new ProcessBuilder(cmd).redirectErrorStream(true).start();
		p.getOutputStream().close();
		String out = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
		if (!p.waitFor(1, TimeUnit.HOURS)) {
			p.destroyForcibly();
			throw new IOException("the session " + mode + " run timed out");
		}
		System.out.println("== " + type + " session " + mode + " (exit " + p.exitValue() + ")\n" + out);
		return parse(out, p.exitValue());
	}

	/** The run's result line among its output, or an error naming the exit code. */
	static Run parse(String output, int exit) throws IOException {
		for (String line : output.split("\n"))
			if (line.startsWith(SessionRunMain.PREFIX))
				return Json.MAPPER.readValue(line.substring(SessionRunMain.PREFIX.length()), Run.class);
		throw new IOException("the session run printed no result (exit " + exit + ")");
	}

	private static void deleteTree(Path dir) throws IOException {
		try (Stream<Path> files = Files.walk(dir)) {
			for (Path f : files.sorted(Comparator.reverseOrder()).toList())
				Files.deleteIfExists(f);
		}
	}
}
