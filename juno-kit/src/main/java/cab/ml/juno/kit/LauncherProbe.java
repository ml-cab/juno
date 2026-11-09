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


import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * The launcher scripts a user runs, {@code juno} or, on Windows, {@code juno.bat} (through {@code cmd.exe}): local
 * mode with {@code --api-port} answers one chat request, then exits by itself, with code 0 and no process left,
 * when {@code exit} is typed at its console. Also the one check only the script can make: {@code juno lora} given
 * {@code --grammar-file} warns that constrained decoding does nothing for training. The script prints no
 * {@code [DBG]} lines in either run, and on Windows it writes nothing to the registry ({@link ConsoleRegistryKey}).
 */
final class LauncherProbe implements KitProbe {

	static final String LORA_GRAMMAR_WARNING = "constrained decoding is a no-op for LoRA training";
	static final String DEBUG_MARK = "[DBG]";
	private static final Duration START = Duration.ofMinutes(10);

	@Override
	public ProbeReport run(String id, Map<String, String> args) throws Exception {
		Path model = Path.of(args.get("model"));
		Path root = Path.of(args.getOrDefault("root", System.getProperty("juno.kit.root", "."))).toAbsolutePath()
				.normalize();
		Path work = Path.of(args.getOrDefault("logDir", ".")).resolve(id);
		boolean windows = System.getProperty("os.name", "").toLowerCase(Locale.ROOT).startsWith("windows");
		List<String> launcher = windows ? List.of("cmd.exe", "/c", root.resolve("juno.bat").toString())
				: List.of(root.resolve("juno").toString());
		List<ProbeReport.Check> checks = new ArrayList<>();
		Map<String, Object> facts = new LinkedHashMap<>();
		facts.put("launcher", windows ? "juno.bat" : "juno");
		String registryBefore = windows ? ConsoleRegistryKey.lastWriteTime() : null;
		List<Path> logs = List.of(work.resolve("local").resolve(JunoProcess.LOG),
				work.resolve("lora-help").resolve(JunoProcess.LOG));

		List<String> local = new ArrayList<>(launcher);
		local.addAll(List.of("local", "--model-path", model.toString(), "--max-tokens", "8", "--temperature", "0",
				"--cpu", "--heap", args.getOrDefault("serverHeap", "4g")));
		int exit;
		List<ProcessHandle> started;
		try {
			JunoProcess s = JunoProcess.serveCommand(work.resolve("local"), local, START);
			JunoProcess.Answer a;
			try {
				a = s.post(GrammarProbe.PATH, GrammarProbe.body(s.modelId(), "Name three colours.", 8).toString());
				started = s.tree();
			} finally {
				exit = s.stop();
			}
			String text = GrammarProbe.content(a);
			checks.add(new ProbeReport.Check("local mode with --api-port answers a chat request",
					a.status() == 200 && !text.isBlank(), GrammarProbe.describe(a)));
			facts.put("reply", text);
		} catch (Exception e) {
			checks.add(new ProbeReport.Check("local mode with --api-port answers a chat request", false,
					String.valueOf(e.getMessage())));
			checks.add(new ProbeReport.Check("the launcher exits cleanly on exit", false, "it never served"));
			checks.add(loraWarning(launcher, work));
			checks.add(noDebugLines(logs));
			if (windows)
				checks.add(registryUnchanged(registryBefore, ConsoleRegistryKey.lastWriteTime()));
			return new ProbeReport(id, ProbeKind.CHECK, false, checks, List.of(), facts);
		}
		Thread.sleep(2000);
		long left = started.stream().filter(ProcessHandle::isAlive).count();
		checks.add(new ProbeReport.Check("the launcher exits cleanly on exit", exit == 0 && left == 0,
				"exit code " + exit + (exit == -1 ? " (did not stop within 90 s)" : "") + ", " + left
						+ " of its processes still running"));
		facts.put("exitCode", exit);
		checks.add(loraWarning(launcher, work));
		checks.add(noDebugLines(logs));
		if (windows)
			checks.add(registryUnchanged(registryBefore, ConsoleRegistryKey.lastWriteTime()));
		return new ProbeReport(id, ProbeKind.CHECK, checks.stream().allMatch(ProbeReport.Check::passed), checks,
				List.of(), facts);
	}

	/** The launcher's debug lines ({@link #DEBUG_MARK}) in its output, both runs. */
	private static ProbeReport.Check noDebugLines(List<Path> logs) throws Exception {
		int count = 0;
		String first = null;
		for (Path log : logs) {
			if (!Files.exists(log))
				continue;
			for (String line : new String(Files.readAllBytes(log), StandardCharsets.UTF_8).lines().toList())
				if (line.contains(DEBUG_MARK)) {
					count++;
					if (first == null)
						first = line.strip();
				}
		}
		return new ProbeReport.Check("the launcher prints no " + DEBUG_MARK + " lines", count == 0,
				count == 0 ? "none in its output" : count + " lines, the first: " + first);
	}

	/**
	 * Windows: {@code HKCU\Console}'s last-write time is the same after both runs as before them. A write by
	 * another program during the run would also move it; the detail gives both times.
	 */
	private static ProbeReport.Check registryUnchanged(String before, String after) {
		boolean readable = !before.startsWith("unreadable") && !after.startsWith("unreadable");
		return new ProbeReport.Check("the launcher writes nothing to the registry (HKCU\\Console unchanged)",
				readable && before.equals(after), "last-write time before " + before + ", after " + after);
	}

	private static ProbeReport.Check loraWarning(List<String> launcher, Path work) throws Exception {
		Path dir = work.resolve("lora-help");
		Files.createDirectories(dir);
		Path log = dir.resolve(JunoProcess.LOG);
		List<String> cmd = new ArrayList<>(launcher);
		cmd.addAll(List.of("lora", "--grammar-file", dir.resolve("unused.gbnf").toString(), "--help"));
		Process p = new ProcessBuilder(cmd).directory(dir.toFile()).redirectErrorStream(true)
				.redirectOutput(ProcessBuilder.Redirect.to(log.toFile())).redirectInput(ProcessBuilder.Redirect.PIPE)
				.start();
		p.getOutputStream().close();
		String name = "lora with --grammar-file warns that " + LORA_GRAMMAR_WARNING;
		if (!p.waitFor(120, TimeUnit.SECONDS)) {
			p.descendants().forEach(ProcessHandle::destroyForcibly);
			p.destroyForcibly();
			return new ProbeReport.Check(name, false, "no exit within 120 s");
		}
		String out = Files.readString(log, StandardCharsets.UTF_8);
		boolean warned = out.contains(LORA_GRAMMAR_WARNING);
		return new ProbeReport.Check(name, warned, warned ? "warned (exit code " + p.exitValue() + ")"
				: "no warning; last lines: " + JunoProcess.tail(log, 5));
	}
}
