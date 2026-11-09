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

package cab.ml.juno.player;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.junit.jupiter.api.Test;

/**
 * The Windows launcher {@code scripts/run.bat} repeats {@code scripts/run.sh} by hand, subcommand by subcommand
 * and flag by flag. This test reads both scripts and holds the Windows one to the other: every subcommand, every
 * flag a subcommand accepts, every option it passes to the Java entry point, and every jar it runs. No
 * {@code cmd.exe} is needed: a flag added to {@code run.sh} alone fails here on every OS.
 * {@link WindowsLauncherScriptTest} holds the batch files' own rules.
 *
 * <p>The reading is textual and follows the two scripts' own conventions: a {@code run.sh} subcommand is a
 * {@code cmd_*} function whose flags are {@code case} patterns and whose forwarded options are {@code *_arg} or
 * {@code *_flag} variables and the literal options of its {@code "$JAVA"} command; a {@code run.bat} subcommand is
 * a label and the labels prefixed with its name, whose flags are {@code "%~1"=="--flag"} tests and whose forwarded
 * options are {@code set "NAME=--option ..."} variables and the literal options of its {@code "%JAVA%"} lines.
 * Batch compares are case-insensitive ({@code if /i}), so flags are compared ignoring case.
 */
class LauncherParityTest {

	private static final Path ROOT = repoRoot();

	private static final Pattern SH_DISPATCH = Pattern.compile("^\\s+([a-z][a-z-]*)\\)\\s+(cmd_[a-z_]+)\\b");
	private static final Pattern SH_CASE_FLAGS = Pattern
			.compile("^\\s+((?:-{1,2}[A-Za-z0-9][A-Za-z0-9-]*\\s*\\|\\s*)*-{1,2}[A-Za-z0-9][A-Za-z0-9-]*)\\s*\\)");
	private static final Pattern SH_ARG_VAR = Pattern.compile("\\b([a-z_]+_(?:arg|flag))=\"(--[^\"]*)");
	private static final Pattern SH_ARG_REF = Pattern.compile("\\$\\{?([a-z_]+_(?:arg|flag))\\}?");

	private static final Pattern BAT_DISPATCH = Pattern
			.compile("^if /i \"%~1\"==\"([a-z][a-z-]*)\"\\s*\\(\\s*shift\\s*&\\s*goto :([a-z_]+)\\s*\\)");
	private static final Pattern BAT_LABEL = Pattern.compile("^:([A-Za-z_]+)");
	private static final Pattern BAT_FLAG = Pattern.compile("\"%~1\"==\"(-{1,2}[A-Za-z0-9][A-Za-z0-9-]*)\"");
	private static final Pattern BAT_ARG_VAR = Pattern.compile("set \"([A-Z0-9_]+)=(--[^\"]*)");
	private static final Pattern BAT_VAR_REF = Pattern.compile("%([A-Z0-9_]+)%");

	private static final Pattern LITERAL_OPTION = Pattern.compile("(?<![\\w$%-])(--[A-Za-z][A-Za-z0-9-]*)");
	private static final Pattern JAR_NAME = Pattern.compile("([A-Za-z0-9_*%.-]+\\.jar)\\b");

	@Test
	void every_subcommand_of_run_sh_is_dispatched_by_run_bat() throws IOException {
		Set<String> missing = new TreeSet<>(shDispatch().keySet());
		missing.removeAll(batDispatch().keySet());
		assertThat(missing).as("subcommands run.sh has and run.bat does not dispatch").isEmpty();
	}

	@Test
	void run_bat_accepts_every_flag_run_sh_accepts() throws IOException {
		List<String> missing = new ArrayList<>();
		Map<String, String> bat = batDispatch();
		for (Map.Entry<String, String> sub : shDispatch().entrySet()) {
			if (!bat.containsKey(sub.getKey()))
				continue; // reported by the dispatch test
			Set<String> batFlags = lower(batFlags(bat.get(sub.getKey())));
			for (String flag : shFlags(sub.getValue()))
				if (!batFlags.contains(flag.toLowerCase(Locale.ROOT)))
					missing.add(sub.getKey() + " " + flag);
		}
		assertThat(missing).as("flags run.sh accepts and run.bat refuses").isEmpty();
	}

	@Test
	void run_bat_passes_on_every_option_run_sh_passes_to_java() throws IOException {
		List<String> missing = new ArrayList<>();
		Map<String, String> bat = batDispatch();
		for (Map.Entry<String, String> sub : shDispatch().entrySet()) {
			if (!bat.containsKey(sub.getKey()))
				continue; // reported by the dispatch test
			Set<String> batForwarded = batForwarded(bat.get(sub.getKey()));
			for (String option : shForwarded(sub.getValue()))
				if (!batForwarded.contains(option))
					missing.add(sub.getKey() + " " + option);
		}
		assertThat(missing).as("options run.sh passes to Java and run.bat does not").isEmpty();
	}

	@Test
	void run_bat_runs_only_jars_run_sh_runs() throws IOException {
		Set<String> shJars = jarNames(read("scripts/run.sh"));
		Set<String> batJars = jarNames(read("scripts/run.bat"));
		batJars.removeAll(shJars);
		assertThat(batJars).as("jars run.bat names and run.sh does not (a thin jar lacks the dependencies)").isEmpty();
	}

	// ── run.sh ───────────────────────────────────────────────────────────────

	/** Subcommand name to the function that runs it, from the dispatch {@code case}. */
	private static Map<String, String> shDispatch() throws IOException {
		Map<String, String> out = new LinkedHashMap<>();
		for (String line : read("scripts/run.sh")) {
			Matcher m = SH_DISPATCH.matcher(line);
			if (m.find())
				out.put(m.group(1), m.group(2));
		}
		assertThat(out).as("run.sh dispatch table").isNotEmpty();
		return out;
	}

	private static List<String> shFunction(String name) throws IOException {
		List<String> lines = read("scripts/run.sh");
		int start = -1;
		for (int i = 0; i < lines.size(); i++)
			if (lines.get(i).startsWith(name + "() {")) {
				start = i;
				break;
			}
		assertThat(start).as("run.sh function " + name).isNotNegative();
		int end = start + 1;
		while (end < lines.size() && !lines.get(end).equals("}"))
			end++;
		return lines.subList(start, end);
	}

	private static Set<String> shFlags(String function) throws IOException {
		Set<String> out = new TreeSet<>();
		for (String line : shFunction(function)) {
			Matcher m = SH_CASE_FLAGS.matcher(line);
			if (m.find())
				for (String flag : m.group(1).split("\\|"))
					out.add(flag.strip());
		}
		return out;
	}

	private static Set<String> shForwarded(String function) throws IOException {
		List<String> body = shFunction(function);
		Map<String, Set<String>> varOptions = new LinkedHashMap<>();
		for (String line : body) {
			if (isShComment(line))
				continue;
			Matcher m = SH_ARG_VAR.matcher(line);
			while (m.find())
				collect(LITERAL_OPTION, m.group(2), varOptions.computeIfAbsent(m.group(1), k -> new TreeSet<>()));
		}
		Set<String> out = new TreeSet<>();
		for (int i = 0; i < body.size(); i++) {
			String line = body.get(i);
			if (isShComment(line) || !line.contains("\"$JAVA\"") || line.contains("-version"))
				continue;
			StringBuilder command = new StringBuilder(line);
			while (body.get(i).stripTrailing().endsWith("\\") && i + 1 < body.size())
				command.append(' ').append(body.get(++i));
			collect(LITERAL_OPTION, command, out);
			Matcher ref = SH_ARG_REF.matcher(command);
			while (ref.find())
				out.addAll(varOptions.getOrDefault(ref.group(1), Set.of()));
		}
		return out;
	}

	private static boolean isShComment(String line) {
		String s = line.strip();
		return s.startsWith("#") || s.startsWith("echo ");
	}

	// ── run.bat ──────────────────────────────────────────────────────────────

	/** Subcommand name to the label that runs it, from the top-level dispatch lines. */
	private static Map<String, String> batDispatch() throws IOException {
		Map<String, String> out = new LinkedHashMap<>();
		for (String line : read("scripts/run.bat")) {
			Matcher m = BAT_DISPATCH.matcher(line);
			if (m.find())
				out.put(m.group(1), m.group(2));
		}
		assertThat(out).as("run.bat dispatch table").isNotEmpty();
		return out;
	}

	/**
	 * A subcommand's lines: from its label to the next label that is neither prefixed with its name nor another
	 * subcommand's ({@code lora_import} is a subcommand of its own, not part of {@code lora}).
	 */
	private static List<String> batSection(String label) throws IOException {
		List<String> lines = read("scripts/run.bat");
		Set<String> labels = new TreeSet<>(batDispatch().values());
		int start = lines.indexOf(":" + label);
		assertThat(start).as("run.bat label :" + label).isNotNegative();
		int end = start + 1;
		for (; end < lines.size(); end++) {
			Matcher m = BAT_LABEL.matcher(lines.get(end));
			if (m.find() && (labels.contains(m.group(1)) || !m.group(1).startsWith(label + "_")))
				break;
		}
		return lines.subList(start, end);
	}

	private static Set<String> batFlags(String label) throws IOException {
		Set<String> out = new TreeSet<>();
		for (String line : batSection(label))
			collect(BAT_FLAG, line, out);
		return out;
	}

	private static Set<String> batForwarded(String label) throws IOException {
		List<String> section = batSection(label);
		Map<String, Set<String>> varOptions = new LinkedHashMap<>();
		for (String line : section) {
			if (isBatComment(line))
				continue;
			Matcher m = BAT_ARG_VAR.matcher(line);
			while (m.find())
				collect(LITERAL_OPTION, m.group(2), varOptions.computeIfAbsent(m.group(1), k -> new TreeSet<>()));
		}
		Set<String> out = new TreeSet<>();
		for (String line : section) {
			if (!line.startsWith("\"%JAVA%\""))
				continue;
			collect(LITERAL_OPTION, line, out);
			Matcher ref = BAT_VAR_REF.matcher(line);
			while (ref.find())
				out.addAll(varOptions.getOrDefault(ref.group(1), Set.of()));
		}
		return out;
	}

	private static boolean isBatComment(String line) {
		String s = line.strip().toLowerCase(Locale.ROOT);
		return s.startsWith("rem ") || s.startsWith("echo ") || s.startsWith("echo.");
	}

	// ── shared ───────────────────────────────────────────────────────────────

	/** Jar file names a script names, with a version or glob part read as {@code *}. */
	private static Set<String> jarNames(List<String> lines) {
		Set<String> out = new TreeSet<>();
		for (String line : lines) {
			if (isShComment(line) || isBatComment(line))
				continue;
			Matcher m = JAR_NAME.matcher(line);
			while (m.find()) {
				String name = m.group(1);
				name = name.substring(name.lastIndexOf('/') + 1);
				out.add(name.replace("%JUNO_VERSION%", "*"));
			}
		}
		return out;
	}

	private static void collect(Pattern p, CharSequence text, Set<String> into) {
		Matcher m = p.matcher(text);
		while (m.find())
			into.add(m.group(1));
	}

	private static Set<String> lower(Set<String> in) {
		Set<String> out = new TreeSet<>();
		for (String s : in)
			out.add(s.toLowerCase(Locale.ROOT));
		return out;
	}

	static List<String> read(String relative) throws IOException {
		return Files.readAllLines(ROOT.resolve(relative), StandardCharsets.UTF_8).stream()
				.map(line -> line.endsWith("\r") ? line.substring(0, line.length() - 1) : line).toList();
	}

	private static Path repoRoot() {
		for (Path p = Path.of("").toAbsolutePath(); p != null; p = p.getParent())
			if (Files.isRegularFile(p.resolve("scripts/run.sh")) && Files.isRegularFile(p.resolve("scripts/run.bat")))
				return p;
		throw new IllegalStateException("scripts/run.sh not found above " + Path.of("").toAbsolutePath());
	}
}
