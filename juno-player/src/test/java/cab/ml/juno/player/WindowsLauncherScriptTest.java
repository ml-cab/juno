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
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;

import org.junit.jupiter.api.Test;

/**
 * Rules for the Windows launchers ({@code juno.bat}, {@code scripts/run.bat}) that hold on any OS, read from the
 * files: they print no debug lines, they write nothing to the registry, and no {@code echo} inside a parenthesized
 * block ends that block early.
 *
 * <p>The last rule is {@code cmd.exe}'s: inside a block, an opening parenthesis in a command's text is plain text,
 * but the first closing one that is neither escaped ({@code ^)}) nor inside double quotes ends the block. Help text
 * such as {@code echo   --top-k N  (default 50)} inside {@code if ... ( ... )} therefore closes the {@code if} at
 * "50)", and every line after it runs whatever the condition was.
 */
class WindowsLauncherScriptTest {

	private static final List<String> SCRIPTS = List.of("juno.bat", "scripts/run.bat");
	private static final Pattern BLOCK_OPENER = Pattern.compile("(?i)^(if|for|else|do)\\b.*|.*\\bdo$|");

	@Test
	void the_windows_launchers_print_no_debug_lines() throws IOException {
		for (String script : SCRIPTS)
			assertThat(LauncherParityTest.read(script).stream().filter(line -> line.contains("[DBG]")).toList())
					.as(script).isEmpty();
	}

	@Test
	void the_windows_launchers_write_nothing_to_the_registry() throws IOException {
		Pattern regWrite = Pattern.compile("(?i)\\breg(?:\\.exe)?\\s+(?:add|import|delete|copy|restore|load)\\b");
		for (String script : SCRIPTS)
			assertThat(LauncherParityTest.read(script).stream().filter(line -> regWrite.matcher(line).find()).toList())
					.as(script).isEmpty();
	}

	@Test
	void no_echo_ends_its_enclosing_block_early() throws IOException {
		for (String script : SCRIPTS)
			assertThat(echoesEndingABlockEarly(LauncherParityTest.read(script))).as(script).isEmpty();
	}

	@Test
	void the_block_check_finds_the_help_text_pattern_and_passes_its_fixes() {
		List<String> helpInBlock = List.of("if /i \"%~1\"==\"--help\" (", "  echo   --top-k N  (default 50)",
				"  echo   --top-p F", "  goto :eof", ")");
		assertThat(echoesEndingABlockEarly(helpInBlock)).containsExactly("2: echo   --top-k N  (default 50)");

		List<String> sameLineTail = List.of("if not \"%HF%\"==\"\" ( set \"A=%HF%\" & echo Resolving (may take a while)... )");
		assertThat(echoesEndingABlockEarly(sameLineTail)).hasSize(1);

		List<String> fixed = List.of("if /i \"%~1\"==\"--help\" (", "  echo   --top-k N  ^(default 50^)",
				"  echo   \"C:\\Program Files (x86)\"", "  goto :eof", ")",
				"if not \"%HF%\"==\"\" ( set \"A=%HF%\" & echo Resolving ^(may take a while^)... )",
				"echo   --top-k N  (default 50)", "for %%S in (off no) do if /i \"%~2\"==\"%%S\" ( shift & goto :x )");
		assertThat(echoesEndingABlockEarly(fixed)).isEmpty();
	}

	/**
	 * Lines holding an {@code echo} whose unescaped, unquoted {@code )} closes a block opened on an earlier line
	 * (blocks here end on a line of their own) or is followed by more text on its line.
	 */
	static List<String> echoesEndingABlockEarly(List<String> lines) {
		List<String> out = new ArrayList<>();
		Deque<Integer> open = new ArrayDeque<>();
		for (int n = 1; n <= lines.size(); n++) {
			String line = lines.get(n - 1);
			String stripped = line.strip().toLowerCase(Locale.ROOT);
			if (stripped.isEmpty() || stripped.equals("rem") || stripped.startsWith("rem ") || stripped.startsWith("::"))
				continue;
			StringBuilder command = new StringBuilder();
			boolean quoted = false;
			for (int i = 0; i < line.length(); i++) {
				char c = line.charAt(i);
				if (quoted) {
					quoted = c != '"';
					command.append(c);
				} else if (c == '"') {
					quoted = true;
					command.append(c);
				} else if (c == '^') {
					if (++i < line.length())
						command.append(line.charAt(i));
				} else if (c == '&' || c == '|') {
					command.setLength(0);
				} else if (c == '(' && BLOCK_OPENER.matcher(command.toString().strip()).matches()) {
					open.push(n);
					command.setLength(0);
				} else if (c == ')' && !open.isEmpty()) {
					int openedOn = open.pop();
					boolean echo = command.toString().strip().toLowerCase(Locale.ROOT).startsWith("echo");
					if (echo && (openedOn != n || !line.substring(i + 1).isBlank()))
						out.add(n + ": " + line.strip());
					command.setLength(0);
				} else {
					command.append(c);
				}
			}
		}
		return out;
	}
}
