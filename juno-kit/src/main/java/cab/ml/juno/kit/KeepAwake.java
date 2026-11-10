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


import java.io.PrintStream;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;

/**
 * Keeps the machine from sleeping while the kit runs, without admin rights: {@code SetThreadExecutionState}
 * through the FFM API on Windows (held by the calling thread, which runs the whole session),
 * {@code caffeinate} on macOS and {@code systemd-inhibit} on Linux where present.
 */
final class KeepAwake implements AutoCloseable {

	private static final int ES_CONTINUOUS = 0x80000000;
	private static final int ES_SYSTEM_REQUIRED = 0x00000001;

	private final Process helper;
	private final MethodHandle windowsCall;

	private KeepAwake(Process helper, MethodHandle windowsCall) {
		this.helper = helper;
		this.windowsCall = windowsCall;
	}

	/** The helper command for {@code os} that ends when process {@code pid} does, or an empty list. */
	static List<String> command(String os, long pid, boolean systemdInhibitPresent) {
		String o = os.toLowerCase(Locale.ROOT);
		if (o.contains("mac"))
			return List.of("caffeinate", "-i", "-w", String.valueOf(pid));
		if (o.contains("win") || !systemdInhibitPresent)
			return List.of();
		return List.of("systemd-inhibit", "--what=idle:sleep", "--who=juno-kit", "--why=Juno hardware validation run",
				"--mode=block", "tail", "--pid=" + pid, "-f", "/dev/null");
	}

	/** What {@code SetThreadExecutionState}'s result means: it returns the previous state, or 0 when it failed. */
	static String windowsMessage(int previousState) {
		return previousState != 0 ? "  Keeping this machine awake for the run (SetThreadExecutionState)."
				: "  Could not keep Windows awake (SetThreadExecutionState returned 0); set sleep to Never meanwhile.";
	}

	static KeepAwake start(PrintStream out) {
		String os = System.getProperty("os.name");
		if (os.toLowerCase(Locale.ROOT).contains("win")) {
			try {
				MethodHandle h = Linker.nativeLinker().downcallHandle(
						SymbolLookup.libraryLookup("kernel32", Arena.global()).find("SetThreadExecutionState")
								.orElseThrow(),
						FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
				int prev = (int) h.invokeExact(ES_CONTINUOUS | ES_SYSTEM_REQUIRED);
				out.println(windowsMessage(prev));
				if (prev != 0)
					return new KeepAwake(null, h);
			} catch (Throwable t) {
				out.println("  Could not keep Windows awake (" + t.getMessage() + "); set sleep to Never meanwhile.");
			}
			return new KeepAwake(null, null);
		}
		boolean inhibit = onPath("systemd-inhibit");
		List<String> cmd = command(os, ProcessHandle.current().pid(), inhibit);
		if (cmd.isEmpty()) {
			out.println("  No systemd-inhibit here: keep the machine from sleeping yourself while the kit runs.");
			return new KeepAwake(null, null);
		}
		try {
			return new KeepAwake(new ProcessBuilder(cmd).redirectErrorStream(true)
					.redirectOutput(ProcessBuilder.Redirect.DISCARD).start(), null);
		} catch (Exception e) {
			out.println("  Could not start " + cmd.get(0) + " (" + e.getMessage()
					+ "); keep the machine from sleeping yourself.");
			return new KeepAwake(null, null);
		}
	}

	private static boolean onPath(String exe) {
		String path = System.getenv("PATH");
		if (path == null)
			return false;
		for (String dir : path.split(java.io.File.pathSeparator))
			if (!dir.isBlank() && Files.isExecutable(Path.of(dir, exe)))
				return true;
		return false;
	}

	@Override
	public void close() {
		if (helper != null)
			helper.destroy();
		if (windowsCall != null) {
			try {
				int ignored = (int) windowsCall.invokeExact(ES_CONTINUOUS);
			} catch (Throwable ignored) {
			}
		}
	}
}
