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
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/** Command lines for the JVMs probes run in: Juno's runtime flags, the probe's heap, this JVM's class path. */
final class JvmLauncher {

	/** The flags {@code scripts/run.sh} gives every Juno JVM. */
	static final List<String> RUNTIME_FLAGS = List.of("--enable-preview", "--enable-native-access=ALL-UNNAMED",
			"--add-opens", "java.base/java.lang=ALL-UNNAMED", "--add-opens", "java.base/java.nio=ALL-UNNAMED",
			"--add-modules", "jdk.incubator.vector", "-XX:+UseG1GC");

	/**
	 * Make JFR read the operating-system clock. The performance harness does this where the kernel has
	 * rejected the CPU timestamp counter, because spans crossing cores whose counters disagree misread.
	 */
	static final List<String> OS_CLOCK_FLAGS = List.of("-XX:+UnlockExperimentalVMOptions",
			"-XX:-UseFastUnorderedTimeStamps");

	private JvmLauncher() {
	}

	/**
	 * @param fixedHeap {@code -Xms} equal to {@code -Xmx} and pre-touched, as the throughput harness runs
	 *                  Juno, so a reading does not include heap growth
	 */
	static List<String> command(String heap, boolean fixedHeap, List<String> props, String mainClass,
			List<String> args) {
		List<String> cmd = new ArrayList<>();
		cmd.add(KitBootstrap.javaBinary());
		cmd.addAll(RUNTIME_FLAGS);
		if (fixedHeap) {
			cmd.add("-Xms" + heap);
			cmd.add("-XX:+AlwaysPreTouch");
		}
		cmd.add("-Xmx" + heap);
		if (osClockNeeded(clockSource()))
			cmd.addAll(OS_CLOCK_FLAGS);
		cmd.addAll(props);
		cmd.add("-cp");
		cmd.add(absoluteClassPath(System.getProperty("java.class.path")));
		cmd.add(mainClass);
		cmd.addAll(args);
		return cmd;
	}

	/** Each entry made absolute: a probe's children, such as the launcher, run in directories of their own. */
	static String absoluteClassPath(String classPath) {
		List<String> out = new ArrayList<>();
		for (String e : classPath.split(java.io.File.pathSeparator))
			if (!e.isEmpty())
				out.add(Path.of(e).toAbsolutePath().normalize().toString());
		return String.join(java.io.File.pathSeparator, out);
	}

	/** The Linux kernel's clock source, or {@code unknown} elsewhere. */
	static String clockSource() {
		Path p = Path.of("/sys/devices/system/clocksource/clocksource0/current_clocksource");
		try {
			return Files.isReadable(p) ? Files.readString(p).trim() : "unknown";
		} catch (IOException e) {
			return "unknown";
		}
	}

	/** Same rule as the harness: the OS clock unless the kernel uses the TSC or the source is unknown. */
	static boolean osClockNeeded(String clockSource) {
		String cs = clockSource.toLowerCase(Locale.ROOT);
		return !cs.equals("tsc") && !cs.equals("unknown");
	}
}
