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
import java.lang.management.ManagementFactory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;

/** Runs the operating system's tools and Juno's backend probe, and assembles a {@link HostInfo}. */
final class HostDetector {

	private HostDetector() {
	}

	/** {@code diskPath} is an existing directory whose file store the results go to. */
	static HostInfo detect(Path diskPath) throws IOException, InterruptedException {
		String os = System.getProperty("os.name");
		String lower = os.toLowerCase(Locale.ROOT);
		boolean windows = lower.contains("win");
		boolean mac = lower.contains("mac");
		Path cwd = Path.of(".").toAbsolutePath();

		DetectionParsers.CpuFacts cpu;
		HostInfo.Power power;
		String metered = "unknown";
		if (windows) {
			cpu = DetectionParsers.windowsProcessor(orEmpty(KitBootstrap.runQuietly(cwd, powershell(
					"Get-CimInstance Win32_Processor | Format-List Name,NumberOfCores,NumberOfLogicalProcessors"))));
			power = DetectionParsers.windowsBattery(orEmpty(KitBootstrap.runQuietly(cwd,
					powershell("Get-CimInstance Win32_Battery | Format-List BatteryStatus,EstimatedChargeRemaining"))));
		} else if (mac) {
			cpu = DetectionParsers.macSysctl(orEmpty(KitBootstrap.runQuietly(cwd, List.of("sysctl", "-a"))));
			power = DetectionParsers.macPmset(orEmpty(KitBootstrap.runQuietly(cwd, List.of("pmset", "-g", "batt"))));
		} else {
			cpu = DetectionParsers.linuxCpuInfo(readOrEmpty(Path.of("/proc/cpuinfo")));
			power = DetectionParsers.linuxPower(linuxSupplies());
			metered = DetectionParsers
					.nmcliMetered(orEmpty(KitBootstrap.runQuietly(cwd, List.of("nmcli", "-t", "-f", "GENERAL.METERED",
							"dev", "show"))));
		}
		List<HostInfo.Gpu> gpus = new ArrayList<>(DetectionParsers.nvidiaSmi(orEmpty(KitBootstrap.runQuietly(cwd,
				List.of("nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
						"--format=csv,noheader,nounits")))));
		gpus.addAll(DetectionParsers.rocmSmi(orEmpty(KitBootstrap.runQuietly(cwd,
				List.of("rocm-smi", "--showproductname", "--showmeminfo", "vram", "--csv")))));

		long ram = ((com.sun.management.OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean())
				.getTotalMemorySize();
		long freeDisk = Files.getFileStore(diskPath).getUsableSpace();
		Map<String, Object> backend = backendProbe();
		int threads = cpu.threads() > 0 ? cpu.threads() : Runtime.getRuntime().availableProcessors();
		return new HostInfo(os, System.getProperty("os.version"), System.getProperty("os.arch"), cpu.model(),
				cpu.cores() > 0 ? cpu.cores() : threads, threads, ram, freeDisk, cpu.flags(),
				intOf(backend.get("intVectorBits")), intOf(backend.get("floatVectorBits")),
				String.valueOf(backend.getOrDefault("useAvx", "n/a")), Boolean.TRUE.equals(backend.get("cuda")),
				Boolean.TRUE.equals(backend.get("rocm")), gpus, power, metered, JvmLauncher.clockSource(), backend);
	}

	/**
	 * {@link EnvProbe} in a JVM of its own, as every probe runs, reporting on stdout so detection writes no
	 * file; its facts, or an error entry.
	 */
	private static Map<String, Object> backendProbe() throws IOException, InterruptedException {
		List<String> cmd = JvmLauncher.command("512m", false, List.of(), ProbeMain.class.getName(),
				List.of("EnvProbe", "env", ProbeMain.STDOUT));
		Process p = new ProcessBuilder(cmd).redirectError(ProcessBuilder.Redirect.DISCARD).start();
		p.getOutputStream().close();
		String out = new String(p.getInputStream().readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
		if (!p.waitFor(5, TimeUnit.MINUTES)) {
			p.destroyForcibly();
			return Map.of("error", "backend probe timed out");
		}
		for (String line : out.split("\n"))
			if (line.startsWith(ProbeMain.STDOUT_PREFIX))
				return new LinkedHashMap<>(Json.MAPPER
						.readValue(line.substring(ProbeMain.STDOUT_PREFIX.length()), ProbeReport.class).facts());
		return Map.of("error", "backend probe exited " + p.exitValue() + " without a report");
	}

	private static Map<String, Map<String, String>> linuxSupplies() {
		Map<String, Map<String, String>> out = new HashMap<>();
		Path root = Path.of("/sys/class/power_supply");
		if (!Files.isDirectory(root))
			return out;
		try (Stream<Path> s = Files.list(root)) {
			for (Path dir : s.toList()) {
				Map<String, String> attrs = new HashMap<>();
				for (String a : List.of("type", "scope", "status", "capacity", "online")) {
					String v = readOrEmpty(dir.resolve(a)).trim();
					if (!v.isEmpty())
						attrs.put(a, v);
				}
				out.put(dir.getFileName().toString(), attrs);
			}
		} catch (IOException ignored) {
		}
		return out;
	}

	private static List<String> powershell(String command) {
		return List.of("powershell", "-NoProfile", "-NonInteractive", "-Command", command);
	}

	private static String readOrEmpty(Path p) {
		try {
			return Files.isReadable(p) ? Files.readString(p) : "";
		} catch (IOException e) {
			return "";
		}
	}

	private static String orEmpty(String s) {
		return s == null ? "" : s;
	}

	private static int intOf(Object o) {
		return o instanceof Number n ? n.intValue() : 0;
	}
}
