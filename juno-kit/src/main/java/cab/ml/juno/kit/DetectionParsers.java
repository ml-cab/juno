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
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/** Pure parsers for what the operating system and the vendor tools print; {@link HostDetector} runs them. */
final class DetectionParsers {

	/** {@code flags} are lower-cased; 0 means unknown. */
	record CpuFacts(String model, int cores, int threads, Set<String> flags) {

		CpuFacts {
			flags = Set.copyOf(flags);
		}
	}

	private static final HostInfo.Power NO_BATTERY = new HostInfo.Power(false, false, -1);

	private DetectionParsers() {
	}

	/** {@code /proc/cpuinfo}: x86 ({@code model name}, {@code flags}) or Arm ({@code Features}, no model line). */
	static CpuFacts linuxCpuInfo(String text) {
		String model = "";
		int threads = 0;
		Set<String> cores = new HashSet<>();
		Set<String> flags = new HashSet<>();
		String physical = "0";
		for (String line : text.split("\n")) {
			int colon = line.indexOf(':');
			if (colon < 0)
				continue;
			String key = line.substring(0, colon).trim();
			String value = line.substring(colon + 1).trim();
			switch (key) {
			case "processor" -> threads++;
			case "model name" -> {
				if (model.isEmpty())
					model = value;
			}
			case "physical id" -> physical = value;
			case "core id" -> cores.add(physical + ":" + value);
			case "flags", "Features" -> {
				for (String f : value.split("\\s+"))
					if (!f.isEmpty())
						flags.add(f.toLowerCase(Locale.ROOT));
			}
			default -> {
			}
			}
		}
		return new CpuFacts(model, cores.isEmpty() ? threads : cores.size(), threads, flags);
	}

	/** {@code sysctl} output for the brand string, core counts and feature keys (Intel lists, Arm {@code FEAT_*}). */
	static CpuFacts macSysctl(String text) {
		Map<String, String> kv = keyValues(text, ':');
		Set<String> flags = new HashSet<>();
		for (String key : List.of("machdep.cpu.features", "machdep.cpu.leaf7_features"))
			for (String f : kv.getOrDefault(key, "").split("\\s+"))
				if (!f.isEmpty())
					flags.add(f.toLowerCase(Locale.ROOT));
		for (Map.Entry<String, String> e : kv.entrySet())
			if (e.getKey().startsWith("hw.optional.arm.") && e.getValue().equals("1"))
				flags.add(e.getKey().substring("hw.optional.arm.".length()).toLowerCase(Locale.ROOT));
		return new CpuFacts(kv.getOrDefault("machdep.cpu.brand_string", ""), intOr(kv.get("hw.physicalcpu")),
				intOr(kv.get("hw.logicalcpu")), flags);
	}

	/** {@code Get-CimInstance Win32_Processor | Format-List Name,NumberOfCores,NumberOfLogicalProcessors}. */
	static CpuFacts windowsProcessor(String text) {
		Map<String, String> kv = keyValues(text, ':');
		return new CpuFacts(kv.getOrDefault("Name", ""), intOr(kv.get("NumberOfCores")),
				intOr(kv.get("NumberOfLogicalProcessors")), Set.of());
	}

	/** {@code nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader,nounits}. */
	static List<HostInfo.Gpu> nvidiaSmi(String text) {
		List<HostInfo.Gpu> gpus = new ArrayList<>();
		for (String line : text.split("\n")) {
			String[] f = line.split(",");
			if (f.length < 4)
				continue;
			long mib = longOr(f[1].trim());
			if (mib <= 0)
				continue;
			gpus.add(new HostInfo.Gpu("NVIDIA", f[0].trim(), mib << 20, f[3].trim()));
		}
		return gpus;
	}

	/** {@code rocm-smi --showproductname --showmeminfo vram --csv}: columns located by header name. */
	static List<HostInfo.Gpu> rocmSmi(String text) {
		String[] lines = text.split("\n");
		int header = -1;
		for (int i = 0; i < lines.length && header < 0; i++)
			if (lines[i].startsWith("device,"))
				header = i;
		if (header < 0)
			return List.of();
		List<String> cols = List.of(lines[header].split(","));
		int name = indexOfIgnoreCase(cols, "Card Series");
		int mem = indexOfIgnoreCase(cols, "VRAM Total Memory (B)");
		List<HostInfo.Gpu> gpus = new ArrayList<>();
		for (int i = header + 1; i < lines.length; i++) {
			String[] f = lines[i].split(",");
			if (f.length != cols.size() || !f[0].startsWith("card"))
				continue;
			gpus.add(new HostInfo.Gpu("AMD", name >= 0 ? f[name].trim() : f[0], mem >= 0 ? longOr(f[mem].trim()) : 0,
					""));
		}
		return gpus;
	}

	/**
	 * {@code /sys/class/power_supply/<name>/<attribute>} values per supply. Peripheral batteries (a wireless
	 * keyboard or mouse: {@code scope=Device}) are not the machine's.
	 */
	static HostInfo.Power linuxPower(Map<String, Map<String, String>> supplies) {
		boolean battery = false;
		boolean discharging = false;
		boolean mainsOnline = false;
		int percent = -1;
		for (Map<String, String> s : supplies.values()) {
			String type = s.getOrDefault("type", "");
			if (type.equals("Mains") && s.getOrDefault("online", "0").equals("1"))
				mainsOnline = true;
			if (!type.equals("Battery") || s.getOrDefault("scope", "System").equals("Device"))
				continue;
			battery = true;
			discharging |= s.getOrDefault("status", "").equalsIgnoreCase("Discharging");
			percent = Math.max(percent, intOr(s.get("capacity"), -1));
		}
		if (!battery)
			return NO_BATTERY;
		return new HostInfo.Power(true, discharging && !mainsOnline, percent);
	}

	/** {@code pmset -g batt}. */
	static HostInfo.Power macPmset(String text) {
		boolean onBattery = text.contains("'Battery Power'");
		int percent = -1;
		java.util.regex.Matcher m = java.util.regex.Pattern.compile("(\\d{1,3})%").matcher(text);
		if (m.find())
			percent = Integer.parseInt(m.group(1));
		boolean battery = text.contains("InternalBattery");
		return battery ? new HostInfo.Power(true, onBattery, percent) : NO_BATTERY;
	}

	/** {@code Get-CimInstance Win32_Battery | Format-List BatteryStatus,EstimatedChargeRemaining}; 1 is discharging. */
	static HostInfo.Power windowsBattery(String text) {
		Map<String, String> kv = keyValues(text, ':');
		if (!kv.containsKey("BatteryStatus"))
			return NO_BATTERY;
		return new HostInfo.Power(true, kv.get("BatteryStatus").equals("1"),
				intOr(kv.get("EstimatedChargeRemaining"), -1));
	}

	/** {@code nmcli -t -f GENERAL.METERED dev show}: {@code yes} if any device is metered. */
	static String nmcliMetered(String text) {
		String state = "unknown";
		for (String line : text.split("\n")) {
			if (!line.startsWith("GENERAL.METERED:"))
				continue;
			String v = line.substring("GENERAL.METERED:".length()).trim();
			if (v.startsWith("yes"))
				return "yes";
			if (v.startsWith("no"))
				state = "no";
		}
		return state;
	}

	private static Map<String, String> keyValues(String text, char sep) {
		Map<String, String> kv = new LinkedHashMap<>();
		for (String line : text.split("\r?\n")) {
			int i = line.indexOf(sep);
			if (i > 0)
				kv.putIfAbsent(line.substring(0, i).trim(), line.substring(i + 1).trim());
		}
		return kv;
	}

	private static int indexOfIgnoreCase(List<String> cols, String name) {
		for (int i = 0; i < cols.size(); i++)
			if (cols.get(i).trim().equalsIgnoreCase(name))
				return i;
		return -1;
	}

	private static int intOr(String s) {
		return intOr(s, 0);
	}

	private static int intOr(String s, int fallback) {
		try {
			return s == null ? fallback : Integer.parseInt(s.trim());
		} catch (NumberFormatException e) {
			return fallback;
		}
	}

	private static long longOr(String s) {
		try {
			return Long.parseLong(s.trim());
		} catch (NumberFormatException e) {
			return 0;
		}
	}
}
