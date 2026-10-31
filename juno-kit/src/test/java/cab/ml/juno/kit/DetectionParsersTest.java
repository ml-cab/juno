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

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

/**
 * The Linux CPU and NVIDIA fixtures are recorded on the reference host (Xeon E5-1650 v2, GTX 1080). The
 * macOS, Windows and ROCm samples follow each tool's documented output format; they are replaced by
 * recordings from the first kit runs on those systems.
 */
class DetectionParsersTest {

	private static String resource(String name) throws IOException {
		try (InputStream in = DetectionParsersTest.class.getResourceAsStream("/detection/" + name)) {
			return new String(in.readAllBytes(), StandardCharsets.UTF_8);
		}
	}

	@Test
	void linuxCpuInfoGivesModelPhysicalCoresThreadsAndFlags() throws IOException {
		DetectionParsers.CpuFacts cpu = DetectionParsers.linuxCpuInfo(resource("linux-cpuinfo-xeon-e5-1650v2.txt"));
		assertThat(cpu.model()).isEqualTo("Intel(R) Xeon(R) CPU E5-1650 v2 @ 3.50GHz");
		assertThat(cpu.cores()).isEqualTo(6);
		assertThat(cpu.threads()).isEqualTo(12);
		assertThat(cpu.flags()).contains("avx", "sse4_2").doesNotContain("avx2", "avx512f");
	}

	@Test
	void linuxCpuInfoOnArmHasNoModelNameLineButStillCountsCpus() {
		String arm = """
				processor\t: 0
				BogoMIPS\t: 48.00
				Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics asimddp
				CPU implementer\t: 0x41

				processor\t: 1
				BogoMIPS\t: 48.00
				Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics asimddp
				CPU implementer\t: 0x41
				""";
		DetectionParsers.CpuFacts cpu = DetectionParsers.linuxCpuInfo(arm);
		assertThat(cpu.threads()).isEqualTo(2);
		assertThat(cpu.cores()).isEqualTo(2);
		assertThat(cpu.flags()).contains("asimddp");
	}

	@Test
	void nvidiaSmiGivesNameMemoryAndDriver() throws IOException {
		List<HostInfo.Gpu> gpus = DetectionParsers.nvidiaSmi(resource("nvidia-smi-gtx1080.csv"));
		assertThat(gpus).singleElement().satisfies(g -> {
			assertThat(g.vendor()).isEqualTo("NVIDIA");
			assertThat(g.name()).isEqualTo("NVIDIA GeForce GTX 1080");
			assertThat(g.memoryBytes()).isEqualTo(8192L << 20);
			assertThat(g.driver()).isEqualTo("580.173.02");
		});
		assertThat(DetectionParsers.nvidiaSmi("")).isEmpty();
		assertThat(DetectionParsers.nvidiaSmi("NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver."))
				.isEmpty();
	}

	@Test
	void rocmSmiCsvGivesNameAndMemory() {
		String csv = """
				device,Card Series,Card Model,Card Vendor,Card SKU,VRAM Total Memory (B),VRAM Total Used Memory (B)
				card0,Radeon RX 7900 XTX,0x744c,Advanced Micro Devices Inc. [AMD/ATI],D7070100,25753026560,1073741824
				""";
		assertThat(DetectionParsers.rocmSmi(csv)).singleElement().satisfies(g -> {
			assertThat(g.vendor()).isEqualTo("AMD");
			assertThat(g.name()).isEqualTo("Radeon RX 7900 XTX");
			assertThat(g.memoryBytes()).isEqualTo(25753026560L);
		});
		assertThat(DetectionParsers.rocmSmi("")).isEmpty();
	}

	@Test
	void macSysctlOnIntelAndAppleSilicon() {
		String intel = """
				machdep.cpu.brand_string: Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz
				machdep.cpu.features: FPU VME DE PSE TSC MSR PAE SSE SSE2 SSE3 SSSE3 FMA SSE4.1 SSE4.2 AVX1.0
				machdep.cpu.leaf7_features: RDWRFSGS TSC_THREAD_OFFSET SGX BMI1 HLE AVX2 SMEP BMI2
				hw.physicalcpu: 8
				hw.logicalcpu: 16
				hw.memsize: 34359738368
				""";
		DetectionParsers.CpuFacts i = DetectionParsers.macSysctl(intel);
		assertThat(i.model()).startsWith("Intel(R) Core(TM) i9-9980HK");
		assertThat(i.cores()).isEqualTo(8);
		assertThat(i.threads()).isEqualTo(16);
		assertThat(i.flags()).contains("avx2", "sse4.2");

		String arm = """
				machdep.cpu.brand_string: Apple M2
				hw.physicalcpu: 8
				hw.logicalcpu: 8
				hw.optional.arm.FEAT_DotProd: 1
				hw.optional.arm.FEAT_I8MM: 0
				""";
		DetectionParsers.CpuFacts a = DetectionParsers.macSysctl(arm);
		assertThat(a.model()).isEqualTo("Apple M2");
		assertThat(a.flags()).contains("feat_dotprod").doesNotContain("feat_i8mm");
	}

	@Test
	void windowsProcessorFormatList() {
		String out = """

				Name                      : AMD Ryzen 9 7950X 16-Core Processor
				NumberOfCores             : 16
				NumberOfLogicalProcessors : 32

				""";
		DetectionParsers.CpuFacts cpu = DetectionParsers.windowsProcessor(out);
		assertThat(cpu.model()).isEqualTo("AMD Ryzen 9 7950X 16-Core Processor");
		assertThat(cpu.cores()).isEqualTo(16);
		assertThat(cpu.threads()).isEqualTo(32);
	}

	@Test
	void aPeripheralBatteryIsNotTheMachineRunningOnBattery() {
		// Recorded on the reference host: a wireless keyboard reports itself as a battery.
		Map<String, Map<String, String>> supplies = Map.of("hidpp_battery_0",
				Map.of("type", "Battery", "scope", "Device", "status", "Unknown", "online", "0"));
		assertThat(DetectionParsers.linuxPower(supplies)).isEqualTo(new HostInfo.Power(false, false, -1));
	}

	@Test
	void aLaptopOnBatteryAndOnMains() {
		Map<String, Map<String, String>> onBattery = Map.of("BAT0",
				Map.of("type", "Battery", "status", "Discharging", "capacity", "54"), "AC",
				Map.of("type", "Mains", "online", "0"));
		assertThat(DetectionParsers.linuxPower(onBattery)).isEqualTo(new HostInfo.Power(true, true, 54));
		Map<String, Map<String, String>> charging = Map.of("BAT0",
				Map.of("type", "Battery", "status", "Charging", "capacity", "80"), "AC",
				Map.of("type", "Mains", "online", "1"));
		assertThat(DetectionParsers.linuxPower(charging)).isEqualTo(new HostInfo.Power(true, false, 80));
	}

	@Test
	void macPmsetAndWindowsBattery() {
		assertThat(DetectionParsers.macPmset("""
				Now drawing from 'Battery Power'
				 -InternalBattery-0 (id=4653155)	85%; discharging; 5:12 remaining present: true
				""")).isEqualTo(new HostInfo.Power(true, true, 85));
		assertThat(DetectionParsers.macPmset("Now drawing from 'AC Power'\n"))
				.isEqualTo(new HostInfo.Power(false, false, -1));
		assertThat(DetectionParsers.windowsBattery("""

				BatteryStatus            : 1
				EstimatedChargeRemaining : 47

				""")).isEqualTo(new HostInfo.Power(true, true, 47));
		assertThat(DetectionParsers.windowsBattery("")).isEqualTo(new HostInfo.Power(false, false, -1));
	}

	@Test
	void nmcliMeteredState() {
		assertThat(DetectionParsers.nmcliMetered("GENERAL.METERED:no (guessed)\n\nGENERAL.METERED:no (guessed)\n"))
				.isEqualTo("no");
		assertThat(DetectionParsers.nmcliMetered("GENERAL.METERED:no\nGENERAL.METERED:yes (guessed)\n"))
				.isEqualTo("yes");
		assertThat(DetectionParsers.nmcliMetered("")).isEqualTo("unknown");
	}
}
