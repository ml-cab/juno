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


import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.sun.management.HotSpotDiagnosticMXBean;

import cab.ml.juno.node.CpuThreads;
import cab.ml.juno.node.CudaAvailability;
import cab.ml.juno.node.RocmAvailability;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;

/**
 * What the JVM and Juno see on this machine: the preferred vector widths, the JVM's AVX level, Juno's CPU
 * thread count and its own CUDA and ROCm backend probes, with the CUDA libraries Juno loaded and where from
 * ({@code cudaLibraries}: a file found on {@code PATH} or under {@code CUDA_PATH}, or the name the system
 * loader resolved), or why CUDA is unavailable ({@code cudaUnavailableReason}). Detection runs it before
 * anything is chosen, and every run records it.
 */
final class EnvProbe implements KitProbe {

	@Override
	public ProbeReport run(String id, Map<String, String> args) {
		Map<String, Object> f = new LinkedHashMap<>();
		for (String k : List.of("java.version", "java.vendor", "java.vm.name", "os.name", "os.version", "os.arch"))
			f.put(k, System.getProperty(k));
		f.put("availableProcessors", Runtime.getRuntime().availableProcessors());
		f.put("maxHeapBytes", Runtime.getRuntime().maxMemory());
		f.put("intVectorBits", ByteVector.SPECIES_PREFERRED.vectorBitSize());
		f.put("floatVectorBits", FloatVector.SPECIES_PREFERRED.vectorBitSize());
		f.put("useAvx", vmOption("UseAVX"));
		f.put("cpuThreads", CpuThreads.threads());
		f.put("clockSource", JvmLauncher.clockSource());
		boolean cuda = false;
		List<Map<String, Object>> cudaDevices = new ArrayList<>();
		try {
			cuda = CudaAvailability.isAvailable();
			for (int i = 0; cuda && i < CudaAvailability.deviceCount(); i++)
				cudaDevices.add(Map.of("name", CudaAvailability.deviceName(i), "memoryBytes",
						CudaAvailability.vramBytes(i)));
		} catch (Throwable t) {
			f.put("cudaError", t.toString());
		}
		boolean rocm = false;
		List<Map<String, Object>> rocmDevices = new ArrayList<>();
		try {
			rocm = RocmAvailability.isAvailable();
			for (int i = 0; rocm && i < RocmAvailability.deviceCount(); i++)
				rocmDevices.add(Map.of("name", RocmAvailability.deviceName(i), "memoryBytes",
						RocmAvailability.vramBytes(i)));
		} catch (Throwable t) {
			f.put("rocmError", t.toString());
		}
		f.put("cuda", cuda);
		f.put("cudaDevices", cudaDevices);
		Map<String, String> cudaLibraries = Map.of();
		try {
			cudaLibraries = CudaAvailability.loadedLibraries();
			String reason = CudaAvailability.unavailableReason();
			if (reason != null)
				f.put("cudaUnavailableReason", reason);
		} catch (Throwable t) {
			f.put("cudaUnavailableReason", t.toString());
		}
		f.put("cudaLibraries", cudaLibraries);
		f.put("rocm", rocm);
		f.put("rocmDevices", rocmDevices);
		return new ProbeReport(id, ProbeKind.READING, null, List.of(), List.of(), f);
	}

	private static String vmOption(String name) {
		try {
			return ManagementFactory.getPlatformMXBean(HotSpotDiagnosticMXBean.class).getVMOption(name).getValue();
		} catch (RuntimeException e) {
			return "n/a";
		}
	}
}
