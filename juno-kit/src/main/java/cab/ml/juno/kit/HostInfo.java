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


import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * What the kit detected about the machine. {@code intVectorBits} and {@code floatVectorBits} are the JVM's
 * preferred Vector API widths for bytes and floats; {@code cuda} and {@code rocm} are Juno's own backend
 * probe, run in a child JVM, and are what decides the GPU probes. {@code gpus} is what the vendor tools
 * report, whether or not Juno can use the device.
 */
record HostInfo(String os, String osVersion, String arch, String cpuModel, int cores, int threads, long ramBytes,
		long freeDiskBytes, Set<String> cpuFlags, int intVectorBits, int floatVectorBits, String useAvx, boolean cuda,
		boolean rocm, List<Gpu> gpus, Power power, String metered, String clockSource, Map<String, Object> backend) {

	HostInfo {
		cpuFlags = Set.copyOf(cpuFlags);
		gpus = List.copyOf(gpus);
		backend = Map.copyOf(backend);
	}

	/** One GPU as its vendor tool reports it. */
	record Gpu(String vendor, String name, long memoryBytes, String driver) {
	}

	/** {@code percent} is -1 when unknown or when there is no battery. */
	record Power(boolean hasBattery, boolean onBattery, int percent) {
	}
}
