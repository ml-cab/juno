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


/**
 * What a probe needs from the host beyond a JVM.
 *
 * @param anyGpu           a GPU Juno can use, CUDA or ROCm
 * @param minIntVectorBits the JVM's preferred integer vector width must be at least this (0: any)
 * @param minDeviceBytes   the largest GPU's memory, as its vendor tool reports it, must be at least this (0:
 *                         any); when no tool reports a size the probe is not held back
 */
record Requirements(boolean cuda, boolean rocm, boolean anyGpu, int minIntVectorBits, long minRamBytes,
		long minDeviceBytes) {

	static Requirements cpu(long minRamBytes) {
		return new Requirements(false, false, false, 0, minRamBytes, 0);
	}

	static Requirements cuda(long minRamBytes) {
		return new Requirements(true, false, false, 0, minRamBytes, 0);
	}

	static Requirements rocm(long minRamBytes) {
		return new Requirements(false, true, false, 0, minRamBytes, 0);
	}

	static Requirements gpu(long minRamBytes) {
		return new Requirements(false, false, true, 0, minRamBytes, 0);
	}

	static Requirements intVectorBits(int bits) {
		return new Requirements(false, false, false, bits, 0, 0);
	}

	Requirements withDeviceMemory(long bytes) {
		return new Requirements(cuda, rocm, anyGpu, minIntVectorBits, minRamBytes, bytes);
	}

	boolean needsGpu() {
		return cuda || rocm || anyGpu;
	}

	/** Why {@code host} cannot run a probe with these requirements, or {@code null} when it can. */
	String unmetBy(HostInfo host) {
		if (cuda && !host.cuda())
			return "CUDA: Juno found no usable NVIDIA GPU";
		if (rocm && !host.rocm())
			return "ROCm: Juno found no usable AMD GPU";
		if (anyGpu && !host.cuda() && !host.rocm())
			return "GPU: Juno found no usable NVIDIA or AMD GPU";
		if (minIntVectorBits > 0 && host.intVectorBits() < minIntVectorBits)
			return minIntVectorBits + "-bit integer vectors: this JVM prefers " + host.intVectorBits() + "-bit";
		if (minRamBytes > 0 && host.ramBytes() > 0 && host.ramBytes() < minRamBytes)
			return "memory: needs " + (minRamBytes >> 30) + " GiB, the machine has " + (host.ramBytes() >> 30)
					+ " GiB";
		long device = host.gpus().stream().mapToLong(HostInfo.Gpu::memoryBytes).max().orElse(0);
		if (minDeviceBytes > 0 && device > 0 && device < minDeviceBytes)
			return "device memory: needs " + (minDeviceBytes >> 30) + " GiB, the largest GPU has " + (device >> 30)
					+ " GiB";
		return null;
	}
}
