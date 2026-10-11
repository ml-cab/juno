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
package cab.ml.juno.node;

/**
 * Chooses between the integer-dot CPU kernels ({@link CpuIntegerKernels},
 * {@link CpuIntegerGemm}) and the float-activation kernels for a quantized CPU
 * matmul.
 *
 * <p>This class references no {@code jdk.incubator.vector} type, so asking it
 * never loads the Vector API. The integer kernels need that module; on a JVM
 * started without {@code --add-modules jdk.incubator.vector} (an embedder's, or
 * a test module that does not pass it) {@link VectorQuantKernels#AVAILABLE} is
 * false and every matmul takes the float kernels, which give the same result
 * they gave before the integer kernels existed.
 *
 * <p>{@code -Djuno.cpu.integerKernels=false} selects the float kernels
 * explicitly, for comparison.
 */
final class CpuKernelSelection {

	private static volatile boolean integerEnabled = !"false"
			.equalsIgnoreCase(System.getProperty("juno.cpu.integerKernels"));

	static {
		java.util.logging.Logger.getLogger(CpuKernelSelection.class.getName()).info(summary());
	}

	private CpuKernelSelection() {
	}

	/** True for the GGUF tensor types the integer kernels implement: Q8_0, Q4_K, Q5_K, Q6_K. */
	static boolean integerFormat(int ggmlType) {
		return ggmlType == 8 || ggmlType == 12 || ggmlType == 13 || ggmlType == 14;
	}

	/** True when a CPU matVec over {@code ggmlType} takes the integer-dot kernel. */
	static boolean integerMatVec(int ggmlType) {
		return integerEnabled && VectorQuantKernels.AVAILABLE && integerFormat(ggmlType);
	}

	/** True when a batched CPU matmul over {@code ggmlType} at width {@code cols} takes the integer matmul. */
	static boolean integerGemm(int ggmlType, int cols) {
		return integerMatVec(ggmlType) && cols % 256 == 0;
	}

	/** Switches the integer kernels on or off for the whole JVM; returns the previous setting. For tests. */
	static boolean setIntegerEnabled(boolean on) {
		boolean was = integerEnabled;
		integerEnabled = on;
		return was;
	}

	/** One line for startup logs: which CPU matmul kernels run, and why. */
	static String summary() {
		if (!integerEnabled)
			return "CPU quantized matmul: float-activation kernels (juno.cpu.integerKernels=false)";
		if (!VectorQuantKernels.AVAILABLE)
			return "CPU quantized matmul: float-activation kernels (jdk.incubator.vector unavailable)";
		return "CPU quantized matmul: integer-dot kernels for Q4_K, Q5_K, Q6_K, Q8_0";
	}
}
