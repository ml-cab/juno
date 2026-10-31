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


import java.util.Arrays;

import cab.ml.juno.node.CpuDecodeKernelVariantMicrobench;

/**
 * One timing of the shipped decode kernel at the integer width and thread count this JVM was started
 * with ({@code -Djuno.cpu.integerWidth}, {@code -Djuno.threads}); both are fixed at class initialisation,
 * so {@link KernelWidthProbe} starts one of these per reading. Prints {@code ms_per_call=<median>}.
 * Arguments: {@code type rows cols reps targetMs warmupMs}.
 */
public final class KernelTimingMain {

	private KernelTimingMain() {
	}

	public static void main(String[] args) {
		double[] ms = CpuDecodeKernelVariantMicrobench.productionMsPerCall(Integer.parseInt(args[0]),
				Integer.parseInt(args[1]), Integer.parseInt(args[2]), Integer.parseInt(args[3]), Long.parseLong(args[4]),
				Long.parseLong(args[5]));
		Arrays.sort(ms);
		System.out.println("ms_per_call=" + ms[ms.length / 2]);
		System.exit(0);
	}
}
