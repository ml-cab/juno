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

import jdk.incubator.vector.ByteVector;

/**
 * The lane width of the CPU integer kernels, chosen once per JVM.
 *
 * <p>By default ({@code auto}) it is 128 bits on every host: the 256-bit
 * kernels are correct everywhere but not yet measured on a host with 256-bit
 * integer instructions (AVX2), so they run only on request.
 * {@code -Djuno.cpu.integerWidth=128|256|auto} forces it; any other value fails
 * at first use with the property named. The host's preferred integer width
 * ({@code ByteVector.SPECIES_PREFERRED}) is reported beside the choice. The 128-bit kernels
 * ({@link CpuIntegerKernels}, {@link CpuIntegerGemm}) are the reference the
 * 256-bit ones ({@link CpuIntegerKernels256}, {@link CpuIntegerGemm256}) are
 * tested against. Forcing 256 on a host without 256-bit integer instructions
 * runs the Vector API's portable form: correct, and much slower.
 *
 * <p>Loading this class needs {@code jdk.incubator.vector}; only the integer
 * kernels and a caller that checked {@link VectorQuantKernels#AVAILABLE} load it.
 */
final class CpuIntegerWidth {

	static final String PROPERTY = "juno.cpu.integerWidth";

	/** Selected lane width in bits: 128 or 256. */
	static final int BITS = select(System.getProperty(PROPERTY), preferredBits());

	/** True when the 256-bit kernels run. A constant, so the dispatch folds away. */
	static final boolean WIDE = BITS == 256;

	private CpuIntegerWidth() {
	}

	/** The host's preferred integer vector width in bits. */
	static int preferredBits() {
		return ByteVector.SPECIES_PREFERRED.vectorBitSize();
	}

	/**
	 * The width for a forced setting ({@code null}, blank or {@code auto} for
	 * none) on a host preferring {@code preferredBits}. {@code auto} is 128 at any
	 * preferred width until the 256-bit kernels are measured on such a host.
	 *
	 * @throws IllegalArgumentException for a setting other than auto, 128 or 256
	 */
	static int select(String forced, int preferredBits) {
		String f = forced == null ? "" : forced.trim();
		if (f.isEmpty() || f.equalsIgnoreCase("auto"))
			return 128;
		return switch (f) {
		case "128" -> 128;
		case "256" -> 256;
		default -> throw new IllegalArgumentException(
				"-D" + PROPERTY + " must be auto, 128 or 256, got '" + forced + "'");
		};
	}

	/** One line for startup logs. */
	static String summary() {
		String forced = System.getProperty(PROPERTY);
		boolean isForced = forced != null && !forced.isBlank() && !forced.trim().equalsIgnoreCase("auto");
		return "CPU integer kernels: " + BITS + "-bit lanes (host preferred " + preferredBits() + "-bit"
				+ (isForced ? ", forced by -D" + PROPERTY : "") + ")";
	}
}
