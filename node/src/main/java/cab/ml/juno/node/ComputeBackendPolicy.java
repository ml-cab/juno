/*
 * Created by Yevhen Soldatov
 * Initial implementation: 2026
 *
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
 * Parses {@code JUNO_USE_GPU} / CLI tokens and decides whether to attempt CUDA
 * matmul for this process.
 */
public final class ComputeBackendPolicy {

	private ComputeBackendPolicy() {
	}

	/**
	 * Parses a raw string from {@code -DJUNO_USE_GPU}, environment, or config.
	 *
	 * @param raw nullable; blank → {@link ComputeBackendPreference#AUTO}
	 */
	public static ComputeBackendPreference parsePreference(String raw) {
		if (raw == null)
			return ComputeBackendPreference.AUTO;
		String s = raw.strip().toLowerCase();
		if (s.isEmpty())
			return ComputeBackendPreference.AUTO;
		if (isDisableToken(s))
			return ComputeBackendPreference.DISABLE;
		if (isRequireToken(s))
			return ComputeBackendPreference.REQUIRE;
		if (isAutoToken(s) || isLegacyEnableToken(s))
			return ComputeBackendPreference.AUTO;
		return ComputeBackendPreference.AUTO;
	}

	private static boolean isDisableToken(String s) {
		return s.equals("false") || s.equals("0") || s.equals("no") || s.equals("off") || s.equals("cpu")
				|| s.equals("disable") || s.equals("disabled");
	}

	/** Legacy: {@code true} meant “try GPU if present” → {@link ComputeBackendPreference#AUTO}. */
	private static boolean isLegacyEnableToken(String s) {
		return s.equals("true") || s.equals("1") || s.equals("yes") || s.equals("on") || s.equals("enable")
				|| s.equals("enabled");
	}

	private static boolean isAutoToken(String s) {
		return s.equals("auto");
	}

	private static boolean isRequireToken(String s) {
		return s.equals("require") || s.equals("required") || s.equals("force") || s.equals("mandatory")
				|| s.equals("gpu-only");
	}

	/**
	 * Whether this JVM should use {@link CudaMatVec} when loading real shards
	 * (CUDA must also be available).
	 */
	public static boolean useCudaMatmul(ComputeBackendPreference preference) {
		return preference != ComputeBackendPreference.DISABLE && CudaAvailability.isAvailable();
	}

	/**
	 * Throws if {@link ComputeBackendPreference#REQUIRE} is set but CUDA is missing.
	 */
	public static void validateRequireSatisfied(ComputeBackendPreference preference) {
		if (preference == ComputeBackendPreference.REQUIRE && !CudaAvailability.isAvailable())
			throw new IllegalStateException(
					"CUDA is required (JUNO_USE_GPU=require) but no CUDA device is available on this machine.");
	}
}
