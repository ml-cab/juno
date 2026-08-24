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
 * LoRA frozen-linear microbatch width: CLI/env {@code --lora-microbatch} /
 * {@code LORA_MICROBATCH} and runtime property {@code juno.lora.microbatch}.
 *
 * <p>{@code > 1} uploads FP32 residency for {@link GpuBlasOps} GEMM; {@code 1}
 * uses sequential GEMV (FP16 when the backend supports half residency).
 *
 * @author Yevhen Soldatov
 */
public final class LoraMicrobatch {

	public static final int DEFAULT = 8;
	public static final int MAX = 128;
	public static final String PROPERTY = "juno.lora.microbatch";

	private LoraMicrobatch() {
	}

	/**
	 * @param n microbatch width
	 * @return {@code n} when in {@code [1, MAX]}
	 * @throws IllegalArgumentException when out of range
	 */
	public static int validate(int n) {
		if (n < 1 || n > MAX)
			throw new IllegalArgumentException(
					"--lora-microbatch must be 1.." + MAX + " (got " + n + ")");
		return n;
	}

	/**
	 * @param raw CLI/env value; blank/null → {@link #DEFAULT}
	 * @return validated microbatch width
	 */
	public static int normalize(String raw) {
		if (raw == null || raw.isBlank())
			return DEFAULT;
		try {
			return validate(Integer.parseInt(raw.strip()));
		} catch (NumberFormatException e) {
			throw new IllegalArgumentException(
					"--lora-microbatch must be an integer 1.." + MAX + " (got " + raw + ")");
		}
	}

	/** Publish {@code n} as the runtime source of truth for handlers. */
	public static void apply(int n) {
		System.setProperty(PROPERTY, Integer.toString(validate(n)));
	}

	/**
	 * Current microbatch width from {@link #PROPERTY}. Unset → {@link #DEFAULT}.
	 * Values below 1 (raw {@code -D}) clamp to 1.
	 */
	public static int current() {
		int n = Integer.getInteger(PROPERTY, DEFAULT);
		return Math.max(1, n);
	}
}
