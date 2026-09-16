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

import java.util.Locale;

/**
 * Policy for the GPU-resident attention kernel ({@code --gpu-attention} /
 * {@code JUNO_GPU_ATTENTION}).
 *
 * <p>When enabled on CUDA with GPU-resident layers, attention (QK^T + softmax +
 * weighted-V-sum) runs on-device against a device-resident KV cache mirror
 * instead of scalar CPU Java. Default is {@link Mode#OFF} until bake-off gates
 * pass.
 */
public final class GpuAttentionOptions {

	public static final String ENV_PROPERTY = "JUNO_GPU_ATTENTION";
	public static final String OFF = "off";
	public static final String ON = "on";
	public static final String AUTO = "auto";

	public enum Mode {
		OFF, ON, AUTO
	}

	private final Mode mode;

	private GpuAttentionOptions(Mode mode) {
		this.mode = mode;
	}

	public static GpuAttentionOptions off() {
		return new GpuAttentionOptions(Mode.OFF);
	}

	public static GpuAttentionOptions on() {
		return new GpuAttentionOptions(Mode.ON);
	}

	public static GpuAttentionOptions auto() {
		return new GpuAttentionOptions(Mode.AUTO);
	}

	/**
	 * Parse CLI / env value: {@code on|off|auto}.
	 *
	 * @param spec raw value; blank → {@code off}
	 */
	public static GpuAttentionOptions parse(String spec) {
		if (spec == null || spec.isBlank())
			return off();
		String s = spec.strip().toLowerCase(Locale.ROOT);
		return switch (s) {
		case OFF, "0", "false", "no" -> off();
		case ON, "1", "true", "yes" -> on();
		case AUTO -> auto();
		default -> throw new IllegalArgumentException(
				"--gpu-attention must be on|off|auto (got " + spec + ")");
		};
	}

	/** Reads {@link #ENV_PROPERTY} (system property, falling back to the env var); defaults to {@code off}. */
	public static GpuAttentionOptions fromEnv() {
		String raw = firstNonBlank(System.getProperty(ENV_PROPERTY), System.getenv(ENV_PROPERTY));
		return parse(raw == null ? OFF : raw);
	}

	private static String firstNonBlank(String a, String b) {
		if (a != null && !a.isBlank())
			return a;
		if (b != null && !b.isBlank())
			return b;
		return null;
	}

	public Mode mode() {
		return mode;
	}

	/**
	 * Whether the GPU-resident attention path should be attempted for this process.
	 *
	 * <p>{@code AUTO} enables when CUDA runtime is present; kernel load failures
	 * fall back to scalar CPU attention.
	 */
	public boolean preferGpuAttention() {
		return switch (mode) {
		case OFF -> false;
		case ON -> true;
		case AUTO -> CudaAvailability.isAvailable();
		};
	}

	public String policyLabel() {
		return mode.name().toLowerCase(Locale.ROOT);
	}
}
