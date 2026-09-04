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
 * Policy for fused Q4_K device-resident matmul ({@code --mmq} / {@code JUNO_MMQ}).
 *
 * <p>When enabled on CUDA, Q4_K projection weights stay packed on the device and
 * a fused dequant+GEMV kernel replaces the FP16-resident cuBLAS path. Default is
 * {@link Mode#OFF} until bake-off gates pass.
 */
public final class MmqOptions {

	public static final String ENV_PROPERTY = "JUNO_MMQ";
	public static final String OFF = "off";
	public static final String ON = "on";
	public static final String AUTO = "auto";

	public enum Mode {
		OFF, ON, AUTO
	}

	private final Mode mode;

	private MmqOptions(Mode mode) {
		this.mode = mode;
	}

	public static MmqOptions off() {
		return new MmqOptions(Mode.OFF);
	}

	public static MmqOptions on() {
		return new MmqOptions(Mode.ON);
	}

	public static MmqOptions auto() {
		return new MmqOptions(Mode.AUTO);
	}

	/**
	 * Parse CLI / env value: {@code on|off|auto}.
	 *
	 * @param spec raw value; blank → {@code off}
	 */
	public static MmqOptions parse(String spec) {
		if (spec == null || spec.isBlank())
			return off();
		String s = spec.strip().toLowerCase(Locale.ROOT);
		return switch (s) {
		case OFF, "0", "false", "no" -> off();
		case ON, "1", "true", "yes" -> on();
		case AUTO -> auto();
		default -> throw new IllegalArgumentException(
				"--mmq must be on|off|auto (got " + spec + ")");
		};
	}

	/** Reads {@link #ENV_PROPERTY}; defaults to {@code off}. */
	public static MmqOptions fromEnv() {
		return parse(System.getProperty(ENV_PROPERTY, OFF));
	}

	public Mode mode() {
		return mode;
	}

	/**
	 * Whether the fused Q4_K path should be attempted for this process.
	 *
	 * <p>{@code AUTO} enables when CUDA runtime is present; kernel load failures
	 * fall back to FP16-resident at upload time.
	 */
	public boolean preferMmq() {
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
