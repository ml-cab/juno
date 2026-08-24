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
 * LoRA train-device mode: CLI/env {@code auto|gpu|cpu} and MatVec selection.
 *
 * <ul>
 * <li>{@code cpu} — force {@link CpuMatVec}
 * <li>{@code gpu} — require a live CUDA/ROCm MatVec; fail closed if unavailable
 * <li>{@code auto} — current LoRA default via {@link ForwardPassHandlerLoader#selectLoraBackend()}
 * </ul>
 */
public final class LoraTrainDevice {

	public static final String AUTO = "auto";
	public static final String GPU = "gpu";
	public static final String CPU = "cpu";

	private LoraTrainDevice() {
	}

	/**
	 * @param mode raw CLI/env value; blank/null → {@link #AUTO}
	 * @return normalized {@code auto|gpu|cpu}
	 */
	public static String normalize(String mode) {
		if (mode == null || mode.isBlank())
			return AUTO;
		String m = mode.strip().toLowerCase(Locale.ROOT);
		return switch (m) {
		case AUTO, GPU, CPU -> m;
		// Resolved JFR labels may appear if a config is reused after open.
		case "cuda", "rocm" -> GPU;
		default -> throw new IllegalArgumentException(
				"--lora-train-device must be auto|gpu|cpu (got " + mode + ")");
		};
	}

	/**
	 * Select the MatVec for LoRA training/playback according to {@code mode}.
	 *
	 * <p>{@code gpu} ignores {@code JUNO_USE_GPU=false}: the mode is an explicit
	 * requirement. {@code auto} still honors {@code JUNO_USE_GPU} / {@code --cpu}.
	 */
	public static MatVec selectBackend(String mode) {
		String m = normalize(mode);
		System.setProperty("juno.lora.train.device", m);
		return switch (m) {
		case CPU -> CpuMatVec.INSTANCE;
		case GPU -> ForwardPassHandlerLoader.requireGpuLoraBackend();
		case AUTO -> ForwardPassHandlerLoader.selectLoraBackendAuto();
		default -> throw new IllegalArgumentException("--lora-train-device must be auto|gpu|cpu (got " + mode + ")");
		};
	}

	/** Stable JFR/metrics label for the resolved backend. */
	public static String labelFor(MatVec backend) {
		if (backend instanceof CudaMatVec)
			return "cuda";
		if (backend instanceof RocmMatVec)
			return "rocm";
		return "cpu";
	}

	/** Whether this mode requires resident GPU transpose (fail closed on upload OOM). */
	public static boolean requireResident(String mode) {
		return GPU.equals(normalize(mode));
	}
}
