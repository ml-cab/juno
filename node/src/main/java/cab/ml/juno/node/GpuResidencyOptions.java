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
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Policy for the device-resident decode region ({@code --gpu-residency} /
 * {@code JUNO_GPU_RESIDENCY}): RMS norm, the Q/K/V projection and RoPE run on
 * one activation that stays on the GPU between them, and with GPU attention on,
 * the KV append into the device KV mirror and attention as well, with one
 * download of k, v and the attention output, instead of a host round trip
 * around each.
 *
 * <p>Default {@link Mode#AUTO}: the region runs wherever CUDA is present and the
 * handler and its weights qualify (measured at 1.56x to 1.99x decode where it
 * runs, prefill unchanged); everywhere else today's path is kept. {@link Mode#OFF}
 * keeps the op-at-a-time path, whose greedy output can differ from the region's
 * after some tokens because the region's norms sum in a different order.
 *
 * <p>Surfaces that cannot run the region (another handler family, LoRA, a backend
 * or layout the kernels do not cover) keep today's path. Under an explicit
 * {@link Mode#ON} each says so once, in the log and on the console, so the flag
 * never silently does nothing. Under {@link Mode#AUTO} the console stays quiet,
 * and the CPU backend and LoRA, which the user did not ask the region for, stay
 * out of the log too ({@link #announceIfExplicit}).
 */
public final class GpuResidencyOptions {

	public static final String ENV_PROPERTY = "JUNO_GPU_RESIDENCY";
	public static final String OFF = "off";
	public static final String ON = "on";
	public static final String AUTO = "auto";

	public enum Mode {
		OFF, ON, AUTO
	}

	private static final Set<String> ANNOUNCED = ConcurrentHashMap.newKeySet();

	private final Mode mode;

	private GpuResidencyOptions(Mode mode) {
		this.mode = mode;
	}

	/**
	 * Parse CLI / env value: {@code on|off|auto}.
	 *
	 * @param spec raw value; blank means {@code auto}
	 */
	public static GpuResidencyOptions parse(String spec) {
		if (spec == null || spec.isBlank())
			return new GpuResidencyOptions(Mode.AUTO);
		String s = spec.strip().toLowerCase(Locale.ROOT);
		return switch (s) {
		case OFF, "0", "false", "no" -> new GpuResidencyOptions(Mode.OFF);
		case ON, "1", "true", "yes" -> new GpuResidencyOptions(Mode.ON);
		case AUTO -> new GpuResidencyOptions(Mode.AUTO);
		default -> throw new IllegalArgumentException("--gpu-residency must be on|off|auto (got " + spec + ")");
		};
	}

	/** Reads {@link #ENV_PROPERTY} (system property, falling back to the env var); defaults to {@code auto}. */
	public static GpuResidencyOptions fromEnv() {
		String prop = System.getProperty(ENV_PROPERTY);
		String raw = prop != null && !prop.isBlank() ? prop : System.getenv(ENV_PROPERTY);
		return parse(raw);
	}

	public Mode mode() {
		return mode;
	}

	/** Whether this process asked for the resident region: {@code on}, or {@code auto} with CUDA present. */
	public boolean requested() {
		return switch (mode) {
		case OFF -> false;
		case ON -> true;
		case AUTO -> CudaAvailability.isAvailable();
		};
	}

	/** Label for logs: {@code on}, {@code off}, or {@code auto(on)} / {@code auto(off)}. */
	public String policyLabel() {
		return mode == Mode.AUTO ? "auto(" + (requested() ? "on" : "off") + ")" : mode.name().toLowerCase(Locale.ROOT);
	}

	/**
	 * Why the device region cannot run a model of this {@code general.architecture}
	 * at all, or {@code null} when a handler with the region serves it: the
	 * LLaMA-family handler with adjacent RoPE, or the Phi-3 or dense Qwen3 handler
	 * (whether each layer then qualifies depends on its weights).
	 */
	public static String unsupportedArchitectureReason(String architecture) {
		String a = architecture == null ? "" : architecture.strip().toLowerCase(Locale.ROOT);
		return switch (a) {
		case "qwen2", "qwen2.5" -> "uses the split-half RoPE layout (and Q/K/V biases), which the device region does not implement";
		case "phi2", "qwen3moe" -> "is served by a handler without a device-resident decode region";
		default -> null;
		};
	}

	/**
	 * The one-line notice a console front end prints at startup when the region was
	 * explicitly requested ({@code on}) where it will not run, or {@code null} when it
	 * was not, under {@code auto}, or when nothing about this launch rules it out.
	 * Console front ends disable library logging unless asked to be verbose, so the
	 * log line alone would be silent.
	 */
	public static String consoleNotice(String architecture, boolean lora, boolean cpu) {
		GpuResidencyOptions opts = fromEnv();
		if (opts.mode() != Mode.ON)
			return null;
		String prefix = "--gpu-residency=" + opts.policyLabel() + " has no effect here: ";
		if (lora)
			return prefix + "LoRA training and --lora-play apply adapters to Q/K/V on the host; the existing path is used";
		if (cpu)
			return prefix + "the CPU backend has no device to keep activations on; the existing path is used";
		String reason = unsupportedArchitectureReason(architecture);
		if (reason != null)
			return prefix + "architecture " + architecture + " " + reason + "; the existing path is used";
		if (GpuAttentionOptions.fromEnv().mode() == GpuAttentionOptions.Mode.OFF)
			return "--gpu-residency=" + opts.policyLabel() + " with --gpu-attention off: the KV append, attention"
					+ " and the rest of the layer stay outside the device region (attention runs on the CPU); norm,"
					+ " Q/K/V projection and RoPE still run in it";
		return null;
	}

	/**
	 * Returns {@code true} the first time {@code surface} is passed in this
	 * process and {@code false} after, so a notice about a surface that cannot
	 * run the region is logged once rather than per handler or per layer.
	 */
	static boolean announceOnce(String surface) {
		return ANNOUNCED.add(surface);
	}

	/**
	 * Logs, once per {@code surface}, that the region was requested where it
	 * cannot run and today's path is used. A no-op when the region was not
	 * requested.
	 */
	static void announceUnsupported(java.util.logging.Logger log, String surface, String reason) {
		GpuResidencyOptions opts = fromEnv();
		if (!opts.requested() || !announceOnce(surface))
			return;
		log.warning("--gpu-residency=" + opts.policyLabel() + " requested, but " + surface + " " + reason
				+ "; using the existing path there");
	}

	/**
	 * {@link #announceUnsupported} for a surface the default never meant to run the
	 * region on (the CPU backend, LoRA): logged only under an explicit {@code on}.
	 */
	static void announceIfExplicit(java.util.logging.Logger log, String surface, String reason) {
		if (fromEnv().mode() == Mode.ON)
			announceUnsupported(log, surface, reason);
	}
}
