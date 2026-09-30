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
import java.util.logging.Logger;

/**
 * Which handlers run the GPU-resident attention kernel ({@code --gpu-attention}),
 * and what a launch says when the kernel cannot take effect.
 *
 * <p>The kernel is CUDA-only. It is wired into the LLaMA-family handler (Llama,
 * Mistral, TinyLlama, Qwen2), the Phi-3 handler and the Qwen3 handler, each of
 * which reports it through {@link ForwardPassHandler#gpuAttentionActive()}. The
 * Phi-2 and Qwen3-MoE handlers compute every matmul and attention on the CPU on
 * any backend, so neither the kernel nor {@code --gpu-layers} reaches them.
 *
 * <p>Nothing resolves to the CPU path silently: a GPU launch of a Phi-2 or
 * Qwen3-MoE model, a request for the kernel on a backend other than CUDA, and an
 * explicit {@code on} with the CPU backend each produce a notice. Console front
 * ends print {@link #consoleNotice} (they turn library logging off unless
 * verbose); every other entry point gets the once-per-process log line.
 */
public final class GpuAttentionSupport {

	private static final Logger log = Logger.getLogger(GpuAttentionSupport.class.getName());

	/** Architectures whose handler computes on the CPU whatever backend it is given. */
	private static final Set<String> CPU_ONLY_HANDLERS = Set.of("phi2", "qwen3moe");

	private static final Set<String> ANNOUNCED = ConcurrentHashMap.newKeySet();

	private GpuAttentionSupport() {
	}

	/**
	 * Whether the handler for this {@code general.architecture} can run the GPU
	 * attention kernel. Meaningful for an architecture the loader accepts; the
	 * loader rejects any other.
	 */
	public static boolean handlerRunsKernel(String architecture) {
		return !CPU_ONLY_HANDLERS.contains(normalize(architecture));
	}

	/**
	 * Whether a GPU launch in this process will select the CUDA backend: forced by
	 * {@code -Djuno.gpu.backend=cuda}, excluded by {@code rocm}, and otherwise
	 * whenever CUDA is present, which is the order {@code GpuContext} tries.
	 */
	public static boolean cudaBackendExpected() {
		String pref = System.getProperty("juno.gpu.backend", "auto").strip().toLowerCase(Locale.ROOT);
		return switch (pref) {
		case "cuda" -> true;
		case "rocm" -> false;
		default -> CudaAvailability.isAvailable();
		};
	}

	/**
	 * The one-line notice a console front end prints at startup when this launch
	 * cannot run what it asked for, or {@code null} when there is nothing to say.
	 *
	 * @param architecture {@code general.architecture} of the model, or {@code null}
	 * @param lora         LoRA training or {@code --lora-play}; those carry their own
	 *                     notice ({@code LoraTrainNotices}), so this returns {@code null}
	 * @param cpu          the launch uses the CPU backend
	 * @param cudaBackend  a GPU launch selects CUDA ({@link #cudaBackendExpected()})
	 */
	public static String consoleNotice(String architecture, boolean lora, boolean cpu, boolean cudaBackend) {
		if (lora)
			return null;
		String arch = normalize(architecture);
		if (!cpu && CPU_ONLY_HANDLERS.contains(arch))
			return cpuOnlyHandlerNotice(arch);
		GpuAttentionOptions opts = GpuAttentionOptions.fromEnv();
		if (opts.mode() == GpuAttentionOptions.Mode.OFF)
			return null;
		if (cpu)
			return opts.mode() == GpuAttentionOptions.Mode.ON
					? "--gpu-attention=on has no effect on the CPU backend; attention runs on the CPU"
					: null;
		if (!cudaBackend)
			return "--gpu-attention=" + opts.policyLabel() + " has no effect on this GPU backend: the attention kernel"
					+ " is CUDA-only, so attention runs on the CPU";
		return null;
	}

	/**
	 * Logs, once per process, that a Phi-2 or Qwen3-MoE handler was given a GPU
	 * backend it will not use. A no-op for any other architecture.
	 */
	static void announceCpuOnlyHandler(String architecture) {
		String arch = normalize(architecture);
		if (CPU_ONLY_HANDLERS.contains(arch) && ANNOUNCED.add(arch))
			log.warning(cpuOnlyHandlerNotice(arch));
	}

	/** Logs, once per process and backend, that the kernel was requested on a backend without it. */
	static void announceBackendWithoutKernel(String backendLabel) {
		if (ANNOUNCED.add("backend:" + backendLabel))
			log.warning("--gpu-attention=" + GpuAttentionOptions.fromEnv().policyLabel() + " has no effect on the "
					+ backendLabel + " backend: the attention kernel is CUDA-only, so attention runs on the CPU");
	}

	private static String cpuOnlyHandlerNotice(String arch) {
		return "architecture " + arch + " runs its matmuls and attention on the CPU on every backend;"
				+ " --gpu-layers and --gpu-attention have no effect on it";
	}

	private static String normalize(String architecture) {
		return architecture == null ? "" : architecture.strip().toLowerCase(Locale.ROOT);
	}
}
