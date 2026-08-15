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

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * User-facing LoRA train fallback notices for the REPL (JUL is often silenced).
 *
 * @author Yevhen Soldatov
 */
public final class LoraTrainNotices {

	/** FP32 microbatch VRAM OOM → retry at microbatch 1 (FP16 GEMV). */
	public static final String FP16_MICROBATCH =
			"Not enough GPU memory for FP32 microbatch residency.\n"
					+ "  Switched to microbatch size 1 (FP16). Training continues on GPU; "
					+ "frozen linears use sequential GEMV (usually slower than batched GEMM, uses less VRAM).";

	/** Resident upload VRAM OOM under auto → CPU quantized frozen matmul. */
	public static final String CPU_RESIDENT =
			"Not enough GPU memory to keep frozen weights resident on the GPU.\n"
					+ "  Frozen layers fall back to CPU quantized matmul. Training still works, "
					+ "but steps will be much slower. Try --lora-microbatch 1 earlier, free VRAM, "
					+ "or a smaller model.";

	/** {@code auto} selected CPU because GPU was disabled. */
	public static final String AUTO_CPU_DISABLED =
			"GPU disabled (--cpu / JUNO_USE_GPU=false) — training on CPU.\n"
					+ "  LoRA runs entirely on CPU (slower). Enable the GPU for faster training.";

	/** {@code auto} selected CPU because no CUDA/ROCm device was found. */
	public static final String AUTO_CPU_UNAVAILABLE =
			"No GPU detected — training on CPU.\n"
					+ "  LoRA runs entirely on CPU (slower). Install CUDA/ROCm or use a GPU machine.";

	/** {@code auto} selected CPU because {@code juno.gpu.device} was out of range. */
	public static final String AUTO_CPU_BAD_DEVICE =
			"GPU device index out of range — training on CPU.\n"
					+ "  LoRA runs entirely on CPU (slower). Fix juno.gpu.device / juno.cuda.device.";

	private static final CopyOnWriteArrayList<String> NOTICES = new CopyOnWriteArrayList<>();

	private LoraTrainNotices() {
	}

	/** Drop any pending notices (call before a LoRA load). */
	public static void clear() {
		NOTICES.clear();
	}

	/** Append a multi-line user-facing notice (why + consequence). */
	public static void add(String notice) {
		if (notice != null && !notice.isBlank())
			NOTICES.add(notice.strip());
	}

	/** Snapshot and clear pending notices. */
	public static List<String> drain() {
		List<String> out = new ArrayList<>(NOTICES);
		NOTICES.clear();
		return List.copyOf(out);
	}
}
