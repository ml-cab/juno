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

import java.util.logging.Logger;

/**
 * Resident-weight upload with VRAM OOM ladder: FP32 microbatch → FP16
 * ({@code microbatch=1}) → CPU under {@code auto} / fail-closed under {@code gpu}.
 *
 * @author Yevhen Soldatov
 */
final class LoraResidentUpload {

	private LoraResidentUpload() {
	}

	/**
	 * Run {@code uploadAttempt}. On VRAM OOM while microbatch {@code > 1} and half
	 * residency is supported: close partial buffers, set microbatch to 1, retry once.
	 * Further OOM uses {@link LoraResidentWeights#tryRecoverFromUploadOom}.
	 */
	static void run(GpuMatVec gpu, Logger log, Runnable closer, Runnable uploadAttempt) {
		run(gpu.supportsHalfResident(), log, closer, uploadAttempt);
	}

	/**
	 * Package-visible overload for unit tests that simulate OOM without a live GPU.
	 *
	 * @param supportsHalfResident whether FP16 residency retry is available
	 */
	static void run(boolean supportsHalfResident, Logger log, Runnable closer, Runnable uploadAttempt) {
		try {
			uploadAttempt.run();
		} catch (IllegalStateException ex) {
			if (canRetryFp16(ex, supportsHalfResident)) {
				closer.run();
				log.warning("LoRA: insufficient GPU VRAM for FP32 microbatch residency (" + ex.getMessage()
						+ "). Retrying with --lora-microbatch 1 (FP16 GEMV).");
				LoraMicrobatch.apply(1);
				try {
					uploadAttempt.run();
				} catch (IllegalStateException ex2) {
					LoraResidentWeights.tryRecoverFromUploadOom(ex2, log, closer);
				}
			} else {
				LoraResidentWeights.tryRecoverFromUploadOom(ex, log, closer);
			}
		}
	}

	private static boolean canRetryFp16(IllegalStateException ex, boolean supportsHalfResident) {
		return LoraResidentWeights.isVramOom(ex) && LoraMicrobatch.current() > 1 && supportsHalfResident;
	}
}
