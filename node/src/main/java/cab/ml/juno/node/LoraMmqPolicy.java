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

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Logger;

/**
 * Playback-only gate for fused Q4_K MMQ under LoRA residency.
 *
 * <p>Training keeps FP16/FP32 frozen uploads (no Q4 transpose kernel). Inference
 * {@code --lora-play} may use packed Q4 when {@code --mmq} prefers it and the
 * CUDA kernel is available.
 *
 * <p>Play context is {@link #PLAY_PATH_PROPERTY} (set by cluster nodes and by
 * local {@code ConsoleMain} when loading play adapters). An explicit
 * {@link LoraMicrobatch} {@code > 1} disables MMQ (FP32 GEMM residency); an
 * <em>unset</em> microbatch property does not — play never calls
 * {@link LoraMicrobatch#apply}, so the train default of 8 must not block play.
 */
public final class LoraMmqPolicy {

	/** Canonical playback marker; also the cluster/local adapter path. */
	public static final String PLAY_PATH_PROPERTY = "juno.lora.play.path";

	private static final AtomicBoolean TRAIN_WARNED = new AtomicBoolean();

	private LoraMmqPolicy() {
	}

	/** True when this JVM is loading / running {@code --lora-play} adapters. */
	public static boolean isPlaybackContext() {
		String path = System.getProperty(PLAY_PATH_PROPERTY);
		return path != null && !path.isBlank();
	}

	/**
	 * Whether LoRA frozen uploads should use packed Q4_K MMQ for this GPU.
	 *
	 * @param supportsQ4KMmq {@link GpuMatVec#supportsQ4KMmq()}
	 */
	public static boolean enabledForPlayback(boolean supportsQ4KMmq) {
		if (!isPlaybackContext() || !supportsQ4KMmq)
			return false;
		if (!MmqOptions.fromEnv().preferMmq())
			return false;
		// Explicit train-style microbatch > 1 forces FP32 residency; block MMQ.
		// Unset property → play path (do not treat DEFAULT 8 as a play gate).
		if (System.getProperty(LoraMicrobatch.PROPERTY) != null && LoraMicrobatch.current() > 1)
			return false;
		return true;
	}

	/** Convenience: {@link #enabledForPlayback(boolean)} with live GPU capability. */
	public static boolean enabledForPlayback(GpuMatVec gpu) {
		return gpu != null && enabledForPlayback(gpu.supportsQ4KMmq());
	}

	/**
	 * When {@code --mmq} is preferred outside playback, log once that training
	 * stays on FP residency (explicit no-op per ROADMAP §6).
	 *
	 * @return {@code true} if a warning was emitted this process
	 */
	public static boolean warnIfTrainIgnoresMmq(Logger log) {
		if (isPlaybackContext() || !MmqOptions.fromEnv().preferMmq())
			return false;
		if (!TRAIN_WARNED.compareAndSet(false, true))
			return false;
		log.warning("LoRA training ignores --mmq (fused Q4_K); frozen weights stay FP16/FP32. "
				+ "Use --lora-play for playback MMQ.");
		return true;
	}

	/** Test hook: allow repeated warn assertions. */
	static void resetWarnStateForTests() {
		TRAIN_WARNED.set(false);
	}
}
