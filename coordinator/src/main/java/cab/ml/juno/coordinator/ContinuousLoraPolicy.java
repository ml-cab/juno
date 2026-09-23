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
package cab.ml.juno.coordinator;

import cab.ml.juno.kvcache.ServeScheduleOptions;

/**
 * Multi-adapter policy: v1 supports only one process-wide {@code --lora-play}
 * adapter set (loaded via {@code cab.ml.juno.lora.LoraPlaySpec}, optionally
 * combining several files with per-file scales). Per-request
 * {@code x_juno_loras} overrides are not wired into the forward-pass
 * pipeline on any schedule yet — selecting or re-scaling adapters per
 * request would need threading a per-request adapter selection through
 * {@code InferenceRequest} / {@code GenerationLoop} / the forward-pass
 * handlers, which has not been implemented. Per ROADMAP Execution rule §6
 * ("no silent ignore"), a
 * request that sets {@code x_juno_loras} fails closed on every schedule
 * (this used to only fail under {@code continuous}; static silently ignored
 * it, which is the gap this generalization closes) rather than accepting it
 * and running the process-wide set as if the override had taken effect.
 */
public final class ContinuousLoraPolicy {

	public static final String ERROR = "per-request adapters (x_juno_loras) are not wired yet on any "
			+ "schedule; use process-wide --lora-play (optionally with multiple files and per-file scales)";

	private ContinuousLoraPolicy() {
	}

	/** {@code schedule} is accepted for call-site compatibility; the check no longer depends on it. */
	public static boolean forbidden(boolean hasPerRequestLoras, ServeScheduleOptions schedule) {
		return hasPerRequestLoras;
	}
}
