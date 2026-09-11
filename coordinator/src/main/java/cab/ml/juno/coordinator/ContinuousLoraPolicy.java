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
 * Multi-adapter × continuous v1: one process-wide {@code --lora-play} set.
 * Per-request {@code x_juno_loras} is fail-closed under continuous schedule.
 */
public final class ContinuousLoraPolicy {

	public static final String ERROR = "per-request adapters (x_juno_loras) are not supported under "
			+ "schedule=continuous; use process-wide --lora-play";

	private ContinuousLoraPolicy() {
	}

	public static boolean forbidden(boolean hasPerRequestLoras, ServeScheduleOptions schedule) {
		return hasPerRequestLoras && schedule != null
				&& schedule.mode() == ServeScheduleOptions.Mode.CONTINUOUS;
	}
}
