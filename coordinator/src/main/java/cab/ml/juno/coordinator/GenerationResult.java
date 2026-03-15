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

import java.time.Duration;
import java.time.Instant;
import java.util.List;

/**
 * Completed inference result for a single request.
 *
 * For streaming use cases, tokens are delivered incrementally via
 * TokenConsumer. This record is the final summary returned when generation
 * completes.
 */
public record GenerationResult(String requestId, String text, // full decoded output
		List<Integer> tokenIds, // all generated token IDs
		int promptTokens, // input token count
		int generatedTokens, StopReason stopReason, Instant completedAt, Duration latency) {

	public GenerationResult {
		tokenIds = List.copyOf(tokenIds);
	}

	public enum StopReason {
		EOS_TOKEN, // model produced end-of-sequence token
		MAX_TOKENS, // hit samplingParams.maxTokens() limit
		STOP_TOKEN, // hit a configured stop token ID
		ERROR // upstream failure during generation
	}

	/** Tokens per second throughput. */
	public double tokensPerSecond() {
		double seconds = latency.toMillis() / 1000.0;
		return seconds > 0 ? generatedTokens / seconds : 0.0;
	}
}
