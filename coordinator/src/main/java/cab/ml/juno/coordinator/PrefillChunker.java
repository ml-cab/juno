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

import java.util.Arrays;

import cab.ml.juno.node.InferencePipeline;

/**
 * Prefill execution for {@link GenerationLoop}: sequential single-token forwards
 * or windowed {@link InferencePipeline#prefillBatch} with optional chunking.
 */
final class PrefillChunker {

	private PrefillChunker() {
	}

	/**
	 * Populate KV cache for prompt positions {@code [startPos, promptLen - 2]}.
	 * The last prompt token is left for the first decode step.
	 */
	static void run(InferencePipeline pipeline, PrefillMode mode, int chunkSize, String requestId,
			int[] promptIds, int startPos) {
		int endExclusive = promptIds.length - 1;
		if (startPos >= endExclusive)
			return;

		if (mode == PrefillMode.SINGLE) {
			for (int p = startPos; p < endExclusive; p++) {
				int[] slice = Arrays.copyOfRange(promptIds, 0, p + 1);
				pipeline.forward(requestId, slice, p);
			}
			return;
		}

		int chunk = Math.max(1, chunkSize);
		for (int pos = startPos; pos < endExclusive;) {
			int chunkEnd = Math.min(pos + chunk, endExclusive);
			int[] window = Arrays.copyOfRange(promptIds, pos, chunkEnd);
			pipeline.prefillBatch(requestId, window, pos);
			pos = chunkEnd;
		}
	}

	/**
	 * Number of {@code prefillBatch} calls for a batched prefill window (for tests).
	 */
	static int batchedCallCount(int startPos, int promptLen, int chunkSize) {
		int tokens = promptLen - 1 - startPos;
		if (tokens <= 0)
			return 0;
		int chunk = Math.max(1, chunkSize);
		return (tokens + chunk - 1) / chunk;
	}
}
