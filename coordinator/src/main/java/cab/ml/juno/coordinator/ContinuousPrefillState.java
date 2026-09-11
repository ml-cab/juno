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

/**
 * Remaining prefill window for one continuous running-set member.
 *
 * <p>Prefill covers prompt positions {@code [cursor, promptLen - 2]}; the last
 * prompt token is reserved for the first decode step (same contract as
 * {@link PrefillChunker}).
 */
final class ContinuousPrefillState {

	private final int[] promptIds;
	private final int endExclusive;
	private int cursor;

	private ContinuousPrefillState(int[] promptIds, int cursor, int endExclusive) {
		this.promptIds = promptIds;
		this.cursor = cursor;
		this.endExclusive = endExclusive;
	}

	/**
	 * @param promptIds full prompt token ids (not copied; caller owns lifetime)
	 * @param startPos  first position still needing prefill (prefix-hit aware)
	 */
	static ContinuousPrefillState start(int[] promptIds, int startPos) {
		if (promptIds == null)
			throw new IllegalArgumentException("promptIds must not be null");
		int end = Math.max(0, promptIds.length - 1);
		int cursor = Math.min(Math.max(0, startPos), end);
		return new ContinuousPrefillState(promptIds, cursor, end);
	}

	boolean isComplete() {
		return cursor >= endExclusive;
	}

	int remainingTokens() {
		return Math.max(0, endExclusive - cursor);
	}

	int cursor() {
		return cursor;
	}

	/**
	 * Peek the next ubatch chunk of at most {@code chunkSize} tokens without advancing.
	 * Returns {@link Chunk#EMPTY} when complete.
	 */
	Chunk nextChunk(int chunkSize) {
		if (isComplete())
			return Chunk.EMPTY;
		int chunk = Math.max(1, chunkSize);
		int chunkEnd = Math.min(cursor + chunk, endExclusive);
		int[] window = Arrays.copyOfRange(promptIds, cursor, chunkEnd);
		return new Chunk(window, cursor);
	}

	void advance(int tokensConsumed) {
		if (tokensConsumed < 0)
			throw new IllegalArgumentException("tokensConsumed must be >= 0");
		cursor = Math.min(endExclusive, cursor + tokensConsumed);
	}

	/**
	 * One prefill ubatch window: tokens {@code prompt[startPos .. startPos+len)}.
	 */
	record Chunk(int[] tokens, int startPos) {
		static final Chunk EMPTY = new Chunk(new int[0], 0);

		boolean isEmpty() {
			return tokens.length == 0;
		}

		int tokenCount() {
			return tokens.length;
		}
	}
}
