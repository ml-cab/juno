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
package cab.ml.juno.player;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Epoch sizing for LoRA train corpora: document-level units and seeded
 * max-token caps over whole chunk windows.
 *
 * <p>
 * Caps reduce wall-clock work; they do not change the CE objective on the
 * included supervised prediction tokens. A capped run is not identical to a
 * full-corpus pass.
 */
public final class LoraCorpusLimit {

	/** Historical REPL default (reproducibility). Prefer 128 for large {@code /train-file}. */
	public static final int DEFAULT_CHUNK_TOKENS = 32;

	/**
	 * Hard ceiling on chunk window size. Larger values increase activation memory
	 * {@code O(N * layers * hidden)}; reject rather than silently clamp.
	 */
	public static final int MAX_CHUNK_TOKENS = 8192;

	private LoraCorpusLimit() {
	}

	/**
	 * @param chunkTokens prediction positions per truncated-BPTT window ({@code >= 1},
	 *                    {@code <= }{@link #MAX_CHUNK_TOKENS})
	 * @return {@code chunkTokens} unchanged when valid
	 */
	public static int validateChunkTokens(int chunkTokens) {
		if (chunkTokens < 1)
			throw new IllegalArgumentException("chunkTokens must be >= 1");
		if (chunkTokens > MAX_CHUNK_TOKENS)
			throw new IllegalArgumentException(
					"chunkTokens must be <= " + MAX_CHUNK_TOKENS + " (got " + chunkTokens + ")");
		return chunkTokens;
	}

	/**
	 * @param maxTrainTokens supervised prediction-token budget; {@code 0} = unlimited
	 */
	public static int validateMaxTrainTokens(int maxTrainTokens) {
		if (maxTrainTokens < 0)
			throw new IllegalArgumentException("maxTrainTokens must be >= 0");
		return maxTrainTokens;
	}

	/**
	 * Build train units from one masked document, optionally capping supervised
	 * prediction tokens via a seeded whole-chunk subsample.
	 *
	 * <p>
	 * When {@code maxTrainTokens == 0} or the document already fits the budget,
	 * returns a single document-level unit. Otherwise chunks with
	 * {@code chunkTokens}, Fisher–Yates-shuffles chunk indices with
	 * {@code Random(seed)}, and takes a prefix until the prediction-token budget
	 * is met (last selected chunk may slightly overshoot).
	 */
	public static List<LoraTrainingLoop.TrainUnit> limitDocument(int[] tokens, boolean[] lossMask, int chunkTokens,
			int maxTrainTokens, long seed) {
		validateChunkTokens(chunkTokens);
		validateMaxTrainTokens(maxTrainTokens);
		if (tokens == null || tokens.length < 2)
			return List.of();
		if (lossMask == null || lossMask.length != tokens.length - 1)
			throw new IllegalArgumentException("lossMask length mismatch");

		int totalPred = predictionCount(lossMask);
		if (totalPred == 0)
			return List.of();

		if (maxTrainTokens == 0 || totalPred <= maxTrainTokens)
			return List.of(new LoraTrainingLoop.TrainUnit(tokens, lossMask));

		List<LoraTrainingSequences.MaskedChunk> chunks = LoraTrainingSequences.chunk(tokens, lossMask, chunkTokens);
		if (chunks.isEmpty())
			return List.of();

		int[] order = new int[chunks.size()];
		for (int i = 0; i < order.length; i++)
			order[i] = i;
		shuffle(order, new Random(seed));

		List<LoraTrainingLoop.TrainUnit> selected = new ArrayList<>();
		int budget = 0;
		for (int idx : order) {
			var c = chunks.get(idx);
			selected.add(new LoraTrainingLoop.TrainUnit(c.tokens(), c.lossMask()));
			budget += c.predictionCount();
			if (budget >= maxTrainTokens)
				break;
		}
		return List.copyOf(selected);
	}

	private static int predictionCount(boolean[] lossMask) {
		int n = 0;
		for (boolean m : lossMask)
			if (m)
				n++;
		return n;
	}

	/** Fisher–Yates in-place shuffle. */
	private static void shuffle(int[] order, Random rng) {
		for (int i = order.length - 1; i > 0; i--) {
			int j = rng.nextInt(i + 1);
			int tmp = order[i];
			order[i] = order[j];
			order[j] = tmp;
		}
	}
}
