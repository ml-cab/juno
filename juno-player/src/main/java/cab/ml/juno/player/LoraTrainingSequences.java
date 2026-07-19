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
import java.util.Arrays;
import java.util.List;

import cab.ml.juno.tokenizer.Tokenizer;

/**
 * Builds token sequences and completion-only loss masks for LoRA SFT.
 */
final class LoraTrainingSequences {

	record MaskedChunk(int[] tokens, boolean[] lossMask) {
		int predictionCount() {
			int n = 0;
			for (boolean m : lossMask)
				if (m)
					n++;
			return n;
		}
	}

	record MaskedSequence(int[] tokens, boolean[] lossMask, String text) {
		int predictionCount() {
			int n = 0;
			for (boolean m : lossMask)
				if (m)
					n++;
			return n;
		}
	}

	private LoraTrainingSequences() {
	}

	/**
	 * Multiple phrasings of one Q&amp;A fact with loss only on answer tokens
	 * (plus the turn-ending special token).
	 */
	static MaskedSequence buildQa(Tokenizer tokenizer, String question, String answer, String modelTypeKey) {
		String q = question.endsWith("?") ? question : question + "?";
		String qLow = q.substring(0, 1).toLowerCase() + q.substring(1);
		String[] questions = { q, qLow, "Can you tell me: " + qLow, "Please answer: " + qLow };

		List<Integer> tokenList = new ArrayList<>();
		List<Boolean> maskList = new ArrayList<>();
		StringBuilder text = new StringBuilder();
		boolean first = true;

		for (String variant : questions) {
			String prefix = ChatTrainingFormats.qaPrefix(variant, modelTypeKey);
			String completion = ChatTrainingFormats.qaCompletion(answer, modelTypeKey);
			text.append(prefix).append(completion);

			int[] prefToks = first ? tokenizer.encode(prefix) : encodeNoBos(tokenizer, prefix);
			int[] fullToks = first ? tokenizer.encode(prefix + completion)
					: encodeNoBos(tokenizer, prefix + completion);
			first = false;

			int prefixLen = prefToks.length;
			if (fullToks.length < prefToks.length || !startsWith(fullToks, prefToks)) {
				// Tokenizer merged across the prefix/completion boundary — train the whole turn.
				prefixLen = 0;
			}

			int prevLen = tokenList.size();
			for (int id : fullToks)
				tokenList.add(id);

			if (prevLen == 0) {
				for (int i = 0; i < fullToks.length - 1; i++)
					maskList.add((i + 1) >= prefixLen);
			} else {
				// Bridge: previous last token → first token of this turn (prompt → not trained).
				maskList.add(0 >= prefixLen);
				for (int i = 0; i < fullToks.length - 1; i++)
					maskList.add((i + 1) >= prefixLen);
			}
		}

		int[] tokens = tokenList.stream().mapToInt(Integer::intValue).toArray();
		if (maskList.size() != tokens.length - 1)
			throw new IllegalStateException(
					"mask size " + maskList.size() + " != tokens-1 " + (tokens.length - 1));
		boolean[] lossMask = new boolean[maskList.size()];
		for (int i = 0; i < maskList.size(); i++)
			lossMask[i] = maskList.get(i);
		return new MaskedSequence(tokens, lossMask, text.toString());
	}

	static List<MaskedChunk> chunk(MaskedSequence seq, int chunkTokens) {
		return chunk(seq.tokens(), seq.lossMask(), chunkTokens);
	}

	static List<MaskedChunk> chunk(int[] tokens, boolean[] lossMask, int chunkTokens) {
		if (tokens.length < 2)
			return List.of();
		if (lossMask.length != tokens.length - 1)
			throw new IllegalArgumentException("lossMask length mismatch");

		List<MaskedChunk> chunks = new ArrayList<>();
		for (int start = 0; start < tokens.length - 1; start += chunkTokens) {
			int end = Math.min(start + chunkTokens + 1, tokens.length);
			if (end - start < 2)
				break;
			int[] chunkTok = Arrays.copyOfRange(tokens, start, end);
			boolean[] chunkMask = Arrays.copyOfRange(lossMask, start, end - 1);
			MaskedChunk chunk = new MaskedChunk(chunkTok, chunkMask);
			if (chunk.predictionCount() == 0)
				continue;
			chunks.add(chunk);
		}
		return chunks;
	}

	/** Full-sequence mask (every prediction position trained) for raw {@code /train}. */
	static boolean[] allTrueMask(int tokenCount) {
		boolean[] m = new boolean[Math.max(0, tokenCount - 1)];
		Arrays.fill(m, true);
		return m;
	}

	static int[] encodeNoBos(Tokenizer tokenizer, String text) {
		int[] ids = tokenizer.encode(text);
		if (ids.length > 0 && ids[0] == tokenizer.bosTokenId())
			return Arrays.copyOfRange(ids, 1, ids.length);
		return ids;
	}

	private static boolean startsWith(int[] full, int[] prefix) {
		if (prefix.length > full.length)
			return false;
		for (int i = 0; i < prefix.length; i++)
			if (full[i] != prefix[i])
				return false;
		return true;
	}
}
