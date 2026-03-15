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
package cab.ml.juno.tokenizer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import cab.ml.juno.node.GgufReader;

/**
 * SentencePiece BPE tokenizer built entirely from GGUF metadata.
 *
 * No external tokenizer.model file required — everything lives in the GGUF.
 *
 * Reads: tokenizer.ggml.tokens String[] — vocab pieces (▁ = space prefix)
 * tokenizer.ggml.scores float[] — BPE merge scores (higher = preferred)
 * tokenizer.ggml.token_type int[] — 1=normal 2=unknown 3=control 6=byte
 * tokenizer.ggml.bos_token_id tokenizer.ggml.eos_token_id
 *
 * Encoding algorithm (SentencePiece BPE): 1. Prepend ▁ to the first word, ▁ to
 * each subsequent word (space normalisation) 2. Initialise: one symbol per
 * UTF-8 character (fall back to byte tokens for OOV) 3. Greedily merge the
 * adjacent pair with the highest score until no merges remain
 *
 * Thread-safe after construction.
 */
public final class GgufTokenizer implements Tokenizer {

	private static final Logger log = Logger.getLogger(GgufTokenizer.class.getName());

	// U+2581 LOWER ONE EIGHTH BLOCK — SentencePiece's space prefix character
	private static final char SP = '\u2581';
	// U+0120 LATIN SMALL LETTER G WITH CEDILLA — BPE (GPT-2/Llama-3) space prefix
	private static final char GP = '\u0120';

	private final String[] vocab; // token ID → piece string
	private final float[] scores; // token ID → BPE score
	private final int[] tokenTypes; // token ID → type
	private final Map<String, Integer> pieceToId;

	private final int bosId;
	private final int eosId;
	private final int padId;
	private final int unkId;

	// ── Factory ───────────────────────────────────────────────────────────────

	public static GgufTokenizer load(GgufReader r) {
		log.info("Loading tokenizer from GGUF metadata...");

		Object[] tokensRaw = (Object[]) r.meta("tokenizer.ggml.tokens");
		Object[] scoresRaw = (Object[]) r.meta("tokenizer.ggml.scores");
		Object[] typesRaw = (Object[]) r.meta("tokenizer.ggml.token_type");

		if (tokensRaw == null)
			throw new IllegalArgumentException("GGUF file has no tokenizer.ggml.tokens metadata");

		int V = tokensRaw.length;
		String[] vocab = new String[V];
		float[] scores = new float[V];
		int[] tokenTypes = new int[V];

		for (int i = 0; i < V; i++) {
			vocab[i] = (String) tokensRaw[i];
			scores[i] = scoresRaw != null ? ((Number) scoresRaw[i]).floatValue() : 0f;
			tokenTypes[i] = typesRaw != null ? ((Number) typesRaw[i]).intValue() : 1;
		}

		int bosId = (int) r.metaLong("tokenizer.ggml.bos_token_id", 1);
		int eosId = (int) r.metaLong("tokenizer.ggml.eos_token_id", 2);
		int padId = (int) r.metaLong("tokenizer.ggml.padding_token_id", 0);
		int unkId = (int) r.metaLong("tokenizer.ggml.unknown_token_id", 0);

		log.info("Tokenizer loaded: vocabSize=" + V + " bos=" + bosId + " eos=" + eosId);
		return new GgufTokenizer(vocab, scores, tokenTypes, bosId, eosId, padId, unkId);
	}

	private GgufTokenizer(String[] vocab, float[] scores, int[] tokenTypes, int bosId, int eosId, int padId,
			int unkId) {
		this.vocab = vocab;
		this.scores = scores;
		this.tokenTypes = tokenTypes;
		this.bosId = bosId;
		this.eosId = eosId;
		this.padId = padId;
		this.unkId = unkId;

		pieceToId = new HashMap<>(vocab.length * 2);
		for (int i = 0; i < vocab.length; i++)
			pieceToId.put(vocab[i], i);
	}

	// ── Tokenizer interface ───────────────────────────────────────────────────

	@Override
	public int[] encode(String text) {
		if (text == null || text.isEmpty())
			return new int[] { bosId };

		// SentencePiece normalisation: replace spaces with ▁, prefix the whole
		// string with ▁ so the first word gets the same treatment as mid-sentence words
		String normalised = SP + text.replace(' ', SP);

		// Initialise symbol list: one node per UTF-8 code point
		List<Sym> syms = new ArrayList<>();
		for (int cp : (Iterable<Integer>) normalised.codePoints()::iterator) {
			String piece = new String(Character.toChars(cp));
			Integer id = pieceToId.get(piece);
			if (id == null) {
				// OOV character: fall back to byte tokens <0xHH>
				byte[] bytes = piece.getBytes(java.nio.charset.StandardCharsets.UTF_8);
				for (byte b : bytes) {
					String byteKey = String.format("<0x%02X>", b & 0xFF);
					id = pieceToId.getOrDefault(byteKey, unkId);
					syms.add(new Sym(byteKey, id, scores[id < scores.length ? id : 0]));
				}
				continue;
			}
			syms.add(new Sym(piece, id, scores[id]));
		}

		// BPE merges: repeatedly find the pair with the highest score and merge
		boolean merged = true;
		while (merged && syms.size() > 1) {
			merged = false;
			int bestIdx = -1;
			float bestScore = Float.NEGATIVE_INFINITY;
			int bestId = -1;

			for (int i = 0; i < syms.size() - 1; i++) {
				String pair = syms.get(i).piece + syms.get(i + 1).piece;
				Integer id = pieceToId.get(pair);
				if (id != null && scores[id] > bestScore) {
					bestScore = scores[id];
					bestIdx = i;
					bestId = id;
				}
			}

			if (bestIdx >= 0) {
				String mergedPiece = syms.get(bestIdx).piece + syms.get(bestIdx + 1).piece;
				syms.set(bestIdx, new Sym(mergedPiece, bestId, bestScore));
				syms.remove(bestIdx + 1);
				merged = true;
			}
		}

		// Prepend BOS token
		int[] result = new int[syms.size() + 1];
		result[0] = bosId;
		for (int i = 0; i < syms.size(); i++)
			result[i + 1] = syms.get(i).id;
		return result;
	}

	@Override
	public String decode(int[] tokenIds) {
		StringBuilder sb = new StringBuilder();
		for (int id : tokenIds)
			sb.append(decodeToken(id));
		// decodeToken() already replaced ▁ with space; just strip the leading space
		// that the first token's ▁ prefix would have introduced.
		String result = sb.toString();
		return result.startsWith(" ") ? result.substring(1) : result;
	}

	@Override
	public String decodeToken(int tokenId) {
		if (tokenId < 0 || tokenId >= vocab.length)
			return "";
		// Skip BOS, EOS, and control tokens
		int type = tokenId < tokenTypes.length ? tokenTypes[tokenId] : 1;
		if (type == 3 /* control */ || tokenId == bosId || tokenId == eosId)
			return "";
		String piece = vocab[tokenId];
		// Byte tokens like <0xHH> → actual byte
		if (piece.matches("<0x[0-9A-Fa-f]{2}>")) {
			int b = Integer.parseInt(piece.substring(3, 5), 16);
			return new String(new byte[] { (byte) b }, java.nio.charset.StandardCharsets.UTF_8);
		}
		// Replace SentencePiece space prefix (▁ U+2581) with a real space so that
		// streaming callers (which receive one piece at a time) see correct whitespace.
		// The full decode() path also does this replacement, but streaming builds
		// fullText directly from decodeToken() pieces without going through decode().
		return piece.replace(SP, ' ').replace(GP, ' ');
	}

	@Override
	public int bosTokenId() {
		return bosId;
	}

	@Override
	public int eosTokenId() {
		return eosId;
	}

	@Override
	public int padTokenId() {
		return padId;
	}

	@Override
	public int vocabSize() {
		return vocab.length;
	}

	@Override
	public String modelType() {
		return "llama";
	}

	@Override
	public boolean isReady() {
		return true;
	}

	// ── Inner types ───────────────────────────────────────────────────────────

	private record Sym(String piece, int id, float score) {
	}
}