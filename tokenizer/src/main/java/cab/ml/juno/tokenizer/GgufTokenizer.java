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
import java.util.Comparator;
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

	/**
	 * True for GPT-2/tiktoken BPE models (e.g. Llama 3+), false for SentencePiece
	 * BPE (Llama 1/2, TinyLlama, Mistral, Gemma, Phi-3).
	 *
	 * Determined by {@code tokenizer.ggml.model == "gpt2"} in GGUF metadata.
	 *
	 * Affects space normalisation in {@link #encode}: GPT-2 BPE represents a
	 * leading space as {@link #GP} (Ġ U+0120) at the start of each token, while
	 * SentencePiece uses {@link #SP} (▁ U+2581) as a word-boundary prefix with a
	 * mandatory leading ▁ before the first token.
	 */
	private final boolean isGpt2Bpe;

	/**
	 * Special-token pieces sorted longest-first. Used in {@link #encode} to
	 * pre-split the input text at special-token boundaries before BPE so that
	 * control tokens like {@code <|begin_of_text|>}, {@code <|eot_id|>} etc. map
	 * to their single vocabulary IDs instead of being decomposed character by
	 * character.
	 *
	 * Contains all vocab entries whose token type is 3 (control) or 4
	 * (user-defined) and whose piece string starts with {@code <|} (to avoid
	 * including single-byte or generic control pieces).
	 */
	private final List<String> sortedSpecialPieces;

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

		// Detect BPE variant: "gpt2" = Llama 3+ tiktoken/BPE, anything else (null,
		// "llama", "llama2") = SentencePiece BPE.
		String ggmlModel = r.metaString("tokenizer.ggml.model");
		boolean isGpt2Bpe = "gpt2".equals(ggmlModel);

		log.info("Tokenizer loaded: vocabSize=" + V + " bos=" + bosId + " eos=" + eosId
				+ " model=" + (ggmlModel != null ? ggmlModel : "llama(default)")
				+ (isGpt2Bpe ? " [GPT-2 BPE]" : " [SentencePiece]"));
		return new GgufTokenizer(vocab, scores, tokenTypes, bosId, eosId, padId, unkId, isGpt2Bpe);
	}

	private GgufTokenizer(String[] vocab, float[] scores, int[] tokenTypes, int bosId, int eosId, int padId,
			int unkId, boolean isGpt2Bpe) {
		this.vocab = vocab;
		this.scores = scores;
		this.tokenTypes = tokenTypes;
		this.bosId = bosId;
		this.eosId = eosId;
		this.padId = padId;
		this.unkId = unkId;
		this.isGpt2Bpe = isGpt2Bpe;

		pieceToId = new HashMap<>(vocab.length * 2);
		for (int i = 0; i < vocab.length; i++)
			pieceToId.put(vocab[i], i);

		// Build the special-token pre-split list: control (type 3) and user-defined
		// (type 4) tokens whose piece looks like <|...|>. Sorted longest-first so
		// that a longer special token is matched before any prefix of it.
		List<String> specials = new ArrayList<>();
		for (int i = 0; i < vocab.length; i++) {
			int t = tokenTypes[i];
			if ((t == 3 || t == 4) && vocab[i].startsWith("<|") && vocab[i].endsWith("|>"))
				specials.add(vocab[i]);
		}
		specials.sort(Comparator.comparingInt(String::length).reversed());
		this.sortedSpecialPieces = List.copyOf(specials);
	}

	// ── Tokenizer interface ───────────────────────────────────────────────────

	@Override
	public int[] encode(String text) {
		if (text == null || text.isEmpty())
			return new int[] { bosId };

		TokenizerEvent evt = new TokenizerEvent();
		evt.begin();

		// Split the raw text into segments: either an exact special-token piece
		// (which maps to a single vocab ID immediately) or a plain-text run (which
		// goes through normalisation + BPE).
		//
		// This is necessary for GPT-2 BPE models (Llama 3+) where the chat template
		// injects literal special-token strings such as <|begin_of_text|>,
		// <|start_header_id|> etc. Without pre-splitting, these 17-char strings get
		// decomposed character-by-character and the model never sees the correct
		// control token IDs — causing garbled output.
		//
		// SentencePiece models benefit too: it prevents accidental BPE merging of
		// what should be atomic control tokens.
		List<Sym> syms = new ArrayList<>();
		for (String segment : splitOnSpecialTokens(text)) {
			Integer specialId = pieceToId.get(segment);
			boolean isSpecial = specialId != null
					&& specialId < tokenTypes.length
					&& (tokenTypes[specialId] == 3 || tokenTypes[specialId] == 4)
					&& segment.startsWith("<|") && segment.endsWith("|>");
			if (isSpecial) {
				// Emit the control token directly — no BPE, no normalisation.
				syms.add(new Sym(segment, specialId, scores[specialId < scores.length ? specialId : 0]));
			} else {
				// Plain-text segment: apply model-appropriate space normalisation, then
				// decompose into initial symbols for BPE.
				//
				// SentencePiece (Llama 1/2, TinyLlama, Mistral, Phi-3):
				//   Prepend ▁ to the segment so the first word is treated identically to
				//   mid-sentence words. Spaces → ▁.
				//
				// GPT-2 BPE (Llama 3+):
				//   Do NOT prepend anything — GPT-2 tokens carry Ġ (U+0120) at their
				//   OWN start when they begin a new word. Spaces → Ġ.
				//   The leading-space prefix logic that SentencePiece needs is baked into
				//   the token strings themselves (e.g. "Ġhello" means " hello").
				String normalised = isGpt2Bpe
						? segment.replace(' ', GP)
						: SP + segment.replace(' ', SP);

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
			}
		}

		// BPE merges: repeatedly find the adjacent pair with the highest score and
		// merge. Special-token symbols are left untouched because their pair score
		// with neighbours will be Float.NEGATIVE_INFINITY (the combined string won't
		// be in pieceToId).
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

		// Prepend BOS token — but only if the text didn't already start with one.
		// GPT-2 BPE chat templates (e.g. Llama 3) inject <|begin_of_text|> as the
		// very first special token; prepending bosId again would produce a double-BOS
		// sequence that causes the model to emit EOS immediately on the first turn.
		// SentencePiece models never include BOS in the formatted text, so the guard
		// is a no-op for them.
		boolean startsWithBos = !syms.isEmpty() && syms.get(0).id == bosId;
		int offset = startsWithBos ? 0 : 1;
		int[] result = new int[syms.size() + offset];
		if (!startsWithBos)
			result[0] = bosId;
		for (int i = 0; i < syms.size(); i++)
			result[i + offset] = syms.get(i).id;

		evt.tokenizerType = "gguf";
		evt.operation = "encode";
		evt.inputLength = text.length();
		evt.outputLength = result.length;
		evt.commit();

		return result;
	}

	/**
	 * Splits {@code text} into alternating runs of special-token pieces and plain
	 * text, using {@link #sortedSpecialPieces} as the delimiter set.
	 *
	 * <p>
	 * Empty strings are omitted from the result. For models with no special pieces
	 * (e.g. plain SentencePiece models) this returns a single-element list
	 * containing the whole text.
	 *
	 * <p>
	 * Example (Llama 3): {@code "<|begin_of_text|>hello<|eot_id|>"} →
	 * {@code ["<|begin_of_text|>", "hello", "<|eot_id|>"]}
	 */
	private List<String> splitOnSpecialTokens(String text) {
		if (sortedSpecialPieces.isEmpty())
			return List.of(text);

		List<String> out = new ArrayList<>();
		int pos = 0;
		outer: while (pos < text.length()) {
			// Try to match any special piece at the current position
			for (String sp : sortedSpecialPieces) {
				if (text.startsWith(sp, pos)) {
					out.add(sp);
					pos += sp.length();
					continue outer;
				}
			}
			// No special token here: advance to the next potential special-token start
			int next = text.length();
			for (String sp : sortedSpecialPieces) {
				int idx = text.indexOf(sp, pos);
				if (idx != -1 && idx < next)
					next = idx;
			}
			if (pos < next)
				out.add(text.substring(pos, next));
			pos = next;
		}
		return out.isEmpty() ? List.of(text) : out;
	}

	@Override
	public String decode(int[] tokenIds) {
		TokenizerEvent evt = new TokenizerEvent();
		evt.begin();
		StringBuilder sb = new StringBuilder();
		for (int id : tokenIds)
			sb.append(decodeToken(id));
		// decodeToken() already replaced ▁ with space; just strip the leading space
		// that the first token's ▁ prefix would have introduced.
		String result = sb.toString();
		result = result.startsWith(" ") ? result.substring(1) : result;
		evt.tokenizerType = "gguf";
		evt.operation = "decode";
		evt.inputLength = tokenIds.length;
		evt.outputLength = result.length();
		evt.commit();
		return result;
	}

	@Override
	public String decodeToken(int tokenId) {
		TokenizerEvent evt = new TokenizerEvent();
		evt.begin();
		String piece;
		if (tokenId < 0 || tokenId >= vocab.length) {
			piece = "";
		} else {
			// Skip BOS, EOS, and control tokens
			int type = tokenId < tokenTypes.length ? tokenTypes[tokenId] : 1;
			if (type == 3 /* control */ || tokenId == bosId || tokenId == eosId) {
				piece = "";
			} else {
				String raw = vocab[tokenId];
				// Byte tokens like <0xHH> → actual byte
				if (raw.matches("<0x[0-9A-Fa-f]{2}>")) {
					int b = Integer.parseInt(raw.substring(3, 5), 16);
					raw = new String(new byte[] { (byte) b }, java.nio.charset.StandardCharsets.UTF_8);
				}
				// Replace SentencePiece space prefix (▁ U+2581) with a real space so that
				// streaming callers (which receive one piece at a time) see correct whitespace.
				// The full decode() path also does this replacement, but streaming builds
				// fullText directly from decodeToken() pieces without going through decode().
				// Ċ (U+010A) is GPT-2 BPE's representation of newline (\n).
				piece = raw.replace(SP, ' ').replace(GP, ' ').replace('Ċ', '\n');
			}
		}
		evt.tokenizerType = "gguf";
		evt.operation = "decodeToken";
		evt.inputLength = 1;
		evt.outputLength = piece.length();
		evt.commit();
		return piece;
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
