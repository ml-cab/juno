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

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.logging.Logger;

import ai.djl.sentencepiece.SpTokenizer;
import ai.djl.sentencepiece.SpVocabulary;

/**
 * Production tokenizer backed by DJL SentencePiece.
 *
 * DJL SpTokenizer public API: tokenize(String) → List<String> piece strings
 * buildSentence(List) → String pieces back to text
 *
 * SpVocabulary (loaded from same model file) handles piece ↔ ID mapping:
 * vocab.getIndex(piece) → long (token ID) vocab.getToken(index) → String
 *
 * Thread-safe — both SpTokenizer and SpVocabulary are stateless after loading.
 *
 * Usage: DJLTokenizer t = DJLTokenizer.load(
 * Path.of("/models/llama3/tokenizer.model"), "llama3"); int[] ids =
 * t.encode("Hello world");
 */
public final class DJLTokenizer implements Tokenizer, AutoCloseable {

	private static final Logger log = Logger.getLogger(DJLTokenizer.class.getName());

	private final SpTokenizer sp;
	private final SpVocabulary vocab;
	private final String modelType;
	private final int bosTokenId;
	private final int eosTokenId;
	private final int padTokenId;

	private DJLTokenizer(SpTokenizer sp, SpVocabulary vocab, String modelType, int bosTokenId, int eosTokenId,
			int padTokenId) {
		this.sp = sp;
		this.vocab = vocab;
		this.modelType = modelType;
		this.bosTokenId = bosTokenId;
		this.eosTokenId = eosTokenId;
		this.padTokenId = padTokenId;
	}

	/**
	 * Load tokenizer from a SentencePiece model file (.model). SpVocabulary is
	 * derived from the same model file.
	 */
	public static DJLTokenizer load(Path modelFile, String modelType) throws IOException {
		log.info("Loading tokenizer from: " + modelFile + " (type=" + modelType + ")");

		SpTokenizer sp = new SpTokenizer(modelFile);
		SpVocabulary vocab = SpVocabulary.from(sp);

		// Standard special token IDs for each family
		int bos = 1, eos = 2, pad = 0;
		if ("llama3".equalsIgnoreCase(modelType)) {
			bos = 128000; // <|begin_of_text|>
			eos = 128009; // <|eot_id|>
		}

		log.info("Tokenizer loaded: vocabSize=" + vocab.size() + " bos=" + bos + " eos=" + eos);
		return new DJLTokenizer(sp, vocab, modelType, bos, eos, pad);
	}

	// ── Tokenizer interface ───────────────────────────────────────────────────

	@Override
	public int[] encode(String text) {
		if (text == null || text.isEmpty())
			return new int[0];

		TokenizerEvent evt = new TokenizerEvent();
		evt.begin();
		List<String> pieces = sp.tokenize(text);
		int[] ids = new int[pieces.size()];
		for (int i = 0; i < pieces.size(); i++) {
			ids[i] = (int) vocab.getIndex(pieces.get(i));
		}
		evt.tokenizerType = "djl";
		evt.operation = "encode";
		evt.inputLength = text.length();
		evt.outputLength = ids.length;
		evt.commit();
		return ids;
	}

	@Override
	public String decode(int[] tokenIds) {
		if (tokenIds == null || tokenIds.length == 0)
			return "";

		TokenizerEvent evt = new TokenizerEvent();
		evt.begin();
		StringBuilder sb = new StringBuilder();
		for (int id : tokenIds) {
			String piece = decodeToken(id);
			if (!piece.isEmpty())
				sb.append(piece);
		}
		// SentencePiece uses U+2581 as space prefix — replace with real space
		String result = sb.toString().replace('\u2581', ' ').stripLeading();
		evt.tokenizerType = "djl";
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
		try {
			piece = vocab.getToken(tokenId);
			if (piece == null || isSpecialToken(piece))
				piece = "";
		} catch (Exception e) {
			piece = ""; // out-of-range or unknown — safe to skip
		}
		evt.tokenizerType = "djl";
		evt.operation = "decodeToken";
		evt.inputLength = 1;
		evt.outputLength = piece.length();
		evt.commit();
		return piece;
	}

	@Override
	public int bosTokenId() {
		return bosTokenId;
	}

	@Override
	public int eosTokenId() {
		return eosTokenId;
	}

	@Override
	public int padTokenId() {
		return padTokenId;
	}

	@Override
	public int vocabSize() {
		return (int) vocab.size();
	}

	@Override
	public String modelType() {
		return modelType;
	}

	@Override
	public boolean isReady() {
		return true;
	}

	@Override
	public void close() {
		sp.close();
	}

	// ── Helpers ───────────────────────────────────────────────────────────────

	private boolean isSpecialToken(String piece) {
		return piece.startsWith("<") && piece.endsWith(">");
	}
}