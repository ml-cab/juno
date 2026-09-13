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
package cab.ml.juno.sampler;

import java.util.function.IntFunction;

/**
 * Per-sequence GBNF cursor. Masks illegal next tokens then records the chosen
 * token's UTF-8 bytes.
 */
public final class GrammarSession {

	static final byte[] EMPTY = new byte[0];
	private static final float NEG_INF = Float.NEGATIVE_INFINITY;

	private final IntFunction<byte[]> tokenUtf8;
	private final int eosTokenId;
	private final byte[][] cache;
	private GbnfGrammar.State state;

	private GrammarSession(GbnfGrammar grammar, IntFunction<byte[]> tokenUtf8, int eosTokenId, int vocabSize) {
		this.tokenUtf8 = tokenUtf8;
		this.eosTokenId = eosTokenId;
		this.cache = new byte[Math.max(0, vocabSize)][];
		this.state = grammar.newState();
		if (this.state == null)
			throw new IllegalArgumentException("grammar has no legal start state");
	}

	public static GrammarSession open(GbnfGrammar grammar, IntFunction<byte[]> tokenUtf8, int eosTokenId,
			int vocabSize) {
		if (grammar == null)
			return null;
		if (tokenUtf8 == null)
			throw new IllegalArgumentException("tokenUtf8 must not be null");
		return new GrammarSession(grammar, tokenUtf8, eosTokenId, vocabSize);
	}

	public boolean complete() {
		return state != null && state.accepting();
	}

	/**
	 * Sets illegal token logits to {@code -inf}. EOS is allowed only when the
	 * grammar is complete. If every token is illegal, EOS is left unmasked so
	 * decode can halt.
	 */
	public void mask(float[] logits) {
		if (logits == null || logits.length == 0 || state == null)
			return;
		boolean any = false;
		boolean[] first = null;
		for (int i = 0; i < logits.length; i++) {
			if (i == eosTokenId) {
				if (state.accepting()) {
					any = true;
				} else {
					logits[i] = NEG_INF;
				}
				continue;
			}
			byte[] piece = piece(i);
			if (piece.length == 0) {
				logits[i] = NEG_INF;
				continue;
			}
			if (first == null)
				first = state.nextBytes();
			if (!first[piece[0] & 0xff]) {
				logits[i] = NEG_INF;
				continue;
			}
			GbnfGrammar.State trial = state.copy();
			if (!trial.consume(piece))
				logits[i] = NEG_INF;
			else
				any = true;
		}
		if (!any && eosTokenId >= 0 && eosTokenId < logits.length)
			logits[eosTokenId] = 0f;
	}

	public void accept(int tokenId) {
		if (state == null)
			return;
		if (tokenId == eosTokenId)
			return;
		byte[] piece = piece(tokenId);
		if (piece.length == 0)
			return;
		if (!state.consume(piece))
			throw new IllegalStateException("sampled token is illegal under the active grammar");
	}

	private byte[] piece(int tokenId) {
		if (tokenId >= 0 && tokenId < cache.length && cache[tokenId] != null)
			return cache[tokenId];
		byte[] p;
		try {
			p = tokenUtf8.apply(tokenId);
		} catch (RuntimeException e) {
			p = EMPTY;
		}
		if (p == null)
			p = EMPTY;
		if (tokenId >= 0 && tokenId < cache.length)
			cache[tokenId] = p;
		return p;
	}
}
