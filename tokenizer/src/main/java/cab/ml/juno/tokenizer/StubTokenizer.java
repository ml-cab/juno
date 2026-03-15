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
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Lightweight test-double tokenizer for unit and integration testing.
 *
 * Does NOT require DJL or any model file. Splits text on whitespace and assigns
 * deterministic integer IDs. Sufficient for testing everything that uses the
 * Tokenizer interface (coordinator, pipeline, sampler integration) without
 * touching real model files.
 *
 * NOT for production use.
 */
public final class StubTokenizer implements Tokenizer {

	private static final int BOS = 1;
	private static final int EOS = 2;
	private static final int PAD = 0;

	private final Map<String, Integer> vocab = new ConcurrentHashMap<>();
	private final Map<Integer, String> reverse = new ConcurrentHashMap<>();
	private final AtomicInteger nextId = new AtomicInteger(10);

	// IDs 3-9 are pre-registered stub response words.
	// CyclicForwardPassHandler rotates its winner token through this range so that
	// decoded output is always visible without depending on prompt vocabulary.
	// nextId starts at 10, so dynamic encoding never collides with these IDs.
	static final int STUB_WORD_FIRST = 3;
	static final int STUB_WORD_LAST = 9;

	private static final String[] STUB_WORDS = { "the", "quick", "brown", "fox", "jumps", "over", "lazy" };

	public StubTokenizer() {
		// pre-register special tokens
		register("<pad>", PAD);
		register("<bos>", BOS);
		register("<eos>", EOS);
		// pre-register stub response words at fixed IDs 3-9
		for (int i = 0; i < STUB_WORDS.length; i++) {
			register(STUB_WORDS[i], STUB_WORD_FIRST + i);
		}
	}

	private void register(String token, int id) {
		vocab.put(token, id);
		reverse.put(id, token);
	}

	private int getOrCreate(String token) {
		return vocab.computeIfAbsent(token, t -> {
			int id = nextId.getAndIncrement();
			reverse.put(id, t);
			return id;
		});
	}

	@Override
	public int[] encode(String text) {
		if (text == null || text.isBlank())
			return new int[0];
		String[] words = text.strip().split("\\s+");
		int[] ids = new int[words.length];
		for (int i = 0; i < words.length; i++) {
			ids[i] = getOrCreate(words[i]);
		}
		return ids;
	}

	@Override
	public String decode(int[] tokenIds) {
		if (tokenIds == null || tokenIds.length == 0)
			return "";
		List<String> parts = new ArrayList<>();
		for (int id : tokenIds) {
			String token = reverse.getOrDefault(id, "<unk>");
			if (!token.startsWith("<") || !token.endsWith(">")) {
				parts.add(token);
			}
		}
		return String.join(" ", parts);
	}

	@Override
	public String decodeToken(int tokenId) {
		String token = reverse.get(tokenId);
		if (token == null) {
			// Token not yet in vocab (e.g. stub winner from CyclicForwardPassHandler).
			// Return a visible placeholder so streaming output is never invisible.
			return "tok" + tokenId + " ";
		}
		// suppress special tokens for streaming
		if (token.startsWith("<") && token.endsWith(">"))
			return "";
		return token + " ";
	}

	@Override
	public int bosTokenId() {
		return BOS;
	}

	@Override
	public int eosTokenId() {
		return EOS;
	}

	@Override
	public int padTokenId() {
		return PAD;
	}

	@Override
	public int vocabSize() {
		return vocab.size();
	}

	@Override
	public String modelType() {
		return "stub";
	}

	@Override
	public boolean isReady() {
		return true;
	}
}