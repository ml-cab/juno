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

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * GPT-2 / tiktoken byte↔unicode mapping used by Qwen, Llama 3+, and other
 * {@code tokenizer.ggml.model=gpt2} GGUF tokenizers.
 *
 * <p>
 * Vocab pieces store UTF-8 bytes as unicode codepoints (e.g. {@code å¥½çļĦ} →
 * {@code 好的}). Encoding and decoding must round-trip through raw bytes.
 */
final class Gpt2ByteCodec {

	private static final Map<Integer, Integer> BYTE_TO_UNICODE;
	private static final Map<Integer, Integer> UNICODE_TO_BYTE;

	static {
		List<Integer> bs = new ArrayList<>();
		for (int i = '!'; i <= '~'; i++)
			bs.add(i);
		for (int i = 0xA1; i <= 0xAC; i++)
			bs.add(i);
		for (int i = 0xAE; i <= 0xFF; i++)
			bs.add(i);
		List<Integer> cs = new ArrayList<>(bs);
		int n = 0;
		for (int b = 0; b < 256; b++) {
			if (!bs.contains(b)) {
				bs.add(b);
				cs.add(256 + n);
				n++;
			}
		}
		Map<Integer, Integer> b2u = new HashMap<>(256);
		Map<Integer, Integer> u2b = new HashMap<>(256);
		for (int i = 0; i < bs.size(); i++) {
			b2u.put(bs.get(i), cs.get(i));
			u2b.put(cs.get(i), bs.get(i));
		}
		BYTE_TO_UNICODE = Collections.unmodifiableMap(b2u);
		UNICODE_TO_BYTE = Collections.unmodifiableMap(u2b);
	}

	private Gpt2ByteCodec() {
	}

	/** UTF-8 text → GPT-2 BPE character sequence (one char per input byte). */
	static String textToBpeChars(String text) {
		byte[] utf8 = text.getBytes(StandardCharsets.UTF_8);
		StringBuilder sb = new StringBuilder(utf8.length);
		for (byte b : utf8)
			sb.appendCodePoint(BYTE_TO_UNICODE.get(b & 0xFF));
		return sb.toString();
	}

	static void appendPieceBytes(String piece, ByteArrayOutputStream out) {
		for (int i = 0; i < piece.length();) {
			int cp = piece.codePointAt(i);
			Integer b = UNICODE_TO_BYTE.get(cp);
			if (b != null)
				out.write(b);
			else if (cp < 0x80)
				out.write(cp);
			i += Character.charCount(cp);
		}
	}

	static String decodeBytes(byte[] bytes) {
		return new String(bytes, 0, completeUtf8PrefixLen(bytes), StandardCharsets.UTF_8);
	}

	static int completeUtf8PrefixLen(byte[] bytes) {
		int i = 0;
		while (i < bytes.length) {
			int c = bytes[i] & 0xFF;
			int need;
			if (c < 0x80)
				need = 1;
			else if ((c & 0xE0) == 0xC0)
				need = 2;
			else if ((c & 0xF0) == 0xE0)
				need = 3;
			else if ((c & 0xF8) == 0xF0)
				need = 4;
			else {
				i++;
				continue;
			}
			if (i + need > bytes.length)
				break;
			i += need;
		}
		return i;
	}

	/** Incremental UTF-8 decoder for streaming token output. */
	static final class Stream {
		private final ByteArrayOutputStream pending = new ByteArrayOutputStream();
		private int emittedPrefixLen;

		String appendPiece(String piece) {
			appendPieceBytes(piece, pending);
			byte[] all = pending.toByteArray();
			int safe = completeUtf8PrefixLen(all);
			if (safe <= emittedPrefixLen)
				return "";
			String text = new String(all, 0, safe, StandardCharsets.UTF_8);
			int emittedChars = new String(all, 0, emittedPrefixLen, StandardCharsets.UTF_8).length();
			String delta = text.substring(emittedChars);
			emittedPrefixLen = safe;
			return delta;
		}

		String flush() {
			byte[] all = pending.toByteArray();
			if (all.length <= emittedPrefixLen)
				return "";
			String text = new String(all, emittedPrefixLen, all.length - emittedPrefixLen,
					StandardCharsets.UTF_8);
			emittedPrefixLen = all.length;
			return text;
		}
	}
}
