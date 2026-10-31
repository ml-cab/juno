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
package cab.ml.juno.kit;


import java.util.List;
import java.util.Locale;

/**
 * The numbered notes the context-policy and long-prompt smokes build their prompts from: distinct
 * sentences of about 24 tokens each, so a prompt of a chosen length is a count of notes, and a question
 * about note 1 has one right answer.
 */
final class LedgerNotes {

	static final List<String> PLACES = List.of("harbour", "orchard", "library", "bridge", "market", "lighthouse",
			"mill", "chapel");
	static final List<String> COLOURS = List.of("red", "green", "blue", "white", "yellow", "grey", "black",
			"orange");
	static final List<String> FILLER = List.of("please", "read", "every", "note", "above", "with", "care",
			"before", "you", "answer");

	private LedgerNotes() {
	}

	/** Notes {@code first} to {@code first + count - 1}, one per line. */
	static String notes(int first, int count) {
		StringBuilder b = new StringBuilder();
		for (int i = first; i < first + count; i++)
			b.append(String.format(Locale.ROOT, "Note %d: the %s in district %d was painted %s in the year %d.\n", i,
					PLACES.get(i % 8), i * 7 % 31, COLOURS.get(i * 3 % 8), 1800 + i * 13));
		return b.toString();
	}

	/** The long-prompt smoke's prompt: {@code n} notes, {@code fillerWords} filler words, a question on note 1. */
	static String question(int n, int fillerWords) {
		StringBuilder b = new StringBuilder("Read the notes below, then answer the question at the end.\n");
		b.append(notes(1, n));
		if (fillerWords > 0) {
			b.append("Remark:");
			for (int i = 0; i < fillerWords; i++)
				b.append(' ').append(FILLER.get(i % FILLER.size()));
			b.append(".\n");
		}
		b.append("Question: according to note 1, what colour was the ").append(PLACES.get(1))
				.append(" painted? Answer in one sentence.");
		return b.toString();
	}
}
