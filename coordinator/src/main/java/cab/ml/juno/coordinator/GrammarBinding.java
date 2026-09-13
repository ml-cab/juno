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

import java.nio.charset.StandardCharsets;

import cab.ml.juno.sampler.GrammarSession;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.Tokenizer;

/**
 * Opens a per-sequence grammar cursor from tokenizer pieces. Unconstrained
 * sampling when {@link SamplingParams#grammar()} is absent.
 */
final class GrammarBinding {

	private GrammarBinding() {
	}

	static GrammarSession open(Tokenizer tokenizer, SamplingParams params) {
		return open(tokenizer, params, null);
	}

	static GrammarSession open(Tokenizer tokenizer, SamplingParams params, String requestId) {
		if (tokenizer == null || params == null || params.grammar() == null)
			return null;
		GrammarSession session = GrammarSession.open(params.grammar(), id -> utf8(tokenizer.decodeToken(id)),
				tokenizer.eosTokenId(), tokenizer.vocabSize());
		if (session != null) {
			GrammarConstrainedEvent ev = new GrammarConstrainedEvent();
			ev.requestId = requestId != null ? requestId : "";
			ev.commit();
		}
		return session;
	}

	private static byte[] utf8(String piece) {
		if (piece == null || piece.isEmpty())
			return new byte[0];
		return piece.getBytes(StandardCharsets.UTF_8);
	}
}
