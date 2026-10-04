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

import cab.ml.juno.tokenizer.Tokenizer;

/**
 * Encodes a request's formatted prompt, timed as one {@link PromptEncodeEvent}. Every generation
 * path encodes through here, so a request's encoding time is recorded wherever it is served: it
 * runs before the first forward pass, outside every forward-pass span, and on long prompts it is
 * hundreds of milliseconds.
 */
final class PromptEncoder {

	private PromptEncoder() {
	}

	static int[] encode(Tokenizer tokenizer, String prompt, String requestId) {
		PromptEncodeEvent ev = new PromptEncodeEvent();
		ev.begin();
		int[] ids = tokenizer.encode(prompt);
		if (ev.shouldCommit()) {
			ev.requestId = requestId;
			ev.characters = prompt.length();
			ev.tokens = ids.length;
			ev.commit();
		}
		return ids;
	}
}
