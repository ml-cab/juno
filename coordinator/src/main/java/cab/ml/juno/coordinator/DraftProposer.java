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

/**
 * Source of speculative-decoding draft tokens for {@link GenerationLoop#generate}.
 * {@link NgramDraftCache} (ngram-simple) and {@link DraftModelSession}
 * (draft-simple) are the two implementations; both share the same propose/observe
 * contract so the draft/verify loop in {@link GenerationLoop} does not need to
 * know which drafting strategy is active.
 */
interface DraftProposer {

	/**
	 * Propose up to {@code maxDraft} future tokens continuing from the end of
	 * {@code allTokens}.
	 *
	 * @return a draft of length {@code [0, maxDraft]}
	 */
	int[] propose(int[] allTokens, int maxDraft);

	/**
	 * Reconcile internal state against the now-known ground truth after a
	 * round's tokens have been verified and emitted (whether or not they
	 * matched the draft).
	 */
	void observe(int[] allTokens);

	/**
	 * Release any per-request resources (e.g. a draft model's own KV state).
	 * Called once at the end of {@link GenerationLoop#generate}. Default is a
	 * no-op for proposers that hold no external resources.
	 */
	default void close() {
	}
}
