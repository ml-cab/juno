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
 * Callback interface for real-time streaming token delivery.
 *
 * Called by GenerationLoop once per generated token, and also notified about
 * the prefill phase so UIs can show a progress indicator.
 *
 * Must be non-blocking — any slow I/O should be handled asynchronously by the
 * implementation. Blocking here stalls the generation loop.
 */
@FunctionalInterface
public interface TokenConsumer {

	/** Singleton no-op consumer for blocking / batch-eligible requests. */
	TokenConsumer DISCARD = new TokenConsumer() {
		@Override
		public void onToken(String piece, int tokenId, int position) {
		}

		@Override
		public boolean batchEligible() {
			return true;
		}
	};

	/**
	 * Called when a new token piece is ready.
	 *
	 * @param piece    decoded text for this token (may be empty for special tokens)
	 * @param tokenId  raw token ID
	 * @param position 0-based position in the generated sequence
	 */
	void onToken(String piece, int tokenId, int position);

	/**
	 * Called once before the prefill loop starts. Default no-op — override to
	 * display a "thinking…" indicator.
	 *
	 * @param promptLen total number of prompt tokens (prefill steps = promptLen -
	 *                  1)
	 */
	default void onPrefillStart(int promptLen) {
	}

	/**
	 * Called once after the prefill loop finishes, immediately before decode
	 * begins. Default no-op — override to clear the "thinking…" indicator.
	 */
	default void onPrefillComplete() {
	}

	/**
	 * Whether this request may join a static micro-batch. Streaming consumers
	 * return false and are dispatched per-request even when batching is enabled.
	 */
	default boolean batchEligible() {
		return false;
	}

	/** No-op consumer — useful for non-streaming (batch) generation. */
	static TokenConsumer discard() {
		return DISCARD;
	}

	/** Collects pieces into a StringBuilder — useful for testing. */
	static TokenConsumer collecting(StringBuilder sb) {
		return (piece, tokenId, position) -> sb.append(piece);
	}
}