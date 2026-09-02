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
package cab.ml.juno.node;

/**
 * Shared JFR wrapper for {@link InferencePipeline#prefillBatch} implementations.
 */
public final class PrefillBatchJfr {

	private PrefillBatchJfr() {
	}

	public static void run(String requestId, int[] newTokens, int startPosition, Runnable body) {
		if (newTokens.length == 0) {
			body.run();
			return;
		}
		PrefillBatchForwardEvent evt = new PrefillBatchForwardEvent();
		evt.requestId = requestId;
		evt.windowSize = newTokens.length;
		evt.startPosition = startPosition;
		evt.begin();
		try {
			body.run();
		} finally {
			evt.commit();
		}
	}
}
