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

import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Which requests a handler is running a pass for right now. A handler's evict hands the
 * request's host KV tensors to the next request only when no pass of it is running:
 * a pass that outlived its caller (a cluster retry after a timeout broadcasts the evict
 * to every node) would otherwise go on writing into tensors another request now owns.
 * A pass enters before it reads the request's tensors and an evict removes them before
 * it asks, so a pass either is counted or finds them gone. One map update per pass on
 * entry and one on exit. Thread-safe.
 */
final class InFlightRequests {

	private final ConcurrentHashMap<String, int[]> passes = new ConcurrentHashMap<>();

	void enter(String requestId) {
		passes.compute(requestId, (id, n) -> {
			if (n == null)
				return new int[] { 1 };
			n[0]++;
			return n;
		});
	}

	void exit(String requestId) {
		passes.computeIfPresent(requestId, (id, n) -> --n[0] == 0 ? null : n);
	}

	void enterAll(List<String> requestIds) {
		for (String id : requestIds)
			enter(id);
	}

	void exitAll(List<String> requestIds) {
		for (String id : requestIds)
			exit(id);
	}

	/** Whether no pass of {@code requestId} is running. */
	boolean idle(String requestId) {
		return !passes.containsKey(requestId);
	}
}
