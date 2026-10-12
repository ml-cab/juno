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
package cab.ml.juno.kvcache;

/**
 * Holds one released request's dense KV layers (one side, K or V) for the next request
 * to take instead of allocating its own. A long prompt's layers are hundreds of megabytes,
 * and allocating them for every request costs the zeroing and the garbage collections
 * the allocation triggers. Holds at most one set, the one with the larger capacity, so an
 * idle handler keeps no more than one request's worth. Thread-safe.
 */
final class DenseKvRecycler {

	private DenseKvTensor[] held;

	/** The held layers if there are {@code layerCount} of {@code type}, ready for a new request; else null. */
	synchronized SessionKvTensor[] take(int layerCount, KvElementType type) {
		DenseKvTensor[] h = held;
		if (h == null || h.length != layerCount || h[0].type() != type)
			return null;
		held = null;
		for (DenseKvTensor t : h)
			t.reuse();
		return h;
	}

	/** Keeps {@code layers} if they are dense and hold more than the set already held. */
	synchronized void offer(SessionKvTensor[] layers) {
		if (layers == null || layers.length == 0)
			return;
		DenseKvTensor[] dense = new DenseKvTensor[layers.length];
		for (int i = 0; i < layers.length; i++) {
			if (!(layers[i] instanceof DenseKvTensor d))
				return;
			dense[i] = d;
		}
		if (held == null || dense[0].capacityTokens() > held[0].capacityTokens())
			held = dense;
	}
}
