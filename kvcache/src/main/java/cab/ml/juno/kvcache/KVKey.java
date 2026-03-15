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
 * Composite cache key — requestId + layerIndex.
 *
 * Each inference node caches KV pairs only for its own assigned layers. The
 * layerIndex identifies which transformer layer this block belongs to.
 */
public record KVKey(String requestId, int layerIndex) {

	public KVKey {
		if (requestId == null || requestId.isBlank())
			throw new IllegalArgumentException("requestId must not be blank");
		if (layerIndex < 0)
			throw new IllegalArgumentException("layerIndex must be >= 0");
	}

	@Override
	public String toString() {
		return requestId + "@layer" + layerIndex;
	}
}
