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

import java.io.Serializable;

/**
 * Contiguous range of transformer layer indices owned by a single node.
 *
 * startLayer is inclusive, endLayer is exclusive — same convention as
 * ShardAssignment in the registry module.
 *
 * LayerRange.all() is a sentinel meaning "no restriction" — used by the
 * coordinator's KVCacheManager for prefix cache operations and backward
 * compatibility with single-node setups.
 *
 * Design contract: Node 1 owns LayerRange.of(0, 8) → stores KV for layers 0–7
 * Node 2 owns LayerRange.of(8, 16) → stores KV for layers 8–15 Node 3 owns
 * LayerRange.of(16, 22) → stores KV for layers 16–21
 *
 * KVCacheManager enforces this at put() time — wrong-range puts are bugs, not
 * recoverable conditions, so they throw IllegalArgumentException.
 */
public record LayerRange(int startLayer, int endLayer, boolean unbounded) implements Serializable {

	public LayerRange {
		if (!unbounded) {
			if (startLayer < 0)
				throw new IllegalArgumentException("startLayer must be >= 0, got: " + startLayer);
			if (endLayer <= startLayer)
				throw new IllegalArgumentException(
						"startLayer must be < endLayer, got: [" + startLayer + ", " + endLayer + ")");
		}
	}

	/**
	 * Bounded range — node owns layers [startLayer, endLayer).
	 */
	public static LayerRange of(int startLayer, int endLayer) {
		return new LayerRange(startLayer, endLayer, false);
	}

	/**
	 * Unbounded sentinel — accepts any layer index. Used by coordinator-side
	 * KVCacheManager for prefix cache + single-node setups.
	 */
	public static LayerRange all() {
		return new LayerRange(0, Integer.MAX_VALUE, true);
	}

	/**
	 * Whether this range covers the given layer index. Always true for
	 * LayerRange.all().
	 */
	public boolean contains(int layerIndex) {
		if (unbounded)
			return layerIndex >= 0;
		return layerIndex >= startLayer && layerIndex < endLayer;
	}

	/**
	 * Whether this range is the unbounded sentinel.
	 */
	public boolean isUnbounded() {
		return unbounded;
	}

	/**
	 * Number of layers in this range. Undefined for unbounded ranges.
	 */
	public int layerCount() {
		if (unbounded)
			return Integer.MAX_VALUE;
		return endLayer - startLayer;
	}

	@Override
	public String toString() {
		if (unbounded)
			return "LayerRange[all]";
		return "LayerRange[" + startLayer + ", " + endLayer + ")";
	}
}
