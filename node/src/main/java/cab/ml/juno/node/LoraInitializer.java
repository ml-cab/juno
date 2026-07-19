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

import java.util.Collection;
import java.util.List;
import java.util.Random;

import cab.ml.juno.lora.LoraAdapter;
import cab.ml.juno.lora.LoraAdapterSet;

/**
 * Creates LoRA adapters in stable layer/projection order and validates loaded
 * checkpoints against model dimensions.
 */
public final class LoraInitializer {

	private LoraInitializer() {
	}

	/**
	 * Create adapters for every layer and every projection in {@code targets},
	 * ordered by increasing layer index then {@link LoraProjection} enum order.
	 */
	public static LoraAdapterSet create(LlamaConfig cfg, Collection<LoraProjection> targets, int rank, float alpha,
			Random rng) {
		List<LoraProjection> ordered = LoraProjection.sortedUnique(targets);
		if (ordered.isEmpty())
			throw new IllegalArgumentException("targets must not be empty");
		if (rank < 1)
			throw new IllegalArgumentException("rank must be >= 1");

		LoraAdapterSet set = new LoraAdapterSet();
		for (int li = 0; li < cfg.numLayers(); li++) {
			for (LoraProjection proj : ordered) {
				set.add(li, proj.key(),
						new LoraAdapter(rank, proj.inDim(cfg), proj.outDim(cfg), alpha, rng));
			}
		}
		return set;
	}

	/** Convenience: parse target spec then {@link #create}. */
	public static LoraAdapterSet create(LlamaConfig cfg, String targetSpec, int rank, float alpha, Random rng) {
		return create(cfg, LoraProjection.parseTargets(targetSpec), rank, alpha, rng);
	}

	/**
	 * Validate that every adapter key, layer, and shape matches {@code cfg}.
	 *
	 * @throws IllegalArgumentException on mismatch or unknown projection
	 */
	public static void validate(LoraAdapterSet adapters, LlamaConfig cfg) {
		if (adapters == null || adapters.size() == 0)
			throw new IllegalArgumentException("adapter set is empty");

		for (var entry : adapters.asMap().entrySet()) {
			String key = entry.getKey();
			int layer;
			String projKey;
			try {
				layer = LoraAdapterSet.keyLayer(key);
				projKey = LoraAdapterSet.keyProj(key);
			} catch (RuntimeException e) {
				throw new IllegalArgumentException("invalid adapter key: " + key, e);
			}
			if (layer < 0 || layer >= cfg.numLayers())
				throw new IllegalArgumentException(
						"adapter layer " + layer + " out of range [0," + cfg.numLayers() + ")");

			LoraProjection proj = LoraProjection.fromKey(projKey);
			LoraAdapter a = entry.getValue();
			int expectIn = proj.inDim(cfg);
			int expectOut = proj.outDim(cfg);
			if (a.inDim != expectIn || a.outDim != expectOut)
				throw new IllegalArgumentException("adapter " + key + " shape " + a.outDim + "×" + a.inDim
						+ " does not match model " + expectOut + "×" + expectIn);
		}
	}
}
