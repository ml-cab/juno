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

import java.util.Random;

import cab.ml.juno.lora.LoraAdapter;
import cab.ml.juno.lora.LoraAdapterSet;

/**
 * Builds a standard query/value LoRA adapter set from model dimensions.
 */
public final class LoraQvInitializer {

	private LoraQvInitializer() {
	}

	/**
	 * Create adapters on {@code wq} and {@code wv} for every layer.
	 */
	public static LoraAdapterSet qv(LlamaConfig cfg, int rank, float alpha, Random rng) {
		LoraAdapterSet set = new LoraAdapterSet();
		for (int li = 0; li < cfg.numLayers(); li++) {
			set.add(li, "wq", new LoraAdapter(rank, cfg.hiddenDim(), cfg.hiddenDim(), alpha, rng));
			set.add(li, "wv", new LoraAdapter(rank, cfg.hiddenDim(), cfg.kvDim(), alpha, rng));
		}
		return set;
	}
}
