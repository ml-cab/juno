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
package cab.ml.juno.registry;

import java.io.Serializable;
import java.time.Instant;

/**
 * Immutable metadata snapshot of a model registered in the cluster. Stored in
 * Hazelcast IMap("model-registry") keyed by modelId.
 *
 * vramPerLayerBytes is derived from model architecture + quantization at
 * construction time — no weight file access needed for shard planning.
 *
 * VRAM estimation formula per transformer layer: params = 4 × hiddenDim² (Q, K,
 * V, O projections — dominant cost) bytes = params ×
 * quantization.bytesPerParam()
 *
 * This is an approximation. Real per-layer cost also includes: - FFN weights
 * (~8 × hiddenDim² for LLaMA-style) - Layer norms, biases (negligible) - KV
 * cache (accounted separately in KVCacheManager) For planning purposes the
 * attention weights dominate and this estimate is accurate enough to prevent
 * over-allocation.
 */
public record ModelDescriptor(String modelId, String architecture, // e.g. "llama", "mistral", "gemma"
		int totalLayers, int hiddenDim, int vocabSize, int numHeads, long vramPerLayerBytes,
		QuantizationType quantization, String path, // local filesystem path to GGUF file
		ModelStatus status, Instant registeredAt) implements Serializable {

	public ModelDescriptor {
		if (modelId == null || modelId.isBlank())
			throw new IllegalArgumentException("modelId must not be blank");
		if (architecture == null || architecture.isBlank())
			throw new IllegalArgumentException("architecture must not be blank");
		if (totalLayers < 1)
			throw new IllegalArgumentException("totalLayers must be >= 1, got: " + totalLayers);
		if (hiddenDim < 1)
			throw new IllegalArgumentException("hiddenDim must be >= 1, got: " + hiddenDim);
		if (vocabSize < 1)
			throw new IllegalArgumentException("vocabSize must be >= 1, got: " + vocabSize);
		if (numHeads < 1)
			throw new IllegalArgumentException("numHeads must be >= 1, got: " + numHeads);
		if (vramPerLayerBytes < 1)
			throw new IllegalArgumentException("vramPerLayerBytes must be >= 1");
		if (quantization == null)
			throw new IllegalArgumentException("quantization must not be null");
		if (path == null || path.isBlank())
			throw new IllegalArgumentException("path must not be blank");
		if (status == null)
			throw new IllegalArgumentException("status must not be null");
	}

	/**
	 * Primary factory method — derives vramPerLayerBytes from architecture. Initial
	 * status is always UNLOADED.
	 */
	public static ModelDescriptor of(String modelId, String architecture, int totalLayers, int hiddenDim, int vocabSize,
			int numHeads, QuantizationType quantization, String path) {

		long vramPerLayer = estimateVramPerLayer(hiddenDim, quantization);
		return new ModelDescriptor(modelId, architecture, totalLayers, hiddenDim, vocabSize, numHeads, vramPerLayer,
				quantization, path, ModelStatus.UNLOADED, Instant.now());
	}

	// ── Derived queries ───────────────────────────────────────────────────────

	/** Total estimated VRAM to load the full model (all layers). */
	public long totalVramBytes() {
		return vramPerLayerBytes * totalLayers;
	}

	/** Human-readable size string e.g. "3.2 GB". */
	public String humanReadableSize() {
		double gb = totalVramBytes() / (1024.0 * 1024 * 1024);
		if (gb >= 1.0)
			return String.format("%.1f GB", gb);
		double mb = totalVramBytes() / (1024.0 * 1024.0);
		return String.format("%.0f MB", mb);
	}

	// ── Fluent updaters ───────────────────────────────────────────────────────

	public ModelDescriptor withStatus(ModelStatus status) {
		return new ModelDescriptor(modelId, architecture, totalLayers, hiddenDim, vocabSize, numHeads,
				vramPerLayerBytes, quantization, path, status, registeredAt);
	}

	// ── Private ───────────────────────────────────────────────────────────────

	/**
	 * Estimate VRAM required per transformer layer.
	 *
	 * Dominant cost per layer: 4 attention projection matrices (Q/K/V/O), each of
	 * size hiddenDim × hiddenDim.
	 *
	 * params = 4 × hiddenDim² bytes = params × quantization.bytesPerParam()
	 */
	private static long estimateVramPerLayer(int hiddenDim, QuantizationType quantization) {
		long params = 4L * hiddenDim * hiddenDim;
		return (long) (params * quantization.bytesPerParam());
	}
}
