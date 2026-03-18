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

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;

/**
 * Central model registry — stores model metadata and shard maps, drives the
 * ShardPlanner when models are registered.
 *
 * Backed by ConcurrentHashMap for now. Hazelcast IMap upgrade path: replace the
 * two maps with IMap("model-registry") and IMap("shard-maps") — the interface
 * is identical.
 *
 * Lifecycle: register() — plan shards, store model (LOADING), store ShardMap
 * markLoaded() — transition model to LOADED after nodes confirm markError() —
 * transition model to ERROR on node failure unregister() — remove model +
 * ShardMap (nodes must unload separately)
 *
 * Thread-safe — all mutations are on ConcurrentHashMap.
 */
public final class ModelRegistry {

	private static final Logger log = Logger.getLogger(ModelRegistry.class.getName());

	private final ShardPlanner planner;

	private final ConcurrentHashMap<String, ModelDescriptor> models = new ConcurrentHashMap<>();
	private final ConcurrentHashMap<String, ShardMap> shardMaps = new ConcurrentHashMap<>();

	public ModelRegistry(ShardPlanner planner) {
		if (planner == null)
			throw new IllegalArgumentException("planner must not be null");
		this.planner = planner;
	}

	/**
	 * Register a model and compute its shard map against the given nodes.
	 *
	 * The model is stored with status LOADING — caller must call markLoaded() once
	 * all nodes confirm their shard is ready.
	 *
	 * If a model with the same id already exists, it is replaced.
	 *
	 * @param model model metadata
	 * @param nodes all available nodes (ShardPlanner filters eligible ones)
	 * @return the computed ShardMap
	 * @throws InsufficientClusterVramException if cluster cannot fit the model
	 */
	public ShardMap register(ModelDescriptor model, List<NodeDescriptor> nodes) {
		ShardMap map = planner.plan(model.modelId(), model.totalLayers(), model.vramPerLayerBytes(), nodes);

		ModelDescriptor loading = model.withStatus(ModelStatus.LOADING);
		models.put(model.modelId(), loading);
		shardMaps.put(model.modelId(), map);

		log.info(String.format("Registered model '%s' (%s %s) across %d nodes — %s total VRAM", model.modelId(),
				model.architecture(), model.quantization().displayName(), map.nodeCount(), model.humanReadableSize()));

		return map;
	}

	/**
	 * Remove a model and its shard map from the registry. Nodes must be notified
	 * separately to unload their shards. No-op if the model is not registered.
	 */
	public void unregister(String modelId) {
		ModelDescriptor removed = models.remove(modelId);
		shardMaps.remove(modelId);
		if (removed != null) {
			log.info("Unregistered model '" + modelId + "'");
		}
	}

	/**
	 * Transition a model to LOADED status. Called by the coordinator once all nodes
	 * confirm shard readiness. No-op if the model is not registered.
	 */
	public void markLoaded(String modelId) {
		models.computeIfPresent(modelId, (_, m) -> m.withStatus(ModelStatus.LOADED));
	}

	/**
	 * Transition a model to ERROR status. Called when one or more nodes fail to
	 * load their shard. No-op if the model is not registered.
	 *
	 * @param reason human-readable reason for logging
	 */
	public void markError(String modelId, String reason) {
		models.computeIfPresent(modelId, (_, m) -> m.withStatus(ModelStatus.ERROR));
		if (models.containsKey(modelId)) {
			log.warning(String.format("Model '%s' entered ERROR state: %s", modelId, reason));
		}
	}

	// ── Queries ───────────────────────────────────────────────────────────────

	public Optional<ModelDescriptor> getModel(String modelId) {
		return Optional.ofNullable(models.get(modelId));
	}

	public Optional<ShardMap> getShardMap(String modelId) {
		return Optional.ofNullable(shardMaps.get(modelId));
	}

	/**
	 * True only when the model is in LOADED status. LOADING and ERROR both return
	 * false.
	 */
	public boolean isLoaded(String modelId) {
		ModelDescriptor m = models.get(modelId);
		return m != null && m.status() == ModelStatus.LOADED;
	}

	/** Snapshot of all registered models (any status). */
	public List<ModelDescriptor> listModels() {
		return new ArrayList<>(models.values());
	}

	public int modelCount() {
		return models.size();
	}
}
