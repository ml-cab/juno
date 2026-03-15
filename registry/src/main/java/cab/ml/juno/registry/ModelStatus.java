/*
 * Copyright 2026 Dmytro Soloviov (soulaway)
 * SPDX-License-Identifier: Apache-2.0
 */

package cab.ml.juno.registry;

/**
 * Lifecycle state of a model within the registry.
 *
 * Transitions: UNLOADED → LOADING (register() called — shards being sent to
 * nodes) LOADING → LOADED (all nodes confirm shard is ready) LOADING → ERROR
 * (any node fails to load its shard) LOADED → UNLOADED (unregister() called)
 * ERROR → LOADING (retry — register() called again)
 */
public enum ModelStatus {

	/** Model is known to the registry but no shards are loaded on any node. */
	UNLOADED,

	/** Shards are being transferred to / loaded by nodes. */
	LOADING,

	/** All nodes have confirmed their shard is loaded and ready. */
	LOADED,

	/** One or more nodes failed to load their shard. */
	ERROR
}
