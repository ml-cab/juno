/*
 * Copyright 2026 Dmytro Soloviov (soulaway)
 * SPDX-License-Identifier: Apache-2.0
 */

package cab.ml.juno.registry;

/**
 * Lifecycle state of an inference node.
 *
 * Transitions: IDLE → LOADING → READY → DEGRADED → IDLE (on unload) ↘ OFFLINE
 * (on failure / timeout)
 */
public enum NodeStatus {

	/** Node is registered but no model shard is loaded. */
	IDLE,

	/** Node is loading a model shard into GPU VRAM. */
	LOADING,

	/** Node has a shard loaded and is ready to serve forward passes. */
	READY,

	/** Node is responding but GPU health is degraded (high temp, high pressure). */
	DEGRADED,

	/** Node is not reachable — presumed failed. Triggers resharding. */
	OFFLINE
}
