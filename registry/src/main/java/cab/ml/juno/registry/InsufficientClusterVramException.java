/*
 * Copyright 2026 Dmytro Soloviov (soulaway)
 * SPDX-License-Identifier: Apache-2.0
 */

package cab.ml.juno.registry;

/**
 * Thrown when the cluster does not have enough free VRAM to accommodate all
 * layers of the requested model.
 */
public final class InsufficientClusterVramException extends RuntimeException {

	public InsufficientClusterVramException(String message) {
		super(message);
	}
}
