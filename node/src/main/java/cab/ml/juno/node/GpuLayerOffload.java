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

import java.util.Locale;

/**
 * Policy for hybrid GPU layer offload ({@code --gpu-layers} / {@code JUNO_GPU_LAYERS}).
 *
 * <p>Global transformer layer index {@code g} (0-based) is GPU-resident when
 * {@code g < resolvedCount()}. Output projection is GPU-resident only when all
 * transformer layers are resident ({@code resolvedCount() >= totalLayers}).
 *
 * <p>Cluster shards apply the policy to their local layer range using global indices
 * ({@code startLayer + localIndex}).
 */
public final class GpuLayerOffload {

	public static final String ENV_PROPERTY = "JUNO_GPU_LAYERS";
	public static final String ALL = "all";
	public static final String AUTO = "auto";

	public enum Mode {
		ALL, NONE, COUNT, AUTO
	}

	private final Mode mode;
	private final int requestedCount;
	private final int autoResolvedCount;

	private GpuLayerOffload(Mode mode, int requestedCount, int autoResolvedCount) {
		this.mode = mode;
		this.requestedCount = requestedCount;
		this.autoResolvedCount = autoResolvedCount;
	}

	public static GpuLayerOffload all() {
		return new GpuLayerOffload(Mode.ALL, Integer.MAX_VALUE, Integer.MAX_VALUE);
	}

	public static GpuLayerOffload none() {
		return new GpuLayerOffload(Mode.NONE, 0, 0);
	}

	public static GpuLayerOffload count(int n) {
		if (n < 0)
			throw new IllegalArgumentException("gpu-layers count must be >= 0 (got " + n + ")");
		return new GpuLayerOffload(Mode.COUNT, n, n);
	}

	public static GpuLayerOffload auto() {
		return new GpuLayerOffload(Mode.AUTO, 0, 0);
	}

	/** Result of {@link #auto()} after VRAM-fit upload completes on a handler. */
	public GpuLayerOffload withAutoResolved(int resolvedGlobalLayers) {
		if (mode != Mode.AUTO)
			throw new IllegalStateException("withAutoResolved only valid for auto mode");
		int r = Math.max(0, resolvedGlobalLayers);
		return new GpuLayerOffload(Mode.AUTO, r, r);
	}

	/**
	 * Parse CLI / env value: {@code all|auto|0|N}.
	 *
	 * @param spec raw value; blank → {@code all}
	 */
	public static GpuLayerOffload parse(String spec) {
		if (spec == null || spec.isBlank())
			return all();
		String s = spec.strip().toLowerCase(Locale.ROOT);
		if (ALL.equals(s))
			return all();
		if (AUTO.equals(s))
			return auto();
		try {
			return count(Integer.parseInt(s));
		} catch (NumberFormatException e) {
			throw new IllegalArgumentException(
					"--gpu-layers must be all|auto|0|N (got " + spec + ")");
		}
	}

	/** Reads {@link #ENV_PROPERTY} (system property, falling back to the env var); defaults to {@code all}. */
	public static GpuLayerOffload fromEnv() {
		String raw = firstNonBlank(System.getProperty(ENV_PROPERTY), System.getenv(ENV_PROPERTY));
		return parse(raw == null ? ALL : raw);
	}

	private static String firstNonBlank(String a, String b) {
		if (a != null && !a.isBlank())
			return a;
		if (b != null && !b.isBlank())
			return b;
		return null;
	}

	public Mode mode() {
		return mode;
	}

	public boolean isAuto() {
		return mode == Mode.AUTO;
	}

	/**
	 * Exclusive upper bound on global layer indices kept on GPU
	 * ({@code [0, resolvedCount)}).
	 */
	public int resolvedCount(int totalLayers) {
		return switch (mode) {
		case ALL -> totalLayers;
		case NONE -> 0;
		case COUNT -> Math.min(Math.max(0, requestedCount), totalLayers);
		case AUTO -> Math.min(Math.max(0, autoResolvedCount), totalLayers);
		};
	}

	public boolean residentForGlobalLayer(int globalLayerIndex, int totalLayers) {
		return globalLayerIndex >= 0 && globalLayerIndex < resolvedCount(totalLayers);
	}

	public boolean residentOutputProjection(int totalLayers) {
		return resolvedCount(totalLayers) >= totalLayers;
	}

	/** JFR / metrics label for the active policy. */
	public String policyLabel(int totalLayers) {
		return switch (mode) {
		case ALL -> ALL;
		case NONE -> "0";
		case AUTO -> AUTO + ":" + resolvedCount(totalLayers);
		case COUNT -> String.valueOf(resolvedCount(totalLayers));
		};
	}

	/**
	 * True when an {@link IllegalStateException} from weight upload is likely VRAM OOM.
	 */
	public static boolean isVramOom(IllegalStateException ex) {
		String msg = ex.getMessage();
		if (msg == null)
			return false;
		return msg.contains("cudaMalloc") || msg.contains("hipMalloc");
	}
}
