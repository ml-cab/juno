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
package cab.ml.juno.coordinator;

import cab.ml.juno.kvcache.ServeScheduleOptions;

/**
 * Resolves {@code --schedule} against launch topology and running-set caps.
 *
 * <p>v1: {@code continuous} is local / in-process only. Cluster, tensor-parallel,
 * and pipeline-parallel launchers auto-fallback to {@code static} (dense KV +
 * static micro-batch) with a clear warning — they do not silently run paged KV
 * under the static collector.
 */
public final class ServeSchedulePolicy {

	public static final String CLUSTER_FALLBACK_WARNING = "continuous unsupported on cluster; using static";

	/** Running-set cap when {@code --parallel} is unset or 1 under continuous. */
	public static final int DEFAULT_RUNNING_SET = 8;

	public enum Topology {
		/** In-process local JVM ({@code juno local}, JunoPlayer). */
		LOCAL,
		/** Forked / gRPC cluster, including a single remote shard. */
		CLUSTER
	}

	private ServeSchedulePolicy() {
	}

	/**
	 * Effective schedule after topology rules. Cluster + continuous → static.
	 */
	public static Resolution resolve(ServeScheduleOptions requested, Topology topology) {
		if (requested == null)
			requested = ServeScheduleOptions.defaults();
		if (topology == null)
			throw new IllegalArgumentException("topology must not be null");
		if (requested.mode() == ServeScheduleOptions.Mode.CONTINUOUS && topology == Topology.CLUSTER) {
			return new Resolution(ServeScheduleOptions.defaults(), true, CLUSTER_FALLBACK_WARNING);
		}
		return new Resolution(requested, false, null);
	}

	/**
	 * Under continuous, a disabled / size-1 {@link BatchConfig} still gets a
	 * running-set cap so overlapping decode can occur without requiring
	 * {@code --parallel}. Static schedule leaves {@code requested} unchanged.
	 */
	public static BatchConfig runningSetConfig(BatchConfig requested) {
		if (requested == null || !requested.isBatchingEnabled())
			return BatchConfig.defaults();
		return requested;
	}

	public record Resolution(ServeScheduleOptions options, boolean fallback, String warning) {

		public Resolution {
			if (options == null)
				throw new IllegalArgumentException("options must not be null");
		}

		/** Write {@link ServeScheduleOptions#ENV} so handlers see the effective KV path. */
		public void applyToEnv() {
			System.setProperty(ServeScheduleOptions.ENV, options.mode().name().toLowerCase());
		}
	}
}
