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

/**
 * The startup statement of a cluster serving with batching on. A cluster's pipeline clients have no batched forward
 * pass: {@code InferencePipeline.forwardBatch}'s default sends the requests of a decode step one after another, so a
 * batch reaches each node as single requests and the batched device decode, which only the local pipeline calls, does
 * not run.
 */
public final class ClusterBatchNotice {

	public static final String TEXT = "--parallel on a cluster decodes a batch one request at a time on each node, "
			+ "which measures slower than --parallel 1: the batched device decode runs in local mode only. "
			+ "Replies do not change";

	private ClusterBatchNotice() {
	}
}
