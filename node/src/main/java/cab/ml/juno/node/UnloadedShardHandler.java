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

/**
 * Handler for a node that was started with a real model but has no loaded shard:
 * either no {@code LoadShard} has completed yet, or the last one failed. Every
 * forward pass fails with the reason instead of returning output, so a node that
 * could not load its model can never look like a healthy shard.
 */
final class UnloadedShardHandler implements ForwardPassHandler {

	private final String reason;

	UnloadedShardHandler(String reason) {
		this.reason = reason;
	}

	@Override
	public ForwardResult forward(ForwardRequest request, ShardContext context) {
		throw new IllegalStateException(reason);
	}

	@Override
	public boolean isReady() {
		return false;
	}
}
