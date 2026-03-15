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
 * Priority level for inference requests. Weight determines effective ordering
 * in PriorityBlockingQueue. Higher weight = scheduled sooner.
 */
public enum RequestPriority {

	HIGH(3), NORMAL(1), LOW(0);

	private final int weight;

	RequestPriority(int weight) {
		this.weight = weight;
	}

	public int weight() {
		return weight;
	}
}
