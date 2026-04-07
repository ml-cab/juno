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

/**
 * Defines how the model is distributed across the inference cluster.
 *
 * PIPELINE — contiguous layer blocks, sequential activation flow (vertical /
 * depth scaling). Each node holds a distinct range of transformer layers
 * [startLayer, endLayer). The activation tensor passes node-0 → node-1 → ... →
 * node-N in strict serial order. Adding nodes increases the total VRAM budget,
 * enabling larger models. Network pattern: N-1 sequential gRPC hops per decode
 * step.
 *
 * TENSOR — column/row weight slices, same layers on every node (horizontal /
 * width scaling). Every node holds all transformer layers [0, totalLayers) but
 * only a slice of the weight matrices: attention heads [headStart, headEnd) and
 * a proportional FFN width slice. All nodes compute in parallel for each decode
 * step; the coordinator collects partial logit vectors and reduces them
 * (element-wise sum, star AllReduce) to produce the full next-token
 * distribution. Adding nodes reduces per-node VRAM pressure and increases
 * decode throughput. Network pattern: one broadcast + N parallel gRPC calls per
 * decode step.
 *
 */
public enum ParallelismType {

	/**
	 * Sequential layer sharding — vertical scaling. Each node computes a different
	 * depth slice of the transformer.
	 */
	PIPELINE,

	/**
	 * Column/row weight slicing — horizontal scaling. Every node computes the same
	 * layers but on a different tensor width slice.
	 */
	TENSOR
}