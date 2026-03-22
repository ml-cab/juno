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

import java.io.Serializable;

/**
 * Tensor-parallel shard assignment for a single node.
 *
 * In tensor-parallel mode every node holds the full layer range [0, totalLayers)
 * but computes only a horizontal slice of the weight matrices:
 *   attention heads  [headStart, headEnd)  where headStart = tensorRank * (numHeads / tensorWorldSize)
 *   FFN width slice  proportional to tensorRank / tensorWorldSize
 *
 * The coordinator broadcasts the input activation to all tensor-parallel nodes simultaneously.
 * Each node returns its partial logit vector. The coordinator reduces (element-wise sums)
 * the partial results to produce the full next-token distribution (star AllReduce).
 *
 * tensorRank    — this node's index in the tensor-parallel group, [0, tensorWorldSize)
 * tensorWorldSize — total number of nodes in the group
 *
 * hasEmbeddings and hasOutputProjection are always true for tensor-parallel nodes:
 * every node independently embeds the input tokens and produces a partial output projection.
 */
public record TensorShardAssignment(
        String  nodeId,
        String  host,
        int     grpcPort,
        int     startLayer,           // inclusive — always 0 in tensor-parallel mode
        int     endLayer,             // exclusive — always totalLayers in tensor-parallel mode
        boolean hasEmbeddings,        // always true — each node does its own embedding lookup
        boolean hasOutputProjection,  // always true — each node produces partial logits
        int     tensorRank,           // this node's rank in [0, tensorWorldSize)
        int     tensorWorldSize       // total nodes in the tensor-parallel group
) implements Serializable {

    public TensorShardAssignment {
        if (nodeId == null || nodeId.isBlank())
            throw new IllegalArgumentException("nodeId must not be blank");
        if (host == null || host.isBlank())
            throw new IllegalArgumentException("host must not be blank");
        if (grpcPort < 1 || grpcPort > 65535)
            throw new IllegalArgumentException("grpcPort out of range: " + grpcPort);
        if (startLayer < 0)
            throw new IllegalArgumentException("startLayer must be >= 0");
        if (endLayer <= startLayer)
            throw new IllegalArgumentException("endLayer must be > startLayer");
        if (tensorRank < 0)
            throw new IllegalArgumentException("tensorRank must be >= 0");
        if (tensorWorldSize < 1)
            throw new IllegalArgumentException("tensorWorldSize must be >= 1");
        if (tensorRank >= tensorWorldSize)
            throw new IllegalArgumentException(
                    "tensorRank (" + tensorRank + ") must be < tensorWorldSize (" + tensorWorldSize + ")");
    }

    /** Number of transformer layers this node owns (all layers in tensor-parallel mode). */
    public int layerCount() {
        return endLayer - startLayer;
    }

    /** gRPC target address for this node, usable as a ManagedChannel forAddress argument. */
    public String grpcTarget() {
        return host + ":" + grpcPort;
    }
}