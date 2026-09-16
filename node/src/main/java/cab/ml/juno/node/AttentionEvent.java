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

import jdk.jfr.Category;
import jdk.jfr.Description;
import jdk.jfr.Event;
import jdk.jfr.Label;
import jdk.jfr.Name;
import jdk.jfr.StackTrace;

/**
 * JFR event emitted once per grouped-query-attention call ({@code gqa}/{@code gqaInto}):
 * QK^T score computation, softmax, and the attention-weighted sum over V.
 *
 * <p>Attention runs as scalar CPU Java in every handler regardless of GPU layer
 * offload — {@code CudaMatVec}/{@code RocmMatVec} only accelerate the linear
 * projection GEMM/GEMV, not attention itself. This event exists to separate
 * attention cost from {@code juno.MatVec} in JFR so the {@code O(seq^2)} cost of
 * attending over a growing KV context (dominant at long prefill windows) is
 * directly visible instead of being folded into the surrounding
 * {@code juno.ForwardPass} / {@code juno.PrefillBatch} span.
 *
 * <p>One event covers a batch of {@link #windowSize} query positions in
 * {@link LlamaTransformerHandler#transformerLayerBatch} (one event per layer per
 * prefill window), or a single query position ({@link #windowSize} = 1) in
 * {@link LlamaTransformerHandler#transformerLayer} (decode).
 */
@Name("juno.Attention")
@Label("Attention")
@Description("QK^T + softmax + attention-weighted V sum for one gqa/gqaInto call")
@Category({ "Juno", "Inference" })
@StackTrace(false)
public final class AttentionEvent extends Event {

    @Label("Window Size")
    @Description("Number of query positions attended in this call (>1 only for batched prefill)")
    public int windowSize;

    @Label("Start Position")
    @Description("Sequence position of the first query token in this call")
    public int startPosition;

    @Label("Context Length")
    @Description("KV context length attended by the last query position in this call "
            + "(startPosition + windowSize) — the O(seq^2) independent variable")
    public int contextLength;
}
