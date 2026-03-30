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
 * JFR event emitted once per {@link ForwardPassHandler#forward} call.
 *
 * <p>Fired by every concrete {@link ForwardPassHandler} implementation:
 * <ul>
 *   <li>{@link LlamaTransformerHandler} — {@code handlerType = "llama"}
 *   <li>{@link Phi3TransformerHandler}  — {@code handlerType = "phi3"}
 *   <li>(removed — merged into {@link LlamaTransformerHandler} with {@link CpuMatVec})
 *   <li>(removed — merged into {@link LlamaTransformerHandler} with {@link CudaMatVec})
 *   <li>{@link CyclicForwardPassHandler}— {@code handlerType = "cyclic"}
 *   <li>{@link LoraTrainableHandler}    — {@code handlerType = "lora"}
 * </ul>
 *
 * <h3>Reading in JDK Mission Control</h3>
 * <em>Event Browser → juno.ForwardPass</em>.
 * The {@code startPosition} field reveals prefill vs. decode behaviour:
 * position 0 is the first prefill token; larger positions are decode steps.
 * {@code layerCount} shows per-node shard depth in distributed deployments.
 *
 * <h3>Flame graph tip</h3>
 * Correlate with {@code juno.MatVec} events at the same timestamp to
 * decompose forward-pass time into attention projections, FFN, and
 * output projection.
 */
@Name("juno.ForwardPass")
@Label("Forward Pass")
@Description("One ForwardPassHandler.forward() call — covers embedding lookup, "
        + "all assigned transformer layers, and (for the last node) the output projection")
@Category({ "Juno", "Inference" })
@StackTrace(false)
public final class ForwardPassEvent extends Event {

    @Label("Handler Type")
    @Description("Implementation: llama | phi3 | cpu | gpu | cyclic | lora")
    public String handlerType;

    @Label("Request ID")
    @Description("Request or session identifier from ForwardRequest.requestId()")
    public String requestId;

    @Label("Start Position")
    @Description("Sequence position processed in this call (0 = first prefill token)")
    public int startPosition;

    @Label("Layer Count")
    @Description("Number of transformer layers executed by this shard")
    public int layerCount;

    @Label("Has Output Projection")
    @Description("True when this is the last node in the pipeline (logits returned)")
    public boolean hasOutputProjection;
}