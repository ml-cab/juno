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
 * JFR event emitted once per SwiGLU call site ({@code silu(gate) * up} over
 * the intermediate dimension): scalar CPU Java today, regardless of GPU layer
 * offload.
 *
 * <p>One event covers a batch of {@link #windowSize} rows in the same call
 * site — mirroring {@link AttentionEvent}'s per-call granularity rather than
 * instrumenting every individual row.
 */
@Name("juno.SwiGlu")
@Label("SwiGLU")
@Description("Scalar CPU silu(gate) * up call over a batch of rows")
@Category({ "Juno", "Inference" })
@StackTrace(false)
public final class SwiGluEvent extends Event {

    @Label("Window Size")
    @Description("Number of rows processed in this call (>1 only for batched prefill/multi-decode)")
    public int windowSize;

    @Label("Start Position")
    @Description("Sequence position of the first row in this call")
    public int startPosition;

    @Label("Dimension")
    @Description("Vector width processed per row (intermediate size)")
    public int dimension;
}
