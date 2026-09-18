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
 * JFR event emitted once per residual-add call site (attention or FFN
 * skip-connection add back onto the running activation): scalar CPU Java
 * today, regardless of GPU layer offload.
 *
 * <p>One event covers a batch of {@link #windowSize} rows added in the same
 * call site — mirroring {@link AttentionEvent}'s per-call granularity rather
 * than instrumenting every individual row.
 */
@Name("juno.ResidualAdd")
@Label("Residual Add")
@Description("Scalar CPU residual skip-connection add over a batch of rows")
@Category({ "Juno", "Inference" })
@StackTrace(false)
public final class ResidualAddEvent extends Event {

    @Label("Window Size")
    @Description("Number of rows added in this call (>1 only for batched prefill/multi-decode)")
    public int windowSize;

    @Label("Start Position")
    @Description("Sequence position of the first row in this call")
    public int startPosition;

    @Label("Dimension")
    @Description("Vector width added per row (hidden dim)")
    public int dimension;
}
