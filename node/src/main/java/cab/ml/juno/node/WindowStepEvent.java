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
 * JFR event for the parts of a forward window that no per-op event covers, so that a
 * prefill window's time can be broken down with nothing left unnamed: the token embedding
 * lookup, each projection call (the matmul dispatch, the backend's own {@code juno.MatVec}
 * span inside it, and the copy of its result into the layer workspace), bias adds, the
 * per-token KV write (host cache plus any device mirror), and the final norm with the LM
 * head.
 *
 * <p>One event per call site per layer, the same granularity as {@link SwiGluEvent}.
 */
@Name("juno.WindowStep")
@Label("Window Step")
@Description("Forward-window work outside the per-op events: embed, projection, bias_add, kv_write, lm_head")
@Category({ "Juno", "Inference" })
@StackTrace(false)
public final class WindowStepEvent extends Event {

    public static final String EMBED = "embed";
    public static final String PROJECTION = "projection";
    public static final String BIAS_ADD = "bias_add";
    public static final String KV_WRITE = "kv_write";
    public static final String LM_HEAD = "lm_head";

    @Label("Step")
    @Description("embed, projection, bias_add, kv_write or lm_head")
    public String step;

    @Label("Window Size")
    @Description("Number of rows in the forward window (>1 for batched prefill/multi-decode)")
    public int windowSize;

    @Label("Start Position")
    @Description("Sequence position of the first row in the window")
    public int startPosition;

    /** Starts timing a step. */
    static WindowStepEvent start() {
        WindowStepEvent ev = new WindowStepEvent();
        ev.begin();
        return ev;
    }

    /** Ends the step and commits it. */
    void end(String step, int windowSize, int startPosition) {
        this.step = step;
        this.windowSize = windowSize;
        this.startPosition = startPosition;
        commit();
    }
}
