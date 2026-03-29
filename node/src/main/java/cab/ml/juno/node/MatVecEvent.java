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
 * JFR event emitted once per matrix-vector multiply call.
 *
 * <p>Fired by {@link CpuMatVec#sgemv} and both overloads of
 * {@link CudaMatVec#sgemv} — the two concrete {@link MatVec} implementations.
 * The event duration covers the full sgemv computation including, for CUDA,
 * the host↔device transfers.
 *
 * <h3>Reading in JDK Mission Control</h3>
 * <em>Event Browser → juno.MatVec</em>.
 * Sort by duration descending to find the most expensive multiply shapes
 * (typically the output projection: 32 000 × 2 048 for TinyLlama).
 * Filter {@code backend = "cuda-resident"} to compare host-path vs
 * weight-resident GPU performance.
 *
 * <h3>Typical call rates</h3>
 * TinyLlama-1.1B (22 layers, 7 matVec/layer + output projection):
 * ~155 events per generated token. Use a JFR threshold ≥ 1 ms to suppress
 * small auxiliary multiplies when profiling at high token throughput.
 */
@Name("juno.MatVec")
@Label("MatVec")
@Description("One matrix-vector multiply call: y[rows] = A[rows,cols] × x[cols]")
@Category({ "Juno", "MatVec" })
@StackTrace(false)
public final class MatVecEvent extends Event {

    @Label("Backend")
    @Description("Compute backend: \"cpu\" (parallel IntStream), \"cuda\" (cublasSgemv host A), "
            + "\"cuda-resident\" (cublasSgemv device A)")
    public String backend;

    @Label("Rows")
    @Description("Number of output elements (output dimension of A)")
    public int rows;

    @Label("Cols")
    @Description("Number of input elements (inner dimension of A, length of x)")
    public int cols;
}