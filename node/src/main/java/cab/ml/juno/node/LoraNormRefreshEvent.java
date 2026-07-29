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
 * JFR event emitted once per DoRA norm-cache refresh (full or partial batch).
 */
@Name("juno.LoraNormRefresh")
@Label("LoRA Norm Refresh")
@Description("DoRA row-norm / coefficient cache refresh")
@Category({ "Juno", "LoRA" })
@StackTrace(false)
public final class LoraNormRefreshEvent extends Event {

	@Label("Algorithm")
	public String algorithm = "";

	@Label("Scaling")
	public String scaling = "";

	@Label("Initialization")
	public String initialization = "";

	@Label("Architecture")
	public String architecture = "";

	@Label("Train Device")
	public String trainDevice = "";

	@Label("Rank")
	public int rank;

	@Label("Alpha")
	public float alpha;

	@Label("Effective Scale")
	public float effectiveScale;

	@Label("Targets")
	public String targets = "";

	@Label("Group Width")
	public int groupWidth;

	@Label("Layer Count")
	@Description("Distinct layers touched by this refresh")
	public int layerCount;

	@Label("Projection Count")
	@Description("Number of DoRA projections refreshed")
	public int projectionCount;

	@Label("Duration ms")
	public long durationMs;

	@Label("Bytes Touched")
	@Description("Optional estimate of bytes read/written; 0 when unknown")
	public long bytesTouched;

	@Label("Reason")
	@Description("load | post-step | reset | explicit")
	public String reason = "";
}
