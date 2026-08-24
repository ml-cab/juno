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
 * JFR event emitted once per {@code juno merge} / {@link LoraMerge} completion.
 */
@Name("juno.LoraMerge")
@Label("LoRA Merge")
@Description("Adapter merge into GGUF (F32 preserve or projected requantization)")
@Category({ "Juno", "LoRA" })
@StackTrace(false)
public final class LoraMergeEvent extends Event {

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

	@Label("Merge Capability")
	@Description("CLI merge policy: f32-preserve | source-type-projected | sidecar-only | …")
	public String mergeCapability = "";

	@Label("Tensors Patched")
	public int tensorsPatched;

	@Label("Bytes Written")
	public long bytesWritten;

	@Label("Duration ms")
	public long durationMs;

	@Label("RMSE")
	@Description("Projected-merge reconstruction RMSE; 0 when not applicable")
	public float rmse;

	@Label("Max Abs Error")
	public float maxAbsError;

	@Label("Changed Blocks")
	public long changedBlocks;

	@Label("Total Blocks")
	public long totalBlocks;

	@Label("Saturation Rate")
	public float saturationRate;

	@Label("Delta Retention")
	@Description("Projected-merge delta retention; 0 when not applicable")
	public float deltaRetention;

	@Label("Success")
	public boolean success = true;

	@Label("Error")
	@Description("Short failure label; empty on success (no stack traces)")
	public String error = "";
}
