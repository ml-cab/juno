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
 * Low-frequency session summary when LoRA adapters are loaded for inference
 * ({@code --lora-play}). Per-token timing remains {@code juno.ForwardPass}.
 */
@Name("juno.LoraPlayback")
@Label("LoRA Playback")
@Description("Sidecar adapter load / playback session summary")
@Category({ "Juno", "LoRA" })
@StackTrace(false)
public final class LoraPlaybackEvent extends Event {

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

	@Label("Adapter Count")
	public int adapterCount;

	@Label("Load ms")
	public long loadMs;

	@Label("Forward Count")
	@Description("Optional session forward count; 0 when not attributed")
	public int forwardCount;

	@Label("Tokens Generated")
	@Description("Optional session token count; 0 when not attributed")
	public int tokensGenerated;
}
