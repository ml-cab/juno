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
 * JFR event emitted once per held-out validation evaluation.
 */
@Name("juno.LoraValidation")
@Label("LoRA Validation")
@Description("Held-out validation loss for LoRA fine-tuning")
@Category({ "Juno", "LoRA" })
@StackTrace(false)
public final class LoraValidationEvent extends Event {

	@Label("Loss")
	@Description("Token-weighted mean validation loss (nats)")
	public float loss;

	@Label("Prediction Count")
	@Description("Number of validation prediction tokens")
	public int predictionCount;

	@Label("Duration ms")
	@Description("Wall time of the validation evaluation (ms)")
	public long durationMs;

	@Label("Best So Far")
	@Description("Whether this validation loss improved the best-so-far")
	public boolean bestSoFar;

	@Label("Pass Index")
	@Description("1-based training pass index when known; 0 otherwise")
	public int passIndex;

	@Label("Optimizer Step")
	@Description("Optimizer update count at evaluation time")
	public int optimizerStep;

	@Label("Train Loss At Eval")
	@Description("Most recent train loss at evaluation; NaN when unknown")
	public float trainLossAtEval = Float.NaN;

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
}
