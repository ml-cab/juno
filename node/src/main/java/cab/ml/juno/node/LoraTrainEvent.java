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
 * JFR event emitted once per {@link LoraTrainableHandler#trainStep} call.
 *
 * <p>
 * Captures a timing breakdown of the three phases inside each gradient step so
 * that a profiler recording can immediately show where time is being spent:
 *
 * <pre>
 *   Forward pass  – embedding lookup, layer forward, output logits, softmax
 *   Backward pass – gradient through output proj, RMSNorm, attention, FFN
 *   Optimizer     – Adam moment update and parameter write-back
 * </pre>
 *
 * <h3>Reading in JDK Mission Control</h3> Open the .jfr file, go to <em>Event
 * Browser</em>, search for {@code juno.LoraTrainStep}. The built-in flame graph
 * will show the Java call stack for each event's duration. The custom fields
 * (loss, forward_ms, etc.) appear in the <em>Event Details</em> panel.
 *
 * <h3>Sampling with jcmd</h3>
 * 
 * <pre>
 *   jcmd &lt;pid&gt; JFR.dump filename=lora.jfr
 * </pre>
 *
 * @see LoraTrainableHandler#trainStep
 */
@Name("juno.LoraTrainStep")
@Label("LoRA Train Step")
@Description("One gradient step in LoRA fine-tuning: forward pass, backward pass, Adam update")
@Category({ "Juno", "LoRA" })
@StackTrace(false) // stack trace is not useful here — the event spans the full step
public final class LoraTrainEvent extends Event {

	@Label("Step")
	@Description("Gradient step index (1-based) within the current /train invocation")
	public int step;

	@Label("Num Tokens")
	@Description("Number of input tokens in this chunk (sequence length)")
	public int numTokens;

	@Label("Loss")
	@Description("Mean cross-entropy loss (nats) for this gradient step")
	public float loss;

	@Label("Forward ms")
	@Description("Wall time for the forward pass across all positions (ms)")
	public long forwardMs;

	@Label("Backward ms")
	@Description("Wall time for the backward pass across all positions (ms)")
	public long backwardMs;

	@Label("Optimizer ms")
	@Description("Wall time for the Adam parameter update (ms)")
	public long optimizerMs;

	@Label("Total ms")
	@Description("Total wall time for this gradient step (ms)")
	public long totalMs;
}