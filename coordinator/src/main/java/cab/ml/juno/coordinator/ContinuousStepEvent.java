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
package cab.ml.juno.coordinator;

import jdk.jfr.Category;
import jdk.jfr.Description;
import jdk.jfr.Event;
import jdk.jfr.Label;
import jdk.jfr.Name;
import jdk.jfr.StackTrace;

/**
 * One continuous-scheduler engine step (shared decode {@code forwardBatch}
 * and optional mixed prefill ubatch chunks).
 */
@Name("juno.ContinuousStep")
@Label("Continuous Step")
@Description("One shared engine step of the continuous running set.")
@Category({ "Juno", "Inference" })
@StackTrace(false)
public final class ContinuousStepEvent extends Event {

	@Label("Running Set")
	@Description("Requests in the running set at this step.")
	public int runningSetSize;

	@Label("Decode Batch")
	@Description("Requests included in this forwardBatch call.")
	public int decodeBatchSize;

	@Label("Prefill Chunks")
	@Description("Prefill ubatch chunks executed in this step.")
	public int prefillChunks;

	@Label("Prefill Tokens")
	@Description("Prompt tokens evaluated across prefill chunks this step.")
	public int prefillTokens;

	@Label("Admitted")
	@Description("Requests admitted since the previous step.")
	public int admitted;

	@Label("Retired")
	@Description("Requests that finished in this step.")
	public int retired;
}
