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
 * JFR event for one {@link InferencePipeline#prefillBatch} window.
 *
 * <p>Emitted by {@link LocalInferencePipeline#prefillBatch} so batched prefill
 * shows up in {@code juno.ForwardPass.prefill.*} metrics (unlike scalar
 * {@link ForwardPassEvent} which only covers {@code forward()}).
 */
@Name("juno.PrefillBatch")
@Label("Prefill Batch")
@Description("One prefillBatch() window — chunked prompt evaluation for KV warm-up")
@Category({ "Juno", "Inference" })
@StackTrace(false)
public final class PrefillBatchForwardEvent extends Event {

	@Label("Request ID")
	public String requestId;

	@Label("Window Size")
	@Description("Number of prompt tokens in this prefill window")
	public int windowSize;

	@Label("Start Position")
	@Description("Sequence position of the first token in the window")
	public int startPosition;
}
