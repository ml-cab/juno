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

/** JFR event for one context shift of a request's KV ({@link ContextWindow}). */
@Name("juno.ContextShift")
@Label("Context Shift")
@Description("Dropping the oldest tokens after the system prompt from a request's KV cache, and turning the kept keys"
		+ " back to their new positions, when the request reaches the context limit with context shifting on.")
@Category({ "Juno", "Inference" })
@StackTrace(false)
public final class ContextShiftEvent extends Event {

	@Label("Request ID")
	public String requestId;

	@Label("Positions Before")
	@Description("Positions written before the shift.")
	public int seqLen;

	@Label("Kept Prefix")
	@Description("Leading positions kept in place (the system prompt).")
	public int keep;

	@Label("Discarded")
	@Description("Positions dropped after the kept prefix; later positions move down by this many.")
	public int discard;
}
