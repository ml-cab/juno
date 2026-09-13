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
 * Fired once when a request opens a GBNF constrained-decoding session.
 */
@Name("juno.GrammarConstrained")
@Label("Grammar Constrained")
@Description("A generation request attached a GBNF grammar and will mask illegal tokens.")
@Category({ "Juno", "Inference" })
@StackTrace(false)
public final class GrammarConstrainedEvent extends Event {

	@Label("Request ID")
	@Description("Request or session identifier.")
	public String requestId;
}
