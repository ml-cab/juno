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

/** JFR event for encoding one request's formatted prompt into token ids ({@link PromptEncoder}). */
@Name("juno.PromptEncode")
@Label("Prompt Encode")
@Description("Encoding a request's formatted prompt into token ids, before its first forward pass.")
@Category({ "Juno", "Inference" })
@StackTrace(false)
public final class PromptEncodeEvent extends Event {

	@Label("Request ID")
	public String requestId;

	@Label("Characters")
	@Description("Length of the formatted prompt.")
	public int characters;

	@Label("Tokens")
	@Description("Number of token ids the prompt encoded to.")
	public int tokens;
}
