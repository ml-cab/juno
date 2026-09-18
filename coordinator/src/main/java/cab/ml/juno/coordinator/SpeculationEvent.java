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
 * JFR event emitted once per ngram-simple draft/verify round in
 * {@link GenerationLoop#generate}. Only fired when {@link NgramDraftCache#propose}
 * actually returned a non-empty draft — a step that fell back to plain one-token
 * decoding (no draft available) does not emit this event.
 */
@Name("juno.Speculation")
@Label("Speculation")
@Description("One ngram-simple draft/verify round: tokens proposed vs. accepted before the target model diverged.")
@Category({ "Juno", "Inference" })
@StackTrace(false)
public final class SpeculationEvent extends Event {

	@Label("Request ID")
	@Description("Request or session identifier.")
	public String requestId;

	@Label("Draft Tokens")
	@Description("Number of tokens proposed by NgramDraftCache this round.")
	public int draftTokens;

	@Label("Accepted Tokens")
	@Description("Number of proposed tokens the target model's own sampling step confirmed before the first "
			+ "divergence (or all of them, if none diverged).")
	public int acceptedTokens;
}
