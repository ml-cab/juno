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
 * JFR event emitted once per token delivered to a client session.
 *
 * <p>Fired by {@link GenerationLoop} immediately after
 * {@link TokenConsumer#onToken} is called — that is, after sampling and
 * EOS checks pass and a real text piece is being streamed out. EOS tokens
 * that terminate generation without producing text are NOT recorded.
 *
 * <p>Because this event is coordinator-side, its timestamps are in the
 * coordinator JFR file alongside {@code juno.Tokenizer} and
 * {@code juno.TemplateFormat} events. {@link cab.ml.juno.metrics.JfrMetricsExtractor}
 * uses the span between the first and last event in a recording to compute
 * aggregate tokens-per-second (TPS) for the run.
 *
 * <h3>Reading in JDK Mission Control</h3>
 * <em>Event Browser → juno.TokenProduced</em>.
 * The event is instantaneous (zero duration). Sort by start time to see the
 * full token delivery timeline. Filter by {@code requestId} to isolate one
 * session. The inter-event gap is the per-token decode latency as observed
 * by the coordinator.
 *
 * <h3>TPS derivation</h3>
 * {@code TPS = count(juno.TokenProduced) / (last.startTime - first.startTime)}
 * This is aggregate throughput across all concurrent sessions in the recording
 * window — the number that matters for capacity planning.
 */
@Name("juno.TokenProduced")
@Label("Token Produced")
@Description("One token delivered to a client session after sampling and EOS checks.")
@Category({"Juno", "Inference"})
@StackTrace(false)
public final class TokenProducedEvent extends Event {

    @Label("Request ID")
    @Description("Request or session identifier — matches ForwardPassEvent.requestId on the node side.")
    public String requestId;

    @Label("Position")
    @Description("0-based position in the generated sequence for this session.")
    public int position;
}