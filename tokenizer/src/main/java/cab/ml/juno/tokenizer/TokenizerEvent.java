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
package cab.ml.juno.tokenizer;

import jdk.jfr.Category;
import jdk.jfr.Description;
import jdk.jfr.Event;
import jdk.jfr.Label;
import jdk.jfr.Name;
import jdk.jfr.StackTrace;

/**
 * JFR event emitted for each {@link Tokenizer} operation.
 *
 * <p>Fired by every concrete {@link Tokenizer} implementation:
 * <ul>
 *   <li>{@link GgufTokenizer}   — {@code tokenizerType = "gguf"}
 *   <li>{@link SimpleTokenizer} — {@code tokenizerType = "simple"}
 * </ul>
 *
 * <p>Three operations are tracked:
 * <ul>
 *   <li>{@code encode}      — full prompt text → int[] token IDs (once per request)
 *   <li>{@code decode}      — int[] token IDs → full text (batch decode)
 *   <li>{@code decodeToken} — single token ID → text piece (once per generated token)
 * </ul>
 *
 * <h3>Reading in JDK Mission Control</h3>
 * <em>Event Browser → juno.Tokenizer</em>.
 * Filter {@code operation = "encode"} to profile prompt-encoding latency.
 * Filter {@code operation = "decodeToken"} and aggregate to measure total
 * streaming decode overhead across an entire generation.
 */
@Name("juno.Tokenizer")
@Label("Tokenizer")
@Description("One Tokenizer encode / decode / decodeToken call")
@Category({ "Juno", "Tokenizer" })
@StackTrace(false)
public final class TokenizerEvent extends Event {

    @Label("Tokenizer Type")
    @Description("Implementation: gguf | djl | stub | simple")
    public String tokenizerType;

    @Label("Operation")
    @Description("encode | decode | decodeToken")
    public String operation;

    @Label("Input Length")
    @Description("For encode: input text character count. "
            + "For decode: number of token IDs. "
            + "For decodeToken: always 1.")
    public int inputLength;

    @Label("Output Token Count")
    @Description("For encode: number of token IDs produced. "
            + "For decode/decodeToken: number of output characters.")
    public int outputLength;
}