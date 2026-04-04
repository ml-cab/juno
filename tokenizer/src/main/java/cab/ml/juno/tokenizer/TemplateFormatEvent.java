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
 * JFR event emitted once per {@link ChatTemplateFormatter#format} call.
 *
 * <p>Captures the cost and shape of applying a model-specific chat template
 * (LLaMA-3, Mistral, Gemma, Phi-3, Zephyr/TinyLlama, ChatML) to a list of
 * messages before tokenisation.
 *
 * <p>In multi-turn sessions {@link #messageCount} grows with each turn,
 * and {@link #outputLength} grows proportionally. This event makes the
 * template-overhead contribution to overall request latency visible.
 *
 * <h3>Reading in JDK Mission Control</h3>
 * <em>Event Browser → juno.TemplateFormat</em>.
 * Group by {@code modelType} to compare template overhead across model families.
 */
@Name("juno.TemplateFormat")
@Label("Template Format")
@Description("One ChatTemplateFormatter.format() call — applies a model-specific chat template "
        + "to a list of messages and produces the formatted prompt string")
@Category({ "Juno", "Tokenizer" })
@StackTrace(false)
public final class TemplateFormatEvent extends Event {

    @Label("Model Type")
    @Description("Chat template family: llama3 | mistral | gemma | phi3 | tinyllama | chatml | …")
    public String modelType;

    @Label("Message Count")
    @Description("Number of ChatMessage entries in the conversation at format time")
    public int messageCount;

    @Label("Output Length")
    @Description("Character count of the formatted prompt string returned by format()")
    public int outputLength;
}