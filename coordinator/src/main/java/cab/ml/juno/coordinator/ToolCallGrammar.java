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

import java.util.ArrayList;
import java.util.List;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import cab.ml.juno.sampler.GbnfGrammar;
import cab.ml.juno.sampler.JsonSchemaToGbnf;

/**
 * Compiles a GBNF envelope {@code <tool_call> {name, arguments} </tool_call>}
 * from tool parameter schemas (JSON Schema subset).
 */
public final class ToolCallGrammar {

	private static final ObjectMapper JSON = new ObjectMapper();

	private ToolCallGrammar() {
	}

	public static GbnfGrammar compile(List<OpenAiTools.FunctionTool> tools, String forcedFunction) {
		List<OpenAiTools.FunctionTool> selected = select(tools, forcedFunction);
		if (selected.isEmpty())
			throw new IllegalArgumentException("tools must not be empty when compiling a tool-call grammar");
		ObjectNode envelope = JSON.createObjectNode();
		envelope.put("type", "object");
		ObjectNode props = envelope.putObject("properties");
		ObjectNode name = props.putObject("name");
		ArrayNode names = name.putArray("enum");
		for (OpenAiTools.FunctionTool t : selected)
			names.add(t.name());
		if (selected.size() == 1 && selected.get(0).parameters() != null && selected.get(0).parameters().isObject())
			props.set("arguments", selected.get(0).parameters());
		else
			props.putObject("arguments").put("type", "object");
		ArrayNode required = envelope.putArray("required");
		required.add("name");
		required.add("arguments");
		String inner = JsonSchemaToGbnf.compile(envelope.toString());
		String renamed = inner.replace("root ::=", "tooljson ::=");
		return GbnfGrammar.parse(renamed + "root ::= \"<tool_call>\" ws tooljson ws \"</tool_call>\" ws\n");
	}

	private static List<OpenAiTools.FunctionTool> select(List<OpenAiTools.FunctionTool> tools, String forcedFunction) {
		if (tools == null)
			return List.of();
		if (forcedFunction == null || forcedFunction.isBlank())
			return List.copyOf(tools);
		List<OpenAiTools.FunctionTool> out = new ArrayList<>();
		for (OpenAiTools.FunctionTool t : tools) {
			if (forcedFunction.equals(t.name()))
				out.add(t);
		}
		if (out.isEmpty())
			throw new IllegalArgumentException("tool_choice function '" + forcedFunction + "' is not in tools");
		return out;
	}
}
