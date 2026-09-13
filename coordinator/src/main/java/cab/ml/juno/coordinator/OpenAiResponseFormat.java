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

import com.fasterxml.jackson.databind.JsonNode;

import cab.ml.juno.sampler.GbnfGrammar;
import cab.ml.juno.sampler.JsonSchemaToGbnf;

/**
 * Compiles OpenAI {@code response_format} and optional {@code x_juno_grammar}
 * into a GBNF grammar. Unsupported schema keywords fail closed.
 */
public final class OpenAiResponseFormat {

	private OpenAiResponseFormat() {
	}

	/**
	 * @return error message, or {@code null} when the shape is allowed
	 */
	public static String validate(JsonNode responseFormat) {
		if (responseFormat == null || responseFormat.isNull() || responseFormat.isMissingNode())
			return null;
		if (!responseFormat.isObject())
			return "response_format must be an object";
		JsonNode type = responseFormat.get("type");
		if (type == null || !type.isTextual())
			return "response_format.type is required";
		String t = type.asText();
		if ("text".equals(t) || "json_object".equals(t) || "json_schema".equals(t))
			return null;
		return "response_format type '" + t + "' is not supported (text, json_object, json_schema)";
	}

	/**
	 * @return compiled grammar, or {@code null} when unconstrained
	 * @throws IllegalArgumentException when the combination or schema is illegal
	 */
	public static GbnfGrammar compile(JsonNode responseFormat, String xJunoGrammar) {
		boolean hasGrammar = xJunoGrammar != null && !xJunoGrammar.isBlank();
		String type = typeOf(responseFormat);
		if (hasGrammar && type != null && !"text".equals(type))
			throw new IllegalArgumentException(
					"x_juno_grammar cannot be combined with response_format.type=" + type);
		if (hasGrammar)
			return GbnfGrammar.parse(xJunoGrammar);
		if (type == null || "text".equals(type))
			return null;
		if ("json_object".equals(type))
			return JsonSchemaToGbnf.jsonObjectGrammar();
		if ("json_schema".equals(type))
			return JsonSchemaToGbnf.compileGrammar(schemaJson(responseFormat));
		throw new IllegalArgumentException("response_format type '" + type + "' is not supported");
	}

	/**
	 * Like {@link #compile(JsonNode, String)}, then a process-wide CLI grammar
	 * when the request does not set {@code response_format.type=text} or a
	 * request grammar.
	 */
	public static GbnfGrammar compile(JsonNode responseFormat, String xJunoGrammar, GbnfGrammar fallback) {
		GbnfGrammar compiled = compile(responseFormat, xJunoGrammar);
		if (compiled != null)
			return compiled;
		if ("text".equals(typeOf(responseFormat)))
			return null;
		return fallback;
	}

	private static String typeOf(JsonNode responseFormat) {
		if (responseFormat == null || responseFormat.isNull() || responseFormat.isMissingNode())
			return null;
		if (!responseFormat.isObject())
			throw new IllegalArgumentException("response_format must be an object");
		JsonNode type = responseFormat.get("type");
		if (type == null || !type.isTextual())
			throw new IllegalArgumentException("response_format.type is required");
		return type.asText();
	}

	private static String schemaJson(JsonNode responseFormat) {
		JsonNode wrapper = responseFormat.get("json_schema");
		if (wrapper == null || wrapper.isNull() || wrapper.isMissingNode())
			throw new IllegalArgumentException("response_format.json_schema is required when type=json_schema");
		JsonNode schema = wrapper.has("schema") ? wrapper.get("schema") : wrapper;
		if (schema == null || !schema.isObject())
			throw new IllegalArgumentException("response_format.json_schema.schema must be a JSON object");
		return schema.toString();
	}
}
