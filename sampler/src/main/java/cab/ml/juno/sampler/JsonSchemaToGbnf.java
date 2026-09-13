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
package cab.ml.juno.sampler;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Compiles a documented JSON Schema subset to GBNF. Unsupported keywords fail
 * closed. Object properties are emitted in schema key order (key permutations
 * are not generated).
 */
public final class JsonSchemaToGbnf {

	private static final Set<String> ALLOWED = Set.of("$schema", "$id", "title", "description", "default",
			"additionalProperties", "type", "properties", "required", "items", "enum", "const");
	private static final Set<String> REJECTED = Set.of("$ref", "oneOf", "anyOf", "allOf", "not", "if", "then", "else",
			"pattern", "format", "patternProperties", "unevaluatedProperties", "dependentRequired", "prefixItems",
			"$dynamicRef", "$defs", "definitions", "unevaluatedItems", "minItems", "maxItems", "minLength", "maxLength",
			"minimum", "maximum");

	private static final String JSON_OBJECT_GBNF = prelude()
			+ "root ::= object ws\n"
			+ "value ::= object | array | string | number | boolean | null\n"
			+ "object ::= \"{\" ws ( string \":\" ws value ( \",\" ws string \":\" ws value )* )? \"}\"\n"
			+ "array ::= \"[\" ws ( value ( \",\" ws value )* )? \"]\"\n";

	static String prelude() {
		return """
				ws ::= ( " " | "\\t" | "\\n" | "\\r" )*
				string ::= dq chars dq
				dq ::= "\\""
				chars ::= char*
				char ::= [a-zA-Z0-9] | extra | "\\\\" escape
				extra ::= " " | "_" | "." | "," | ":" | ";" | "!" | "?" | "'" | "(" | ")" | "+" | "/" | "-"
				escape ::= dq | "\\\\" | "/" | "b" | "f" | "n" | "r" | "t" | "u" hex hex hex hex
				hex ::= [0-9a-fA-F]
				integer ::= "-"? ("0" | [1-9] [0-9]*)
				number ::= integer frac? exp?
				frac ::= "." [0-9]+
				exp ::= [eE] ("+" | "-")? [0-9]+
				boolean ::= "true" | "false"
				null ::= "null"
				""";
	}

	private final StringBuilder out = new StringBuilder();
	private int anon;
	private boolean freeValueEmitted;

	private JsonSchemaToGbnf() {
	}

	public static String compile(String schemaJson) {
		if (schemaJson == null || schemaJson.isBlank())
			throw new IllegalArgumentException("JSON Schema is empty");
		MiniJson.Val root = MiniJson.parse(schemaJson);
		JsonSchemaToGbnf c = new JsonSchemaToGbnf();
		c.emitPrelude();
		String start = c.emitSchema(root, "rootval");
		c.out.append("root ::= ").append(start).append(" ws\n");
		return c.out.toString();
	}

	public static GbnfGrammar compileGrammar(String schemaJson) {
		return GbnfGrammar.parse(compile(schemaJson));
	}

	/** Any JSON object (OpenAI {@code response_format.type=json_object}). */
	public static GbnfGrammar jsonObjectGrammar() {
		return GbnfGrammar.parse(JSON_OBJECT_GBNF);
	}

	private void emitPrelude() {
		out.append(prelude());
	}

	private String emitSchema(MiniJson.Val node, String hint) {
		if (!(node instanceof MiniJson.JObj obj))
			throw new IllegalArgumentException("JSON Schema must be an object");
		rejectUnsupported(obj.map(), true);
		if (obj.map().containsKey("enum"))
			return emitEnum(obj.map().get("enum"), hint);
		if (obj.map().containsKey("const"))
			return emitConst(obj.map().get("const"), hint);
		String type = typeOf(obj);
		return switch (type) {
		case "object" -> emitObject(obj, hint);
		case "array" -> emitArray(obj, hint);
		case "string" -> "string";
		case "integer" -> "integer";
		case "number" -> "number";
		case "boolean" -> "boolean";
		case "null" -> "null";
		default -> throw new IllegalArgumentException("unsupported JSON Schema type '" + type + "'");
		};
	}

	private String freeValue() {
		if (!freeValueEmitted) {
			out.append("value ::= obj | arr | string | number | boolean | null\n");
			out.append("obj ::= \"{\" ws ( string \":\" ws value ( \",\" ws string \":\" ws value )* )? \"}\"\n");
			out.append("arr ::= \"[\" ws ( value ( \",\" ws value )* )? \"]\"\n");
			freeValueEmitted = true;
		}
		return "value";
	}

	private String emitObject(MiniJson.JObj obj, String hint) {
		MiniJson.Val propsVal = obj.map().get("properties");
		LinkedHashMap<String, MiniJson.Val> props = new LinkedHashMap<>();
		if (propsVal instanceof MiniJson.JObj p)
			props.putAll(p.map());
		else if (propsVal != null)
			throw new IllegalArgumentException("properties must be an object");
		MiniJson.Val add = obj.map().get("additionalProperties");
		if (add instanceof MiniJson.JBool b && b.value())
			throw new IllegalArgumentException("additionalProperties: true is not supported");
		if (add != null && !(add instanceof MiniJson.JBool) && !(add instanceof MiniJson.JNull))
			throw new IllegalArgumentException("additionalProperties schemas are not supported");
		if (props.isEmpty())
			return freeObject(hint);
		List<String> required = requiredNames(obj.map().get("required"), props);
		List<Prop> ordered = new ArrayList<>();
		for (Map.Entry<String, MiniJson.Val> e : props.entrySet()) {
			String rule = emitSchema(e.getValue(), hint + "-" + sanitize(e.getKey()));
			ordered.add(new Prop(e.getKey(), rule, required.contains(e.getKey())));
		}
		String name = uniq(hint);
		out.append(name).append(" ::= \"{\" ws ").append(objectBody(ordered, 0, false)).append(" \"}\"\n");
		return name;
	}

	private String freeObject(String hint) {
		freeValue();
		String name = uniq(hint);
		out.append(name).append(" ::= obj\n");
		return name;
	}

	private String objectBody(List<Prop> props, int i, boolean needComma) {
		if (i >= props.size())
			return "";
		Prop p = props.get(i);
		String kv = "\"" + escapeGbnf(jsonString(p.name)) + "\" ws \":\" ws " + p.rule;
		String with = (needComma ? "\",\" ws " : "") + kv;
		String restOn = objectBody(props, i + 1, true);
		if (!restOn.isEmpty())
			with = with + " " + restOn;
		if (p.required)
			return with;
		String restOff = objectBody(props, i + 1, needComma);
		if (restOff.isEmpty())
			return "( " + with + " )?";
		return "( " + with + " | " + restOff + " )";
	}

	private String emitArray(MiniJson.JObj obj, String hint) {
		MiniJson.Val items = obj.map().get("items");
		String itemRule = items == null ? freeValue() : emitSchema(items, hint + "-item");
		String name = uniq(hint);
		out.append(name).append(" ::= \"[\" ws ( ").append(itemRule).append(" ( \",\" ws ").append(itemRule)
				.append(" )* )? \"]\"\n");
		return name;
	}

	private String emitEnum(MiniJson.Val node, String hint) {
		if (!(node instanceof MiniJson.JArr arr) || arr.items().isEmpty())
			throw new IllegalArgumentException("enum must be a non-empty array");
		String name = uniq(hint);
		out.append(name).append(" ::=");
		for (int i = 0; i < arr.items().size(); i++) {
			if (i > 0)
				out.append(" |");
			out.append(" \"").append(escapeGbnf(MiniJson.encode(arr.items().get(i)))).append('"');
		}
		out.append('\n');
		return name;
	}

	private String emitConst(MiniJson.Val node, String hint) {
		String name = uniq(hint);
		out.append(name).append(" ::= \"").append(escapeGbnf(MiniJson.encode(node))).append("\"\n");
		return name;
	}

	private static String typeOf(MiniJson.JObj obj) {
		MiniJson.Val t = obj.map().get("type");
		if (t == null) {
			if (obj.map().containsKey("properties"))
				return "object";
			if (obj.map().containsKey("items"))
				return "array";
			throw new IllegalArgumentException("JSON Schema is missing type");
		}
		if (t instanceof MiniJson.JStr s)
			return s.value();
		throw new IllegalArgumentException("JSON Schema type unions are not supported");
	}

	private static List<String> requiredNames(MiniJson.Val required, Map<String, MiniJson.Val> props) {
		if (required == null)
			return List.of();
		if (!(required instanceof MiniJson.JArr arr))
			throw new IllegalArgumentException("required must be an array of strings");
		List<String> names = new ArrayList<>();
		for (MiniJson.Val v : arr.items()) {
			if (!(v instanceof MiniJson.JStr s))
				throw new IllegalArgumentException("required entries must be strings");
			if (!props.containsKey(s.value()))
				throw new IllegalArgumentException("required key '" + s.value() + "' is not in properties");
			names.add(s.value());
		}
		return names;
	}

	private static void rejectUnsupported(Map<String, MiniJson.Val> map, boolean schemaObject) {
		if (schemaObject) {
			for (String k : map.keySet()) {
				if (REJECTED.contains(k))
					throw new IllegalArgumentException("unsupported JSON Schema keyword '" + k + "'");
				if (!ALLOWED.contains(k))
					throw new IllegalArgumentException("unsupported JSON Schema keyword '" + k + "'");
			}
		}
		MiniJson.Val props = map.get("properties");
		if (props instanceof MiniJson.JObj p) {
			for (MiniJson.Val v : p.map().values())
				if (v instanceof MiniJson.JObj o)
					rejectUnsupported(o.map(), true);
		}
		MiniJson.Val items = map.get("items");
		if (items instanceof MiniJson.JObj o)
			rejectUnsupported(o.map(), true);
	}

	private String uniq(String hint) {
		return sanitize(hint) + "_" + (anon++);
	}

	private static String sanitize(String raw) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < raw.length(); i++) {
			char c = raw.charAt(i);
			if (Character.isLetterOrDigit(c) || c == '-')
				sb.append(c);
			else
				sb.append('_');
		}
		if (sb.isEmpty() || !Character.isLetter(sb.charAt(0)))
			sb.insert(0, 'r');
		return sb.toString();
	}

	private static String jsonString(String s) {
		return MiniJson.encode(new MiniJson.JStr(s));
	}

	private static String escapeGbnf(String json) {
		return json.replace("\\", "\\\\").replace("\"", "\\\"");
	}

	private record Prop(String name, String rule, boolean required) {
	}
}
