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

/** Minimal JSON reader/writer for schema compilation (no extra deps). */
final class MiniJson {

	sealed interface Val {
	}

	record JObj(Map<String, Val> map) implements Val {
	}

	record JArr(List<Val> items) implements Val {
	}

	record JStr(String value) implements Val {
	}

	record JNum(String raw) implements Val {
	}

	record JBool(boolean value) implements Val {
	}

	record JNull() implements Val {
	}

	static Val parse(String s) {
		return new Parser(s).parseValue();
	}

	static String encode(Val v) {
		StringBuilder sb = new StringBuilder();
		write(v, sb);
		return sb.toString();
	}

	private static void write(Val v, StringBuilder sb) {
		switch (v) {
		case JNull ignored -> sb.append("null");
		case JBool b -> sb.append(b.value());
		case JNum n -> sb.append(n.raw());
		case JStr s -> quote(s.value(), sb);
		case JArr a -> {
			sb.append('[');
			for (int i = 0; i < a.items().size(); i++) {
				if (i > 0)
					sb.append(',');
				write(a.items().get(i), sb);
			}
			sb.append(']');
		}
		case JObj o -> {
			sb.append('{');
			int i = 0;
			for (Map.Entry<String, Val> e : o.map().entrySet()) {
				if (i++ > 0)
					sb.append(',');
				quote(e.getKey(), sb);
				sb.append(':');
				write(e.getValue(), sb);
			}
			sb.append('}');
		}
		}
	}

	private static void quote(String s, StringBuilder sb) {
		sb.append('"');
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			switch (c) {
			case '"' -> sb.append("\\\"");
			case '\\' -> sb.append("\\\\");
			case '\n' -> sb.append("\\n");
			case '\r' -> sb.append("\\r");
			case '\t' -> sb.append("\\t");
			default -> {
				if (c < 0x20)
					sb.append(String.format("\\u%04x", (int) c));
				else
					sb.append(c);
			}
			}
		}
		sb.append('"');
	}

	private static final class Parser {
		private final String s;
		private int i;
		private final int n;

		Parser(String s) {
			this.s = s;
			this.n = s.length();
		}

		Val parseValue() {
			Val v = value();
			skip();
			if (i != n)
				throw err("trailing JSON");
			return v;
		}

		private Val value() {
			skip();
			if (i >= n)
				throw err("unexpected end");
			char c = s.charAt(i);
			return switch (c) {
			case '{' -> object();
			case '[' -> array();
			case '"' -> new JStr(string());
			case 't' -> { expect("true"); yield new JBool(true); }
			case 'f' -> { expect("false"); yield new JBool(false); }
			case 'n' -> { expect("null"); yield new JNull(); }
			default -> number();
			};
		}

		private JObj object() {
			expect("{");
			LinkedHashMap<String, Val> map = new LinkedHashMap<>();
			skip();
			if (i < n && s.charAt(i) == '}') {
				i++;
				return new JObj(map);
			}
			while (true) {
				skip();
				String k = string();
				skip();
				expect(":");
				Val v = value();
				if (map.put(k, v) != null)
					throw err("duplicate key");
				skip();
				if (i < n && s.charAt(i) == ',') {
					i++;
					continue;
				}
				expect("}");
				return new JObj(map);
			}
		}

		private JArr array() {
			expect("[");
			List<Val> items = new ArrayList<>();
			skip();
			if (i < n && s.charAt(i) == ']') {
				i++;
				return new JArr(items);
			}
			while (true) {
				items.add(value());
				skip();
				if (i < n && s.charAt(i) == ',') {
					i++;
					continue;
				}
				expect("]");
				return new JArr(items);
			}
		}

		private String string() {
			expect("\"");
			StringBuilder sb = new StringBuilder();
			while (i < n) {
				char c = s.charAt(i++);
				if (c == '"')
					return sb.toString();
				if (c == '\\') {
					if (i >= n)
						throw err("dangling escape");
					char e = s.charAt(i++);
					sb.append(switch (e) {
					case '"', '\\', '/' -> e;
					case 'b' -> '\b';
					case 'f' -> '\f';
					case 'n' -> '\n';
					case 'r' -> '\r';
					case 't' -> '\t';
					case 'u' -> (char) hex(4);
					default -> throw err("bad escape");
					});
				} else
					sb.append(c);
			}
			throw err("unterminated string");
		}

		private JNum number() {
			int start = i;
			if (i < n && s.charAt(i) == '-')
				i++;
			if (i < n && s.charAt(i) == '0')
				i++;
			else if (i < n && s.charAt(i) >= '1' && s.charAt(i) <= '9') {
				while (i < n && Character.isDigit(s.charAt(i)))
					i++;
			} else
				throw err("bad number");
			if (i < n && s.charAt(i) == '.') {
				i++;
				if (i >= n || !Character.isDigit(s.charAt(i)))
					throw err("bad number");
				while (i < n && Character.isDigit(s.charAt(i)))
					i++;
			}
			if (i < n && (s.charAt(i) == 'e' || s.charAt(i) == 'E')) {
				i++;
				if (i < n && (s.charAt(i) == '+' || s.charAt(i) == '-'))
					i++;
				if (i >= n || !Character.isDigit(s.charAt(i)))
					throw err("bad number");
				while (i < n && Character.isDigit(s.charAt(i)))
					i++;
			}
			return new JNum(s.substring(start, i));
		}

		private int hex(int digits) {
			int v = 0;
			for (int k = 0; k < digits; k++) {
				if (i >= n)
					throw err("bad hex");
				int d = Character.digit(s.charAt(i++), 16);
				if (d < 0)
					throw err("bad hex");
				v = (v << 4) + d;
			}
			return v;
		}

		private void expect(String tok) {
			skip();
			if (!s.startsWith(tok, i))
				throw err("expected " + tok);
			i += tok.length();
		}

		private void skip() {
			while (i < n && Character.isWhitespace(s.charAt(i)))
				i++;
		}

		private IllegalArgumentException err(String msg) {
			return new IllegalArgumentException("JSON Schema: " + msg + " at " + i);
		}
	}
}
