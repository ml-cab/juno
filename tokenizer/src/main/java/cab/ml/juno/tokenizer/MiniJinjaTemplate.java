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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A restricted Jinja-subset template engine sufficient to render real-world
 * GGUF-embedded {@code tokenizer.chat_template} strings (Llama-3, Phi-3,
 * ChatML-family templates and similar).
 *
 * <p>Supported syntax:
 * <ul>
 * <li>{@code {{ expr }}} output, with {@code {{-}}/{-}}} explicit whitespace
 * trim; block tags ({@code {% %}}) also get Jinja2's <b>default</b>
 * {@code trim_blocks}/{@code lstrip_blocks} treatment even without explicit
 * dashes (one newline eaten after a block tag; same-line leading
 * spaces/tabs eaten before one) — matching the environment HF's
 * {@code apply_chat_template} and llama.cpp's renderer both use, since most
 * real-world templates (e.g. TinyLlama/Zephyr-style) rely on this default
 * rather than writing dashes themselves. This does <b>not</b> apply around
 * {@code {{ }}} output tags, only {@code {% %}} block tags — see
 * {@link #applyBlockDefaults}.
 * <li>{@code {% for x in messages %} ... {% endfor %}} loops, exposing a
 * {@code loop} object ({@code index0}, {@code index}, {@code first},
 * {@code last}, {@code length})
 * <li>{@code {% if cond %} ... {% elif cond %} ... {% else %} ... {% endif %}}
 * <li>{@code {# comment #}} (discarded)
 * <li>expressions: string/number/bool literals, variable and member access
 * ({@code message.role}, {@code message['role']}), string concatenation
 * ({@code +}), comparisons ({@code == != < <= > >=}), boolean
 * ({@code and or not}), the {@code trim} filter
 * </ul>
 *
 * <p><b>Explicitly not supported</b> (throws {@link MiniJinjaException} at
 * parse time so callers can fail closed to a named template): {@code set},
 * macros, includes, arbitrary filters/tests, arithmetic beyond string
 * concatenation, non-literal bracket indices. This is a deliberate scope cut,
 * not a defect.
 *
 * <p>Stateless and immutable after construction — a parsed instance is safe
 * to share and render concurrently.
 */
final class MiniJinjaTemplate {

	// ── Segment scanning (delimiters + whitespace trim markers) ────────────────

	private enum SegType {
		TEXT, OUTPUT, TAG
	}

	private record Seg(SegType type, String value, boolean trimLeft, boolean trimRight) {
		static Seg text(String v) {
			return new Seg(SegType.TEXT, v, false, false);
		}
	}

	private static List<Seg> tokenize(String src) {
		List<Seg> segs = new ArrayList<>();
		int i = 0, n = src.length();
		while (i < n) {
			int nextTag = src.indexOf("{%", i);
			int nextOut = src.indexOf("{{", i);
			int nextCom = src.indexOf("{#", i);
			int next = minPositive(nextTag, minPositive(nextOut, nextCom));
			if (next < 0) {
				segs.add(Seg.text(src.substring(i)));
				break;
			}
			if (next > i)
				segs.add(Seg.text(src.substring(i, next)));
			if (next == nextCom) {
				int end = src.indexOf("#}", next + 2);
				if (end < 0)
					throw new MiniJinjaException("unterminated {# comment");
				i = end + 2;
				continue;
			}
			boolean isTag = next == nextTag;
			String closeDelim = isTag ? "%}" : "}}";
			int contentStart = next + 2;
			boolean trimLeft = contentStart < n && src.charAt(contentStart) == '-';
			if (trimLeft)
				contentStart++;
			int end = src.indexOf(closeDelim, contentStart);
			if (end < 0)
				throw new MiniJinjaException("unterminated " + (isTag ? "{%" : "{{") + " tag");
			int contentEnd = end;
			boolean trimRight = contentEnd > contentStart && src.charAt(contentEnd - 1) == '-';
			if (trimRight)
				contentEnd--;
			String content = src.substring(contentStart, contentEnd).strip();
			segs.add(new Seg(isTag ? SegType.TAG : SegType.OUTPUT, content, trimLeft, trimRight));
			i = end + 2;
		}
		applyBlockDefaults(segs);
		applyTrim(segs);
		return segs;
	}

	/**
	 * Mirrors Jinja2's environment defaults {@code trim_blocks=True} and
	 * {@code lstrip_blocks=True} — the settings HF's {@code apply_chat_template}
	 * and llama.cpp's own template renderer both use. Most real-world
	 * {@code chat_template} strings (e.g. TinyLlama/Zephyr-style) are authored
	 * assuming these defaults and never write explicit {@code {%-}/{-%}}
	 * markers; without this, every block tag on its own line leaves a blank
	 * line behind, and the noise compounds with every conversation turn.
	 * Applies to {@code {% %}} tags only (matching Jinja2's own scope), and
	 * runs before {@link #applyTrim}, which handles the explicit-dash case.
	 */
	private static void applyBlockDefaults(List<Seg> segs) {
		for (int k = 0; k < segs.size(); k++) {
			Seg s = segs.get(k);
			if (s.type() != SegType.TAG)
				continue;
			if (k > 0 && segs.get(k - 1).type() == SegType.TEXT) {
				Seg prev = segs.get(k - 1);
				segs.set(k - 1, Seg.text(lstripBlockLine(prev.value())));
			}
			if (k + 1 < segs.size() && segs.get(k + 1).type() == SegType.TEXT) {
				Seg nxt = segs.get(k + 1);
				segs.set(k + 1, Seg.text(trimOneLeadingNewline(nxt.value())));
			}
		}
	}

	/** lstrip_blocks: strip spaces/tabs from the start of a line up to a block tag, when
	 * that line is otherwise blank so far (i.e. the text since the last newline is pure
	 * whitespace-or-tabs). Leaves the newline itself and everything before it untouched. */
	private static String lstripBlockLine(String s) {
		int lastNl = s.lastIndexOf('\n');
		String tail = s.substring(lastNl + 1);
		for (int i = 0; i < tail.length(); i++) {
			char c = tail.charAt(i);
			if (c != ' ' && c != '\t')
				return s;
		}
		return s.substring(0, lastNl + 1);
	}

	/** trim_blocks: strip exactly one newline (or "\r\n") immediately following a block tag. */
	private static String trimOneLeadingNewline(String s) {
		if (s.startsWith("\r\n"))
			return s.substring(2);
		if (s.startsWith("\n"))
			return s.substring(1);
		return s;
	}

	private static int minPositive(int a, int b) {
		if (a < 0)
			return b;
		if (b < 0)
			return a;
		return Math.min(a, b);
	}

	private static void applyTrim(List<Seg> segs) {
		for (int k = 0; k < segs.size(); k++) {
			Seg s = segs.get(k);
			if (s.type() == SegType.TEXT)
				continue;
			if (s.trimLeft() && k > 0 && segs.get(k - 1).type() == SegType.TEXT) {
				Seg prev = segs.get(k - 1);
				segs.set(k - 1, Seg.text(stripTrailingWs(prev.value())));
			}
			if (s.trimRight() && k + 1 < segs.size() && segs.get(k + 1).type() == SegType.TEXT) {
				Seg nxt = segs.get(k + 1);
				segs.set(k + 1, Seg.text(stripLeadingWs(nxt.value())));
			}
		}
	}

	private static String stripTrailingWs(String s) {
		int end = s.length();
		while (end > 0 && Character.isWhitespace(s.charAt(end - 1)))
			end--;
		return s.substring(0, end);
	}

	private static String stripLeadingWs(String s) {
		int start = 0;
		while (start < s.length() && Character.isWhitespace(s.charAt(start)))
			start++;
		return s.substring(start);
	}

	// ── AST ──────────────────────────────────────────────────────────────────

	private sealed interface Node permits TextNode, OutputNode, ForNode, IfNode {
	}

	private record TextNode(String text) implements Node {
	}

	private record OutputNode(Expr expr) implements Node {
	}

	private record ForNode(String varName, Expr iterable, List<Node> body) implements Node {
	}

	private record Branch(Expr cond, List<Node> body) {
	}

	private record IfNode(List<Branch> branches, List<Node> elseBody) implements Node {
	}

	// ── Parser (segment cursor → nested AST) ────────────────────────────────────

	private static final class Cursor {
		final List<Seg> segs;
		int pos;

		Cursor(List<Seg> segs) {
			this.segs = segs;
		}
	}

	private static String firstWord(String tagContent) {
		int sp = tagContent.indexOf(' ');
		return sp < 0 ? tagContent : tagContent.substring(0, sp);
	}

	private static List<Node> parseBlock(Cursor c, Set<String> stopKeywords) {
		List<Node> nodes = new ArrayList<>();
		while (c.pos < c.segs.size()) {
			Seg s = c.segs.get(c.pos);
			if (s.type() == SegType.TEXT) {
				if (!s.value().isEmpty())
					nodes.add(new TextNode(s.value()));
				c.pos++;
				continue;
			}
			if (s.type() == SegType.OUTPUT) {
				nodes.add(new OutputNode(ExprParser.parse(s.value())));
				c.pos++;
				continue;
			}
			String keyword = firstWord(s.value());
			if (stopKeywords.contains(keyword))
				return nodes;
			switch (keyword) {
			case "for" -> {
				c.pos++;
				String rest = s.value().length() > 3 ? s.value().substring(3).strip() : "";
				int inIdx = rest.indexOf(" in ");
				if (inIdx < 0)
					throw new MiniJinjaException("malformed for-tag: " + s.value());
				String varName = rest.substring(0, inIdx).strip();
				if (!varName.matches("[A-Za-z_][A-Za-z0-9_]*"))
					throw new MiniJinjaException("bad loop variable name: " + varName);
				Expr iterable = ExprParser.parse(rest.substring(inIdx + 4).strip());
				List<Node> body = parseBlock(c, Set.of("endfor"));
				expectTag(c, "endfor");
				c.pos++;
				nodes.add(new ForNode(varName, iterable, body));
			}
			case "if" -> {
				List<Branch> branches = new ArrayList<>();
				Expr cond = ExprParser.parse(tagArgs(s.value(), "if"));
				c.pos++;
				List<Node> body = parseBlock(c, Set.of("elif", "else", "endif"));
				branches.add(new Branch(cond, body));
				List<Node> elseBody = List.of();
				while (true) {
					Seg tag = c.segs.get(c.pos);
					String kw = firstWord(tag.value());
					if (kw.equals("elif")) {
						Expr econd = ExprParser.parse(tagArgs(tag.value(), "elif"));
						c.pos++;
						List<Node> ebody = parseBlock(c, Set.of("elif", "else", "endif"));
						branches.add(new Branch(econd, ebody));
					} else if (kw.equals("else")) {
						c.pos++;
						elseBody = parseBlock(c, Set.of("endif"));
						expectTag(c, "endif");
						c.pos++;
						break;
					} else if (kw.equals("endif")) {
						c.pos++;
						break;
					} else {
						throw new MiniJinjaException("expected elif/else/endif, found: " + kw);
					}
				}
				nodes.add(new IfNode(branches, elseBody));
			}
			default -> throw new MiniJinjaException("unsupported tag: {% " + s.value() + " %}");
			}
		}
		if (!stopKeywords.isEmpty())
			throw new MiniJinjaException("unexpected end of template, expected one of " + stopKeywords);
		return nodes;
	}

	private static String tagArgs(String tagContent, String keyword) {
		return tagContent.length() > keyword.length() ? tagContent.substring(keyword.length()).strip() : "";
	}

	private static void expectTag(Cursor c, String keyword) {
		if (c.pos >= c.segs.size() || c.segs.get(c.pos).type() != SegType.TAG
				|| !firstWord(c.segs.get(c.pos).value()).equals(keyword))
			throw new MiniJinjaException("expected {% " + keyword + " %}");
	}

	// ── Expression AST + parser ─────────────────────────────────────────────

	private sealed interface Expr permits Lit, Var, Member, BinOp, Not, FilterExpr {
		Object eval(Map<String, Object> ctx);
	}

	private record Lit(Object value) implements Expr {
		@Override
		public Object eval(Map<String, Object> ctx) {
			return value;
		}
	}

	private record Var(String name) implements Expr {
		@Override
		public Object eval(Map<String, Object> ctx) {
			return ctx.get(name);
		}
	}

	private record Member(Expr base, String key) implements Expr {
		@Override
		public Object eval(Map<String, Object> ctx) {
			Object b = base.eval(ctx);
			if (b instanceof Map<?, ?> m)
				return m.get(key);
			return null;
		}
	}

	private record BinOp(String op, Expr left, Expr right) implements Expr {
		@Override
		public Object eval(Map<String, Object> ctx) {
			return switch (op) {
			case "+" -> stringify(left.eval(ctx)) + stringify(right.eval(ctx));
			case "and" -> truthy(left.eval(ctx)) ? right.eval(ctx) : left.eval(ctx);
			case "or" -> truthy(left.eval(ctx)) ? left.eval(ctx) : right.eval(ctx);
			case "==" -> equalsLoose(left.eval(ctx), right.eval(ctx));
			case "!=" -> !equalsLoose(left.eval(ctx), right.eval(ctx));
			case "<", "<=", ">", ">=" -> compare(op, left.eval(ctx), right.eval(ctx));
			default -> throw new MiniJinjaException("unsupported operator: " + op);
			};
		}
	}

	private record Not(Expr inner) implements Expr {
		@Override
		public Object eval(Map<String, Object> ctx) {
			return !truthy(inner.eval(ctx));
		}
	}

	/** Filter names accepted by the parser — checked eagerly so a typo'd or exotic
	 * filter fails at parse time, not mid-render. */
	private static final Set<String> SUPPORTED_FILTERS = Set.of("trim", "upper", "lower");

	private record FilterExpr(Expr base, String filter) implements Expr {
		@Override
		public Object eval(Map<String, Object> ctx) {
			String v = stringify(base.eval(ctx));
			return switch (filter) {
			case "trim" -> v.strip();
			case "upper" -> v.toUpperCase(java.util.Locale.ROOT);
			case "lower" -> v.toLowerCase(java.util.Locale.ROOT);
			default -> throw new MiniJinjaException("unsupported filter: " + filter);
			};
		}
	}

	private static boolean equalsLoose(Object a, Object b) {
		if (a == null || b == null)
			return a == b;
		return a.equals(b);
	}

	private static boolean compare(String op, Object a, Object b) {
		double x, y;
		try {
			x = Double.parseDouble(stringify(a));
			y = Double.parseDouble(stringify(b));
		} catch (NumberFormatException e) {
			throw new MiniJinjaException("non-numeric comparison: " + op);
		}
		return switch (op) {
		case "<" -> x < y;
		case "<=" -> x <= y;
		case ">" -> x > y;
		case ">=" -> x >= y;
		default -> throw new MiniJinjaException("unsupported comparison: " + op);
		};
	}

	private static boolean truthy(Object v) {
		if (v == null)
			return false;
		if (v instanceof Boolean b)
			return b;
		if (v instanceof String s)
			return !s.isEmpty();
		if (v instanceof Number n)
			return n.doubleValue() != 0;
		if (v instanceof java.util.Collection<?> col)
			return !col.isEmpty();
		return true;
	}

	private static String stringify(Object v) {
		if (v == null)
			return "";
		if (v instanceof Boolean b)
			return b ? "true" : "false";
		return v.toString();
	}

	/** Recursive-descent expression parser over a raw expression substring. */
	private static final class ExprParser {
		private final String src;
		private int pos;

		private ExprParser(String src) {
			this.src = src;
			this.pos = 0;
		}

		static Expr parse(String src) {
			if (src == null || src.isBlank())
				throw new MiniJinjaException("empty expression");
			ExprParser p = new ExprParser(src);
			Expr e = p.parseOr();
			p.skipWs();
			if (p.pos < p.src.length())
				throw new MiniJinjaException("unexpected trailing expression content: " + p.src.substring(p.pos));
			return e;
		}

		private Expr parseOr() {
			Expr left = parseAnd();
			while (true) {
				skipWs();
				if (matchWord("or")) {
					left = new BinOp("or", left, parseAnd());
				} else
					return left;
			}
		}

		private Expr parseAnd() {
			Expr left = parseNot();
			while (true) {
				skipWs();
				if (matchWord("and")) {
					left = new BinOp("and", left, parseNot());
				} else
					return left;
			}
		}

		private Expr parseNot() {
			skipWs();
			if (matchWord("not"))
				return new Not(parseNot());
			return parseCompare();
		}

		private Expr parseCompare() {
			Expr left = parseConcat();
			skipWs();
			String op = matchAnyOp("==", "!=", ">=", "<=", ">", "<");
			if (op != null)
				return new BinOp(op, left, parseConcat());
			return left;
		}

		private Expr parseConcat() {
			Expr left = parseFilter();
			while (true) {
				skipWs();
				if (pos < src.length() && src.charAt(pos) == '+') {
					pos++;
					left = new BinOp("+", left, parseFilter());
				} else
					return left;
			}
		}

		private Expr parseFilter() {
			Expr base = parsePrimary();
			while (true) {
				skipWs();
				if (pos < src.length() && src.charAt(pos) == '|') {
					pos++;
					skipWs();
					String name = ident();
					skipWs();
					if (pos < src.length() && src.charAt(pos) == '(') {
						// no-arg-only filters supported; reject any filter call with args.
						throw new MiniJinjaException("filter arguments are not supported: " + name);
					}
					if (!SUPPORTED_FILTERS.contains(name))
						throw new MiniJinjaException("unsupported filter: " + name);
					base = new FilterExpr(base, name);
				} else
					return base;
			}
		}

		private Expr parsePrimary() {
			skipWs();
			if (pos >= src.length())
				throw new MiniJinjaException("unexpected end of expression");
			char ch = src.charAt(pos);
			if (ch == '\'' || ch == '"')
				return trailer(new Lit(stringLiteral()));
			if (ch == '(') {
				pos++;
				Expr inner = parseOr();
				skipWs();
				expectChar(')');
				return trailer(inner);
			}
			if (Character.isDigit(ch)) {
				return trailer(new Lit(number()));
			}
			String id = ident();
			return switch (id) {
			case "true" -> trailer(new Lit(Boolean.TRUE));
			case "false" -> trailer(new Lit(Boolean.FALSE));
			case "none", "null" -> trailer(new Lit(null));
			case "" -> throw new MiniJinjaException("expected identifier at: " + src.substring(pos));
			default -> trailer(new Var(id));
			};
		}

		/** Consume {@code .name} / {@code ['literal']} trailers after a primary. */
		private Expr trailer(Expr base) {
			while (true) {
				skipWs();
				if (pos < src.length() && src.charAt(pos) == '.') {
					pos++;
					String name = ident();
					if (name.isEmpty())
						throw new MiniJinjaException("expected member name after '.'");
					base = new Member(base, name);
					continue;
				}
				if (pos < src.length() && src.charAt(pos) == '[') {
					pos++;
					skipWs();
					if (pos >= src.length() || (src.charAt(pos) != '\'' && src.charAt(pos) != '"'))
						throw new MiniJinjaException("only string-literal bracket indices are supported");
					String key = stringLiteral();
					skipWs();
					expectChar(']');
					base = new Member(base, key);
					continue;
				}
				return base;
			}
		}

		private String stringLiteral() {
			char quote = src.charAt(pos);
			pos++;
			StringBuilder sb = new StringBuilder();
			while (pos < src.length() && src.charAt(pos) != quote) {
				char c = src.charAt(pos);
				if (c == '\\' && pos + 1 < src.length()) {
					char next = src.charAt(pos + 1);
					sb.append(switch (next) {
					case 'n' -> '\n';
					case 't' -> '\t';
					case 'r' -> '\r';
					case '\\' -> '\\';
					case '\'' -> '\'';
					case '"' -> '"';
					default -> next;
					});
					pos += 2;
				} else {
					sb.append(c);
					pos++;
				}
			}
			if (pos >= src.length())
				throw new MiniJinjaException("unterminated string literal");
			pos++; // closing quote
			return sb.toString();
		}

		private Double number() {
			int start = pos;
			while (pos < src.length() && (Character.isDigit(src.charAt(pos)) || src.charAt(pos) == '.'))
				pos++;
			return Double.parseDouble(src.substring(start, pos));
		}

		private String ident() {
			int start = pos;
			while (pos < src.length() && (Character.isLetterOrDigit(src.charAt(pos)) || src.charAt(pos) == '_'))
				pos++;
			return src.substring(start, pos);
		}

		private void skipWs() {
			while (pos < src.length() && Character.isWhitespace(src.charAt(pos)))
				pos++;
		}

		private void expectChar(char c) {
			if (pos >= src.length() || src.charAt(pos) != c)
				throw new MiniJinjaException("expected '" + c + "' at: " + src.substring(pos));
			pos++;
		}

		/** Match a keyword operator ("and"/"or"/"not") as a whole word, consuming it. */
		private boolean matchWord(String word) {
			int end = pos + word.length();
			if (end <= src.length() && src.regionMatches(pos, word, 0, word.length())
					&& (end == src.length() || !Character.isLetterOrDigit(src.charAt(end)))) {
				pos = end;
				skipWs();
				return true;
			}
			return false;
		}

		private String matchAnyOp(String... ops) {
			for (String op : ops) {
				if (src.regionMatches(pos, op, 0, op.length())) {
					pos += op.length();
					skipWs();
					return op;
				}
			}
			return null;
		}
	}

	// ── Public API ───────────────────────────────────────────────────────────

	private final List<Node> root;

	/**
	 * Parse {@code source} into an immutable AST. Throws {@link MiniJinjaException}
	 * for any syntax this restricted engine does not support — callers should
	 * catch this and fall back to a named {@link ChatTemplate}.
	 */
	MiniJinjaTemplate(String source) {
		if (source == null || source.isBlank())
			throw new MiniJinjaException("empty template source");
		// Mirrors Jinja2's default keep_trailing_newline=False: a template file's
		// own single trailing newline is not part of the rendered output.
		if (source.endsWith("\n"))
			source = source.substring(0, source.length() - 1);
		Cursor c = new Cursor(tokenize(source));
		this.root = parseBlock(c, Set.of());
		if (c.pos < c.segs.size())
			throw new MiniJinjaException("unexpected trailing tag: " + c.segs.get(c.pos).value());
	}

	/**
	 * Render this template against {@code context}. Throws
	 * {@link MiniJinjaException} on any runtime construct the restricted engine
	 * does not support (e.g. a for-loop target that is not a list) — callers
	 * should catch this and fall back to a named {@link ChatTemplate}.
	 */
	String render(Map<String, Object> context) {
		StringBuilder out = new StringBuilder();
		exec(root, new HashMap<>(context), out);
		return out.toString();
	}

	private static void exec(List<Node> nodes, Map<String, Object> ctx, StringBuilder out) {
		for (Node n : nodes) {
			if (n instanceof TextNode t) {
				out.append(t.text());
			} else if (n instanceof OutputNode o) {
				out.append(stringify(o.expr().eval(ctx)));
			} else if (n instanceof ForNode f) {
				Object iterableVal = f.iterable().eval(ctx);
				if (!(iterableVal instanceof List<?> list))
					throw new MiniJinjaException("for-loop target is not a list: " + f.iterable());
				int total = list.size();
				for (int idx = 0; idx < total; idx++) {
					Map<String, Object> loopCtx = new HashMap<>(ctx);
					loopCtx.put(f.varName(), list.get(idx));
					Map<String, Object> loopObj = new HashMap<>();
					loopObj.put("index0", (double) idx);
					loopObj.put("index", (double) (idx + 1));
					loopObj.put("first", idx == 0);
					loopObj.put("last", idx == total - 1);
					loopObj.put("length", (double) total);
					loopCtx.put("loop", loopObj);
					exec(f.body(), loopCtx, out);
				}
			} else if (n instanceof IfNode iff) {
				boolean matched = false;
				for (Branch b : iff.branches()) {
					if (truthy(b.cond().eval(ctx))) {
						exec(b.body(), ctx, out);
						matched = true;
						break;
					}
				}
				if (!matched)
					exec(iff.elseBody(), ctx, out);
			}
		}
	}
}
