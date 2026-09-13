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

/**
 * Compiled GBNF grammar used for constrained decoding. Matching is a
 * byte-level pushdown automaton (UTF-8).
 */
public final class GbnfGrammar {

	static final int END = 0;
	static final int ALT = 1;
	static final int REF = 2;
	static final int CHAR = 3;
	static final int CLS = 4;

	private final List<int[]> rules;
	private final boolean[][] classes;
	private final int rootId;

	GbnfGrammar(List<int[]> rules, boolean[][] classes, int rootId) {
		this.rules = rules;
		this.classes = classes;
		this.rootId = rootId;
	}

	public static GbnfGrammar parse(String source) {
		if (source == null || source.isBlank())
			throw new IllegalArgumentException("grammar text is empty");
		Map<String, Expr> named = new GbnfParser(source).parseFile();
		if (!named.containsKey("root"))
			throw new IllegalArgumentException("grammar must define a 'root' rule");
		return new GbnfCompiler(named).compile();
	}

	public boolean acceptsPrefix(byte[] bytes) {
		State st = State.start(this);
		return st != null && st.consume(bytes);
	}

	public boolean isComplete(byte[] bytes) {
		State st = State.start(this);
		return st != null && st.consume(bytes) && st.accepting;
	}

	/**
	 * Bytes that can legally extend {@code prefix}. Index is unsigned byte 0–255.
	 */
	public boolean[] nextBytes(byte[] prefix) {
		boolean[] out = new boolean[256];
		State st = State.start(this);
		if (st == null || !st.consume(prefix))
			return out;
		for (int b = 0; b < 256; b++) {
			State c = st.copy();
			if (c.consume(new byte[] { (byte) b }))
				out[b] = true;
		}
		return out;
	}

	State newState() {
		return State.start(this);
	}

	static final class State {
		private final GbnfGrammar g;
		private List<int[]> stacks;
		private boolean accepting;

		private State(GbnfGrammar g, List<int[]> stacks, boolean accepting) {
			this.g = g;
			this.stacks = stacks;
			this.accepting = accepting;
		}

		static State start(GbnfGrammar g) {
			List<int[]> raw = new ArrayList<>();
			raw.add(new int[] { pack(g.rootId, 0) });
			State s = new State(g, raw, false);
			s.epsilon();
			if (s.stacks.isEmpty() && !s.accepting)
				return null;
			return s;
		}

		State copy() {
			List<int[]> c = new ArrayList<>(stacks.size());
			for (int[] st : stacks)
				c.add(st.clone());
			return new State(g, c, accepting);
		}

		boolean accepting() {
			return accepting;
		}

		boolean[] nextBytes() {
			boolean[] out = new boolean[256];
			for (int b = 0; b < 256; b++) {
				State c = copy();
				if (c.consume(new byte[] { (byte) b }))
					out[b] = true;
			}
			return out;
		}

		boolean consume(byte[] bytes) {
			if (bytes == null)
				return false;
			for (byte b : bytes) {
				if (!consumeOne(b))
					return false;
			}
			return true;
		}

		private boolean consumeOne(byte b) {
			if (stacks.isEmpty())
				return false;
			List<int[]> next = new ArrayList<>();
			for (int[] st : stacks) {
				Holder h = new Holder();
				int[] copy = grow(st.clone(), st.length + 4);
				matchChar(copy, st.length, b & 0xff, next, h);
			}
			stacks = dedup(next);
			accepting = false;
			epsilon();
			return !stacks.isEmpty() || accepting;
		}

		private void epsilon() {
			List<int[]> out = new ArrayList<>();
			boolean acc = false;
			for (int[] st : stacks) {
				Holder h = new Holder();
				int[] copy = grow(st.clone(), st.length + 4);
				expand(copy, st.length, out, h);
				acc |= h.accepting;
			}
			stacks = dedup(out);
			accepting = acc;
		}

		private void expand(int[] stack, int sp, List<int[]> out, Holder h) {
			if (sp == 0) {
				h.accepting = true;
				return;
			}
			int packed = stack[sp - 1];
			int rule = packed >>> 16;
			int pc = packed & 0xFFFF;
			int[] r = g.rules.get(rule);
			if (pc >= r.length) {
				popAdvance(stack, sp, out, h, true);
				return;
			}
			int kind = r[pc] >>> 24;
			int val = r[pc] & 0xFFFFFF;
			int npc = skipToAlt(r, pc);
			switch (kind) {
			case REF -> {
				if (!leftRec(stack, sp, val)) {
					int[] pushed = grow(java.util.Arrays.copyOf(stack, sp), sp + 1);
					pushed[sp] = pack(val, 0);
					expand(pushed, sp + 1, out, h);
				}
				if (npc >= 0) {
					int[] alt = java.util.Arrays.copyOf(stack, sp);
					alt[sp - 1] = pack(rule, npc);
					expand(grow(alt, sp + 1), sp, out, h);
				}
			}
			case END, ALT -> popAdvance(stack, sp, out, h, true);
			case CHAR, CLS -> {
				out.add(java.util.Arrays.copyOf(stack, sp));
				if (npc >= 0) {
					int[] alt = java.util.Arrays.copyOf(stack, sp);
					alt[sp - 1] = pack(rule, npc);
					expand(grow(alt, sp + 1), sp, out, h);
				}
			}
			default -> {
			}
			}
		}

		private void matchChar(int[] stack, int sp, int b, List<int[]> out, Holder h) {
			if (sp == 0)
				return;
			int packed = stack[sp - 1];
			int rule = packed >>> 16;
			int pc = packed & 0xFFFF;
			int[] r = g.rules.get(rule);
			if (pc >= r.length) {
				popAdvanceMatch(stack, sp, b, out, h);
				return;
			}
			int kind = r[pc] >>> 24;
			int val = r[pc] & 0xFFFFFF;
			int npc = skipToAlt(r, pc);
			switch (kind) {
			case REF -> {
				if (!leftRec(stack, sp, val)) {
					int[] pushed = grow(java.util.Arrays.copyOf(stack, sp), sp + 1);
					pushed[sp] = pack(val, 0);
					matchChar(pushed, sp + 1, b, out, h);
				}
				if (npc >= 0) {
					int[] alt = java.util.Arrays.copyOf(stack, sp);
					alt[sp - 1] = pack(rule, npc);
					matchChar(grow(alt, sp + 1), sp, b, out, h);
				}
			}
			case END, ALT -> popAdvanceMatch(stack, sp, b, out, h);
			case CHAR -> {
				if (val == b) {
					stack[sp - 1] = pack(rule, pc + 1);
					out.add(java.util.Arrays.copyOf(stack, sp));
				} else {
					tryAlt(stack, sp, rule, pc, b, out, h);
				}
			}
			case CLS -> {
				if (g.classes[val][b]) {
					stack[sp - 1] = pack(rule, pc + 1);
					out.add(java.util.Arrays.copyOf(stack, sp));
				} else {
					tryAlt(stack, sp, rule, pc, b, out, h);
				}
			}
			default -> {
			}
			}
		}

		private void tryAlt(int[] stack, int sp, int rule, int pc, int b, List<int[]> out, Holder h) {
			int npc = skipToAlt(g.rules.get(rule), pc);
			if (npc < 0)
				return;
			stack[sp - 1] = pack(rule, npc);
			matchChar(stack, sp, b, out, h);
		}

		private void popAdvance(int[] stack, int sp, List<int[]> out, Holder h, boolean epsilon) {
			if (sp <= 1) {
				h.accepting = true;
				return;
			}
			int parent = stack[sp - 2];
			int pr = parent >>> 16;
			int ppc = parent & 0xFFFF;
			stack[sp - 2] = pack(pr, ppc + 1);
			if (epsilon)
				expand(stack, sp - 1, out, h);
			else
				out.add(java.util.Arrays.copyOf(stack, sp - 1));
		}

		private void popAdvanceMatch(int[] stack, int sp, int b, List<int[]> out, Holder h) {
			if (sp <= 1)
				return;
			int parent = stack[sp - 2];
			int pr = parent >>> 16;
			int ppc = parent & 0xFFFF;
			stack[sp - 2] = pack(pr, ppc + 1);
			matchChar(stack, sp - 1, b, out, h);
		}

		private static int skipToAlt(int[] r, int pc) {
			for (int i = pc; i < r.length; i++) {
				int kind = r[i] >>> 24;
				if (kind == ALT)
					return i + 1;
				if (kind == END)
					return -1;
			}
			return -1;
		}

		private static boolean leftRec(int[] stack, int sp, int rule) {
			for (int i = 0; i < sp; i++) {
				if ((stack[i] >>> 16) == rule && (stack[i] & 0xFFFF) == 0)
					return true;
			}
			return false;
		}

		private static int[] grow(int[] stack, int need) {
			if (stack.length >= need)
				return stack;
			return java.util.Arrays.copyOf(stack, Math.max(need, stack.length * 2));
		}

		private static List<int[]> dedup(List<int[]> in) {
			if (in.size() <= 1)
				return in;
			List<int[]> out = new ArrayList<>(in.size());
			loop: for (int[] a : in) {
				for (int[] b : out) {
					if (java.util.Arrays.equals(a, b))
						continue loop;
				}
				out.add(a);
			}
			return out;
		}

		private static int pack(int rule, int pc) {
			if (rule > 0xFFFF || pc > 0xFFFF)
				throw new IllegalArgumentException("rule/pc overflow");
			return (rule << 16) | pc;
		}

		private static final class Holder {
			boolean accepting;
		}
	}

	sealed interface Expr {
		record Lit(byte[] bytes) implements Expr {
		}

		record Cls(boolean[] allow) implements Expr {
		}

		record Seq(List<Expr> items) implements Expr {
		}

		record Alt(List<Expr> alts) implements Expr {
		}

		record Rep(Expr inner, int min, int max) implements Expr {
		}

		record Ref(String name) implements Expr {
		}
	}

	private static final class GbnfCompiler {
		private final Map<String, Expr> named;
		private final List<int[]> rules = new ArrayList<>();
		private final List<boolean[]> classList = new ArrayList<>();
		private final Map<String, Integer> ids = new LinkedHashMap<>();

		GbnfCompiler(Map<String, Expr> named) {
			this.named = named;
		}

		GbnfGrammar compile() {
			for (String name : named.keySet()) {
				ids.put(name, rules.size());
				rules.add(null);
			}
			for (Map.Entry<String, Expr> e : named.entrySet())
				rules.set(ids.get(e.getKey()), compileRule(e.getValue()));
			int root = ids.get("root");
			boolean[][] cls = new boolean[classList.size()][];
			for (int c = 0; c < classList.size(); c++)
				cls[c] = classList.get(c);
			return new GbnfGrammar(rules, cls, root);
		}

		private int[] compileRule(Expr e) {
			List<Integer> elems = new ArrayList<>();
			emitAlts(flattenAlt(e), elems);
			elems.add(elem(END, 0));
			return toArray(elems);
		}

		private void emitAlts(List<Expr> alts, List<Integer> elems) {
			for (int i = 0; i < alts.size(); i++) {
				if (i > 0)
					elems.add(elem(ALT, 0));
				emitSeq(flattenSeq(alts.get(i)), elems);
			}
		}

		private void emitSeq(List<Expr> items, List<Integer> elems) {
			for (Expr it : items)
				emitAtom(it, elems);
		}

		private void emitAtom(Expr e, List<Integer> elems) {
			switch (e) {
			case Expr.Lit lit -> {
				for (byte b : lit.bytes())
					elems.add(elem(CHAR, b & 0xff));
			}
			case Expr.Cls cls -> elems.add(elem(CLS, addClass(cls.allow())));
			case Expr.Ref ref -> elems.add(elem(REF, ruleId(ref.name())));
			case Expr.Rep rep -> elems.add(elem(REF, compileRep(rep)));
			case Expr.Seq seq -> {
				if (seq.items().isEmpty())
					return;
				if (seq.items().size() == 1) {
					emitAtom(seq.items().get(0), elems);
					return;
				}
				elems.add(elem(REF, addAnon(seq)));
			}
			case Expr.Alt alt -> elems.add(elem(REF, addAnon(alt)));
			}
		}

		private int compileRep(Expr.Rep rep) {
			Expr inner = rep.inner();
			int min = rep.min();
			int max = rep.max();
			if (min == 0 && max == 1)
				return addAnon(new Expr.Alt(List.of(inner, new Expr.Seq(List.of()))));
			if (min == 0 && max == Integer.MAX_VALUE) {
				String star = "_star" + rules.size();
				int id = newRuleSlot(star);
				Expr.Ref self = new Expr.Ref(star);
				rules.set(id, compileRule(new Expr.Alt(List.of(new Expr.Seq(List.of(inner, self)), new Expr.Seq(List.of())))));
				return id;
			}
			if (min == 1 && max == Integer.MAX_VALUE) {
				String plus = "_plus" + rules.size();
				int id = newRuleSlot(plus);
				Expr.Ref self = new Expr.Ref(plus);
				rules.set(id, compileRule(new Expr.Alt(List.of(new Expr.Seq(List.of(inner, self)), inner))));
				return id;
			}
			List<Expr> seq = new ArrayList<>();
			for (int i = 0; i < min; i++)
				seq.add(inner);
			for (int i = min; i < Math.min(max, min + 8); i++)
				seq.add(new Expr.Rep(inner, 0, 1));
			if (max > min + 8)
				throw new IllegalArgumentException("bounded repetition max too large: " + max);
			return addAnon(new Expr.Seq(seq));
		}

		private int addAnon(Expr e) {
			String name = "_anon" + rules.size();
			int id = newRuleSlot(name);
			rules.set(id, compileRule(e));
			return id;
		}

		private int newRuleSlot(String name) {
			int id = rules.size();
			ids.put(name, id);
			rules.add(null);
			return id;
		}

		private int ruleId(String name) {
			Integer id = ids.get(name);
			if (id == null)
				throw new IllegalArgumentException("undefined rule '" + name + "'");
			return id;
		}

		private int addClass(boolean[] allow) {
			classList.add(allow);
			return classList.size() - 1;
		}

		private static List<Expr> flattenAlt(Expr e) {
			if (e instanceof Expr.Alt a)
				return a.alts();
			return List.of(e);
		}

		private static List<Expr> flattenSeq(Expr e) {
			if (e instanceof Expr.Seq s)
				return s.items();
			return List.of(e);
		}

		private static int elem(int kind, int val) {
			return (kind << 24) | (val & 0xFFFFFF);
		}

		private static int[] toArray(List<Integer> elems) {
			int[] a = new int[elems.size()];
			for (int i = 0; i < a.length; i++)
				a[i] = elems.get(i);
			return a;
		}
	}

	private static final class GbnfParser {
		private final String s;
		private int i;
		private final int n;

		GbnfParser(String s) {
			this.s = s;
			this.n = s.length();
		}

		Map<String, Expr> parseFile() {
			Map<String, Expr> rules = new LinkedHashMap<>();
			while (true) {
				skipWsAndComments();
				if (i >= n)
					break;
				String name = parseIdent();
				skipWsAndComments();
				expect("::=");
				skipWsAndComments();
				Expr rhs = parseAlts();
				if (rules.put(name, rhs) != null)
					throw err("duplicate rule '" + name + "'");
			}
			if (rules.isEmpty())
				throw err("no rules");
			return rules;
		}

		private Expr parseAlts() {
			List<Expr> alts = new ArrayList<>();
			alts.add(parseSeq());
			while (true) {
				skipWsAndComments();
				if (i < n && s.charAt(i) == '|') {
					i++;
					skipWsAndComments();
					alts.add(parseSeq());
				} else
					break;
			}
			return alts.size() == 1 ? alts.get(0) : new Expr.Alt(List.copyOf(alts));
		}

		private Expr parseSeq() {
			List<Expr> items = new ArrayList<>();
			while (true) {
				skipWsAndComments();
				if (i >= n)
					break;
				char c = s.charAt(i);
				if (c == '|' || c == ')' || c == '#')
					break;
				if (c == ':' && i + 2 < n && s.charAt(i + 1) == ':' && s.charAt(i + 2) == '=')
					break;
				if (isIdentStart(c) && looksLikeNextRule())
					break;
				items.add(parseTerm());
			}
			if (items.isEmpty())
				return new Expr.Seq(List.of());
			return items.size() == 1 ? items.get(0) : new Expr.Seq(List.copyOf(items));
		}

		private boolean looksLikeNextRule() {
			int j = i;
			while (j < n && isIdentPart(s.charAt(j)))
				j++;
			while (j < n && Character.isWhitespace(s.charAt(j)))
				j++;
			return j + 2 < n && s.charAt(j) == ':' && s.charAt(j + 1) == ':' && s.charAt(j + 2) == '=';
		}

		private Expr parseTerm() {
			Expr atom = parseAtom();
			if (i < n) {
				char q = s.charAt(i);
				if (q == '*') {
					i++;
					return new Expr.Rep(atom, 0, Integer.MAX_VALUE);
				}
				if (q == '+') {
					i++;
					return new Expr.Rep(atom, 1, Integer.MAX_VALUE);
				}
				if (q == '?') {
					i++;
					return new Expr.Rep(atom, 0, 1);
				}
			}
			return atom;
		}

		private Expr parseAtom() {
			skipWsAndComments();
			if (i >= n)
				throw err("unexpected end");
			char c = s.charAt(i);
			if (c == '"')
				return new Expr.Lit(parseString());
			if (c == '[')
				return new Expr.Cls(parseClass());
			if (c == '(') {
				i++;
				Expr inner = parseAlts();
				skipWsAndComments();
				expect(")");
				return inner;
			}
			if (isIdentStart(c))
				return new Expr.Ref(parseIdent());
			throw err("unexpected '" + c + "'");
		}

		private byte[] parseString() {
			expect("\"");
			StringBuilder sb = new StringBuilder();
			while (i < n) {
				char c = s.charAt(i++);
				if (c == '"')
					return sb.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8);
				if (c == '\\')
					sb.append(parseEscape());
				else
					sb.append(c);
			}
			throw err("unterminated string");
		}

		private boolean[] parseClass() {
			expect("[");
			boolean neg = i < n && s.charAt(i) == '^';
			if (neg)
				i++;
			boolean[] allow = new boolean[256];
			int prev = -1;
			boolean range = false;
			boolean any = false;
			while (i < n) {
				char c = s.charAt(i);
				if (c == ']' && any) {
					i++;
					if (neg) {
						for (int b = 0; b < 256; b++)
							allow[b] = !allow[b];
					}
					return allow;
				}
				i++;
				int ch;
				if (c == '\\')
					ch = parseEscape();
				else
					ch = c;
				if (range) {
					if (prev < 0 || ch < prev)
						throw err("bad char class range");
					for (int b = prev; b <= ch; b++)
						allow[b] = true;
					prev = -1;
					range = false;
					any = true;
				} else if (c != '\\' && ch == '-' && prev >= 0 && i < n && s.charAt(i) != ']') {
					range = true;
				} else {
					if (ch > 255)
						throw err("char class is byte-oriented (0-255)");
					allow[ch] = true;
					prev = ch;
					any = true;
				}
			}
			throw err("unterminated character class");
		}

		private char parseEscape() {
			if (i >= n)
				throw err("dangling escape");
			char c = s.charAt(i++);
			return switch (c) {
			case 'n' -> '\n';
			case 'r' -> '\r';
			case 't' -> '\t';
			case '\\', '"', '\'', '[', ']', '-' -> c;
			case 'x' -> (char) parseHex(2);
			case 'u' -> (char) parseHex(4);
			default -> c;
			};
		}

		private int parseHex(int digits) {
			int v = 0;
			for (int k = 0; k < digits; k++) {
				if (i >= n)
					throw err("bad hex escape");
				char c = s.charAt(i++);
				int d = Character.digit(c, 16);
				if (d < 0)
					throw err("bad hex escape");
				v = (v << 4) + d;
			}
			return v;
		}

		private String parseIdent() {
			skipWsAndComments();
			if (i >= n || !isIdentStart(s.charAt(i)))
				throw err("expected identifier");
			int start = i++;
			while (i < n && isIdentPart(s.charAt(i)))
				i++;
			return s.substring(start, i);
		}

		private void expect(String tok) {
			skipWsAndComments();
			if (!s.startsWith(tok, i))
				throw err("expected '" + tok + "'");
			i += tok.length();
		}

		private void skipWsAndComments() {
			while (i < n) {
				char c = s.charAt(i);
				if (c == '#') {
					while (i < n && s.charAt(i) != '\n')
						i++;
				} else if (Character.isWhitespace(c))
					i++;
				else
					break;
			}
		}

		private IllegalArgumentException err(String msg) {
			int line = 1, col = 1;
			for (int k = 0; k < i && k < n; k++) {
				if (s.charAt(k) == '\n') {
					line++;
					col = 1;
				} else
					col++;
			}
			return new IllegalArgumentException("GBNF:" + line + ":" + col + ": " + msg);
		}

		private static boolean isIdentStart(char c) {
			return Character.isLetter(c) || c == '_';
		}

		private static boolean isIdentPart(char c) {
			return Character.isLetterOrDigit(c) || c == '_' || c == '-';
		}
	}
}
