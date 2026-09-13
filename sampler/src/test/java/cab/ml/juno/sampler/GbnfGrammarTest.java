package cab.ml.juno.sampler;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.charset.StandardCharsets;

import org.junit.jupiter.api.Test;

class GbnfGrammarTest {

	@Test
	void literal_prefix_and_complete() {
		GbnfGrammar g = GbnfGrammar.parse("root ::= \"ab\"");
		assertThat(g.acceptsPrefix(bytes(""))).isTrue();
		assertThat(g.isComplete(bytes(""))).isFalse();
		assertThat(g.acceptsPrefix(bytes("a"))).isTrue();
		assertThat(g.isComplete(bytes("a"))).isFalse();
		assertThat(g.acceptsPrefix(bytes("ab"))).isTrue();
		assertThat(g.isComplete(bytes("ab"))).isTrue();
		assertThat(g.acceptsPrefix(bytes("abc"))).isFalse();
		assertThat(g.acceptsPrefix(bytes("x"))).isFalse();
	}

	@Test
	void alternatives_and_repetition() {
		GbnfGrammar g = GbnfGrammar.parse("""
				root ::= ("a" | "bb")+
				""");
		assertThat(g.isComplete(bytes("a"))).isTrue();
		assertThat(g.isComplete(bytes("bb"))).isTrue();
		assertThat(g.isComplete(bytes("abb"))).isTrue();
		assertThat(g.acceptsPrefix(bytes("b"))).isTrue();
		assertThat(g.isComplete(bytes("b"))).isFalse();
		assertThat(g.acceptsPrefix(bytes("c"))).isFalse();
	}

	@Test
	void char_class_and_star() {
		GbnfGrammar g = GbnfGrammar.parse("""
				root ::= [ab]* "c"
				""");
		assertThat(g.isComplete(bytes("c"))).isTrue();
		assertThat(g.isComplete(bytes("aac"))).isTrue();
		assertThat(g.acceptsPrefix(bytes("aa"))).isTrue();
		assertThat(g.acceptsPrefix(bytes("x"))).isFalse();
	}

	@Test
	void char_class_ending_with_range() {
		GbnfGrammar digits = GbnfGrammar.parse("root ::= [0-9]+");
		assertThat(digits.isComplete(bytes("0"))).isTrue();
		assertThat(digits.isComplete(bytes("42"))).isTrue();
		assertThat(digits.acceptsPrefix(bytes("a"))).isFalse();
		GbnfGrammar hex = GbnfGrammar.parse("root ::= [0-9a-fA-F]+");
		assertThat(hex.isComplete(bytes("aF0"))).isTrue();
		GbnfGrammar alnum = GbnfGrammar.parse("root ::= [a-zA-Z0-9]+");
		assertThat(alnum.isComplete(bytes("Ada9"))).isTrue();
	}

	@Test
	void comments_and_named_rules() {
		GbnfGrammar g = GbnfGrammar.parse("""
				# greeting
				root ::= hi
				hi ::= "hi"
				""");
		assertThat(g.isComplete(bytes("hi"))).isTrue();
	}

	@Test
	void rejects_missing_root() {
		assertThatThrownBy(() -> GbnfGrammar.parse("foo ::= \"a\""))
				.isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("root");
	}

	@Test
	void rejects_invalid_syntax() {
		assertThatThrownBy(() -> GbnfGrammar.parse("root ::= ("))
				.isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> GbnfGrammar.parse("root ::= \"unterminated"))
				.isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	void next_bytes_for_literal() {
		GbnfGrammar g = GbnfGrammar.parse("root ::= \"ab\"");
		boolean[] start = g.nextBytes(bytes(""));
		assertThat(start['a']).isTrue();
		assertThat(start['b']).isFalse();
		boolean[] afterA = g.nextBytes(bytes("a"));
		assertThat(afterA['b']).isTrue();
		assertThat(afterA['a']).isFalse();
	}

	private static byte[] bytes(String s) {
		return s.getBytes(StandardCharsets.UTF_8);
	}
}
