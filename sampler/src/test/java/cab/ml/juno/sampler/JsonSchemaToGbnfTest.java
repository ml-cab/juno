package cab.ml.juno.sampler;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.charset.StandardCharsets;

import org.junit.jupiter.api.Test;

class JsonSchemaToGbnfTest {

	@Test
	void object_string_integer_required() {
		GbnfGrammar g = JsonSchemaToGbnf.compileGrammar("""
				{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name"]}
				""");
		assertThat(g.isComplete(utf8("{\"name\":\"Ada\"}"))).isTrue();
		assertThat(g.isComplete(utf8("{\"name\":\"Ada\",\"age\":36}"))).isTrue();
		assertThat(g.acceptsPrefix(utf8("{\"age\":1}"))).isFalse();
	}

	@Test
	void enum_and_array() {
		GbnfGrammar g = JsonSchemaToGbnf.compileGrammar("""
				{"type":"object","properties":{
				  "tone":{"enum":["low","high"]},
				  "vals":{"type":"array","items":{"type":"integer"}}
				},"required":["tone"]}
				""");
		assertThat(g.isComplete(utf8("{\"tone\":\"low\"}"))).isTrue();
		assertThat(g.isComplete(utf8("{\"tone\":\"high\",\"vals\":[1,2]}"))).isTrue();
		assertThat(g.acceptsPrefix(utf8("{\"tone\":\"mid\"}"))).isFalse();
	}

	@Test
	void rejects_ref_and_pattern() {
		assertThatThrownBy(() -> JsonSchemaToGbnf.compile("{\"$ref\":\"#/defs/x\"}"))
				.isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("$ref");
		assertThatThrownBy(() -> JsonSchemaToGbnf.compile("""
				{"type":"string","pattern":"^[a-z]+$"}
				""")).isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("pattern");
	}

	@Test
	void json_object_grammar_accepts_object() {
		GbnfGrammar g = JsonSchemaToGbnf.jsonObjectGrammar();
		assertThat(g.isComplete(utf8("{\"a\":1}"))).isTrue();
		assertThat(g.isComplete(utf8("[1]"))).isFalse();
	}

	private static byte[] utf8(String s) {
		return s.getBytes(StandardCharsets.UTF_8);
	}
}
