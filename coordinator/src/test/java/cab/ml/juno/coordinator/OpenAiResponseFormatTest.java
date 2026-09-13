package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.charset.StandardCharsets;

import org.junit.jupiter.api.Test;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

class OpenAiResponseFormatTest {

	private static final ObjectMapper JSON = new ObjectMapper();

	@Test
	void text_and_absent_are_unconstrained() {
		assertThat(OpenAiResponseFormat.compile(null, null)).isNull();
		ObjectNode text = JSON.createObjectNode().put("type", "text");
		assertThat(OpenAiResponseFormat.compile(text, null)).isNull();
		assertThat(OpenAiResponseFormat.validate(text)).isNull();
	}

	@Test
	void json_object_compiles() {
		ObjectNode json = JSON.createObjectNode().put("type", "json_object");
		assertThat(OpenAiResponseFormat.validate(json)).isNull();
		assertThat(OpenAiResponseFormat.compile(json, null).isComplete("{\"a\":1}".getBytes(StandardCharsets.UTF_8)))
				.isTrue();
	}

	@Test
	void json_schema_compiles_and_rejects_unsupported() {
		ObjectNode schema = JSON.createObjectNode().put("type", "object");
		schema.putObject("properties").putObject("ok").put("type", "boolean");
		schema.putArray("required").add("ok");
		ObjectNode rf = JSON.createObjectNode().put("type", "json_schema");
		rf.putObject("json_schema").put("name", "ok").set("schema", schema);
		assertThat(OpenAiResponseFormat.compile(rf, null).isComplete("{\"ok\":true}".getBytes(StandardCharsets.UTF_8)))
				.isTrue();

		ObjectNode bad = JSON.createObjectNode().put("type", "json_schema");
		bad.putObject("json_schema").putObject("schema").put("type", "string").put("pattern", "^a$");
		assertThatThrownBy(() -> OpenAiResponseFormat.compile(bad, null)).isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("pattern");
	}

	@Test
	void grammar_conflicts_with_json_types() {
		ObjectNode json = JSON.createObjectNode().put("type", "json_object");
		assertThatThrownBy(() -> OpenAiResponseFormat.compile(json, "root ::= \"a\""))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("x_juno_grammar");
	}

	@Test
	void fallback_applies_when_request_unconstrained() {
		var fallback = cab.ml.juno.sampler.GbnfGrammar.parse("root ::= \"yes\" | \"no\"");
		assertThat(OpenAiResponseFormat.compile(null, null, fallback)).isSameAs(fallback);
		ObjectNode text = JSON.createObjectNode().put("type", "text");
		assertThat(OpenAiResponseFormat.compile(text, null, fallback)).isNull();
		ObjectNode json = JSON.createObjectNode().put("type", "json_object");
		assertThat(OpenAiResponseFormat.compile(json, null, fallback)).isNotSameAs(fallback);
	}

	@Test
	void unknown_type_fails_closed() {
		ObjectNode rf = JSON.createObjectNode().put("type", "xml");
		assertThat(OpenAiResponseFormat.validate(rf)).contains("not supported");
	}
}
