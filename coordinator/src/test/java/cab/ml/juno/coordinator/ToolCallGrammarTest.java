package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.charset.StandardCharsets;
import java.util.List;

import org.junit.jupiter.api.Test;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import cab.ml.juno.sampler.GbnfGrammar;

class ToolCallGrammarTest {

	private static final ObjectMapper JSON = new ObjectMapper();

	@Test
	void required_envelope_accepts_tagged_call() {
		ObjectNode params = JSON.createObjectNode().put("type", "object");
		params.putObject("properties").putObject("city").put("type", "string");
		params.putArray("required").add("city");
		OpenAiTools.FunctionTool tool = new OpenAiTools.FunctionTool("get_weather", "Weather", params);
		GbnfGrammar g = ToolCallGrammar.compile(List.of(tool), null);
		String ok = "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Boston\"}}</tool_call>";
		assertThat(g.isComplete(ok.getBytes(StandardCharsets.UTF_8))).isTrue();
		assertThat(g.acceptsPrefix("hello".getBytes(StandardCharsets.UTF_8))).isFalse();
	}

	@Test
	void named_choice_rejects_other_function() {
		OpenAiTools.FunctionTool a = new OpenAiTools.FunctionTool("a", "", JSON.createObjectNode().put("type", "object"));
		OpenAiTools.FunctionTool b = new OpenAiTools.FunctionTool("b", "", JSON.createObjectNode().put("type", "object"));
		GbnfGrammar g = ToolCallGrammar.compile(List.of(a, b), "a");
		String ok = "<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call>";
		String bad = "<tool_call>{\"name\":\"b\",\"arguments\":{}}</tool_call>";
		assertThat(g.isComplete(ok.getBytes(StandardCharsets.UTF_8))).isTrue();
		assertThat(g.isComplete(bad.getBytes(StandardCharsets.UTF_8))).isFalse();
	}

	@Test
	void empty_tools_fail_closed() {
		assertThatThrownBy(() -> ToolCallGrammar.compile(List.of(), null))
				.isInstanceOf(IllegalArgumentException.class);
	}
}
