package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;

import org.junit.jupiter.api.Test;

class ToolCallParserTest {

	@Test
	void parses_single_and_multiple_tool_call_tags() {
		String text = """
				<tool_call>
				{"name": "get_weather", "arguments": {"city": "Boston"}}
				</tool_call>
				<tool_call>
				{"name": "get_time", "parameters": {"tz": "ET"}}
				</tool_call>
				""";
		List<ToolCallParser.ParsedToolCall> calls = ToolCallParser.parse(text);
		assertThat(calls).hasSize(2);
		assertThat(calls.get(0).name()).isEqualTo("get_weather");
		assertThat(calls.get(0).argumentsJson()).contains("Boston");
		assertThat(calls.get(1).name()).isEqualTo("get_time");
		assertThat(calls.get(1).argumentsJson()).contains("ET");
	}

	@Test
	void parses_raw_json_object_and_fenced_block() {
		List<ToolCallParser.ParsedToolCall> raw = ToolCallParser
				.parse("{\"name\":\"echo\",\"arguments\":\"{\\\"x\\\":1}\"}");
		assertThat(raw).hasSize(1);
		assertThat(raw.get(0).name()).isEqualTo("echo");
		assertThat(raw.get(0).argumentsJson()).isEqualTo("{\"x\":1}");

		List<ToolCallParser.ParsedToolCall> fenced = ToolCallParser.parse("""
				```json
				{"name":"echo","arguments":{"x":2}}
				```
				""");
		assertThat(fenced).hasSize(1);
		assertThat(fenced.get(0).argumentsJson()).contains("\"x\":2");
	}

	@Test
	void malformed_and_plain_text_yield_empty() {
		assertThat(ToolCallParser.parse("hello")).isEmpty();
		assertThat(ToolCallParser.parse("<tool_call>not-json</tool_call>")).isEmpty();
		assertThat(ToolCallParser.parse("{\"foo\":1}")).isEmpty();
		assertThat(ToolCallParser.parse(null)).isEmpty();
	}
}
