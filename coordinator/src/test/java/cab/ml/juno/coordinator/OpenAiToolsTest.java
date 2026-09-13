package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import cab.ml.juno.tokenizer.ChatMessage;

class OpenAiToolsTest {

	private static final ObjectMapper JSON = new ObjectMapper();

	@Test
	void parse_tools_and_choice() throws Exception {
		ObjectNode body = JSON.readValue("""
				{
				  "tools": [{
				    "type": "function",
				    "function": {
				      "name": "get_weather",
				      "description": "Weather",
				      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
				    }
				  }],
				  "tool_choice": "required"
				}
				""", ObjectNode.class);
		OpenAiTools.ParsedRequest req = OpenAiTools.parse(body.get("tools"), body.get("tool_choice"));
		assertThat(req.active()).isTrue();
		assertThat(req.choice().requiresCall()).isTrue();
		assertThat(req.tools()).extracting(OpenAiTools.FunctionTool::name).containsExactly("get_weather");
	}

	@Test
	void none_never_active_named_must_exist() throws Exception {
		ArrayNode tools = JSON.createArrayNode();
		ObjectNode item = tools.addObject();
		item.put("type", "function");
		item.putObject("function").put("name", "echo");
		OpenAiTools.ParsedRequest none = OpenAiTools.parse(tools, JSON.getNodeFactory().textNode("none"));
		assertThat(none.active()).isFalse();
		assertThatThrownBy(() -> OpenAiTools.parse(tools, JSON.readTree("{\"type\":\"function\",\"function\":{\"name\":\"missing\"}}")))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("missing");
		assertThatThrownBy(() -> OpenAiTools.parse(null, JSON.getNodeFactory().textNode("required")))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("tools must be set");
	}

	@Test
	void bind_prompt_llama3_and_reject_phi3() {
		ArrayNode tools = JSON.createArrayNode();
		tools.addObject().put("type", "function").putObject("function").put("name", "echo");
		OpenAiTools.ParsedRequest req = OpenAiTools.parse(tools, JSON.getNodeFactory().textNode("auto"));
		List<ChatMessage> bound = OpenAiTools.bindPrompt(List.of(ChatMessage.user("Hi")), req, "llama3-8b");
		assertThat(bound.get(0).isSystem()).isTrue();
		assertThat(bound.get(0).content()).contains("echo");
		assertThatThrownBy(() -> OpenAiTools.bindPrompt(List.of(ChatMessage.user("Hi")), req, "phi3-mini"))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("phi3");
	}

	@Test
	void tool_choice_none_skips_bind() {
		ArrayNode tools = JSON.createArrayNode();
		tools.addObject().put("type", "function").putObject("function").put("name", "echo");
		OpenAiTools.ParsedRequest none = OpenAiTools.parse(tools, JSON.getNodeFactory().textNode("none"));
		List<ChatMessage> msgs = List.of(ChatMessage.user("Hi"));
		assertThat(OpenAiTools.bindPrompt(msgs, none, "llama3-8b")).isSameAs(msgs);
		assertThat(OpenAiTools.grammar(none)).isNull();
	}

	@Test
	void round_trip_tool_messages_and_response_shape() throws Exception {
		ChatMessage assistant = OpenAiTools.toChatMessage("assistant", null, JSON.readTree("""
				[{"id":"call_1","type":"function","function":{"name":"echo","arguments":"{\\"x\\":1}"}}]
				"""), null);
		assertThat(assistant.content()).contains("<tool_call>").contains("echo");
		ChatMessage tool = OpenAiTools.toChatMessage("tool", JSON.getNodeFactory().textNode("ok"), null, "call_1");
		assertThat(tool.isTool()).isTrue();
		assertThat(tool.content()).contains("ok");

		List<Map<String, Object>> calls = OpenAiTools.toOpenAiToolCalls("550e8400-e29b-41d4-a716-446655440000",
				ToolCallParser.parse(assistant.content()));
		assertThat(calls).hasSize(1);
		assertThat(calls.get(0).get("id").toString()).startsWith("call_");
		Map<String, Object> message = OpenAiTools.assistantMessage("ignored", calls);
		assertThat(message.get("content")).isNull();
		assertThat(message.get("tool_calls")).isEqualTo(calls);
		assertThat(OpenAiTools.finishReason(GenerationResult.StopReason.EOS_TOKEN, true)).isEqualTo("tool_calls");
		assertThat(OpenAiTools.finishReason(GenerationResult.StopReason.EOS_TOKEN, false)).isEqualTo("stop");
	}

	@Test
	void grammar_conflicts_fail_closed() {
		ArrayNode tools = JSON.createArrayNode();
		tools.addObject().put("type", "function").putObject("function").put("name", "echo");
		OpenAiTools.ParsedRequest req = OpenAiTools.parse(tools, JSON.getNodeFactory().textNode("auto"));
		ObjectNode jsonObject = JSON.createObjectNode().put("type", "json_object");
		assertThatThrownBy(() -> OpenAiTools.rejectGrammarConflict(req, jsonObject, null, false))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("response_format");
		assertThatThrownBy(() -> OpenAiTools.rejectGrammarConflict(req, null, "root ::= \"a\"", false))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("x_juno_grammar");
		assertThatThrownBy(() -> OpenAiTools.rejectGrammarConflict(req, null, null, true))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("grammar-file");
		OpenAiTools.rejectGrammarConflict(req, JSON.createObjectNode().put("type", "text"), null, false);
	}

	@Test
	void unsupported_role_rejected() {
		assertThatThrownBy(() -> OpenAiTools.toChatMessage("developer", JSON.getNodeFactory().textNode("x"), null, null))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("role");
	}
}
