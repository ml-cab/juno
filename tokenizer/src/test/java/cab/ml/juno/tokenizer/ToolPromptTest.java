package cab.ml.juno.tokenizer;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;

import org.junit.jupiter.api.Test;

class ToolPromptTest {

	private static final String TOOLS_JSON = "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\"}}]";

	@Test
	void llama3_and_chatml_prompts_contain_tool_name() {
		List<ChatMessage> injected = ToolPrompt.inject(List.of(ChatMessage.user("Weather in Boston?")), TOOLS_JSON,
				false, null);
		String llama = ChatTemplate.llama3().format(injected);
		String chatml = ChatTemplate.chatml().format(injected);
		assertThat(llama).contains("get_weather").contains("<tool_call>").contains("<|start_header_id|>system");
		assertThat(chatml).contains("get_weather").contains("<tool_call>").contains("<|im_start|>system");
	}

	@Test
	void required_instructions_forbid_plain_text() {
		String text = ToolPrompt.instructions(TOOLS_JSON, true, null);
		assertThat(text).contains("You must call a tool");
		assertThat(ToolPrompt.instructions(TOOLS_JSON, true, "get_weather")).contains("get_weather");
	}

	@Test
	void merges_into_existing_system_message() {
		List<ChatMessage> injected = ToolPrompt.inject(
				List.of(ChatMessage.system("Be terse."), ChatMessage.user("Hi")), TOOLS_JSON, false, null);
		assertThat(injected).hasSize(2);
		assertThat(injected.get(0).content()).contains("Be terse.").contains("get_weather");
	}

	@Test
	void unsupported_templates_fail_closed() {
		assertThatThrownBy(() -> ToolPrompt.requireSupported("mistral-7b-instruct"))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("mistral");
		assertThatThrownBy(() -> ToolPrompt.requireSupported("Phi-3.5-mini-instruct"))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("phi3");
		assertThatThrownBy(() -> ToolPrompt.requireSupported("tinyllama-1.1b-chat"))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("tinyllama");
		ToolPrompt.requireSupported("llama3-8b");
		ToolPrompt.requireSupported("qwen2.5-3b-instruct");
		ToolPrompt.requireSupported("qwen3-4b");
	}

	@Test
	void chatml_and_llama3_emit_tool_role() {
		List<ChatMessage> turns = List.of(ChatMessage.user("Weather?"),
				ChatMessage.assistant(ToolPrompt.formatAssistantToolCalls(List.of("{\"name\":\"get_weather\"}"))),
				ChatMessage.tool(ToolPrompt.formatToolResult("72F")));
		assertThat(ChatTemplate.chatml().format(turns)).contains("<|im_start|>tool").contains("72F");
		assertThat(ChatTemplate.llama3().format(turns)).contains("<|start_header_id|>tool").contains("72F");
	}
}
