package cab.ml.juno.tokenizer;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;

import org.junit.jupiter.api.Test;

import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.ChatTemplate;

class ChatTemplateTest {

	private final List<ChatMessage> messages = List.of(ChatMessage.system("You are helpful."),
			ChatMessage.user("Hello!"), ChatMessage.assistant("Hi there!"));

	@Test
	void llama3_contains_correct_special_tokens() {
		String prompt = ChatTemplate.llama3().format(messages);

		assertThat(prompt).startsWith("<|begin_of_text|>");
		assertThat(prompt).contains("<|start_header_id|>system<|end_header_id|>");
		assertThat(prompt).contains("<|start_header_id|>user<|end_header_id|>");
		assertThat(prompt).contains("<|eot_id|>");
		assertThat(prompt).endsWith("<|start_header_id|>assistant<|end_header_id|>\n\n");
	}

	@Test
	void mistral_wraps_user_in_inst_tags() {
		String prompt = ChatTemplate.mistral().format(messages);

		assertThat(prompt).contains("[INST]");
		assertThat(prompt).contains("[/INST]");
		// System message should be prepended into first user turn
		assertThat(prompt).contains("You are helpful.");
	}

	@Test
	void gemma_uses_start_of_turn_tokens() {
		String prompt = ChatTemplate.gemma().format(messages);

		assertThat(prompt).contains("<start_of_turn>user");
		assertThat(prompt).contains("<start_of_turn>model");
		assertThat(prompt).contains("<end_of_turn>");
		assertThat(prompt).endsWith("<start_of_turn>model\n");
	}

	@Test
	void chatml_uses_im_start_end_tokens() {
		String prompt = ChatTemplate.chatml().format(messages);

		assertThat(prompt).contains("<|im_start|>system");
		assertThat(prompt).contains("<|im_start|>user");
		assertThat(prompt).contains("<|im_end|>");
		assertThat(prompt).endsWith("<|im_start|>assistant\n");
	}

	@Test
	void all_templates_include_message_content() {
		for (ChatTemplate t : ChatTemplate.BUILT_IN.values()) {
			String prompt = t.format(messages);
			assertThat(prompt).as("Template %s should contain user message", t.modelType()).contains("Hello!");
		}
	}

	@Test
	void tinyllama_uses_zephyr_format() {
		String prompt = ChatTemplate.tinyllama().format(messages);

		// Zephyr turn delimiters
		assertThat(prompt).contains("<|system|>\n");
		assertThat(prompt).contains("<|user|>\n");
		assertThat(prompt).contains("<|assistant|>\n");
		// EOS marker after every turn
		assertThat(prompt).contains("</s>\n");
		// Must NOT contain ChatML tokens — that was the original bug
		assertThat(prompt).doesNotContain("<|im_start|>");
		assertThat(prompt).doesNotContain("<|im_end|>");
		// Prompt ends with the assistant turn opener (no content yet)
		assertThat(prompt).endsWith("<|assistant|>\n");
	}

	@Test
	void tinyllama_single_user_turn() {
		String prompt = ChatTemplate.tinyllama().format(List.of(ChatMessage.user("Hello!")));

		assertThat(prompt).isEqualTo("<|user|>\nHello!</s>\n<|assistant|>\n");
	}

	@Test
	void zephyr_alias_resolves_to_same_format_as_tinyllama() {
		List<ChatMessage> msgs = List.of(ChatMessage.user("Ping"));
		String viaZephyr = ChatTemplate.forModelType("zephyr").format(msgs);
		String viaTinyllama = ChatTemplate.forModelType("tinyllama").format(msgs);

		assertThat(viaZephyr).isEqualTo(viaTinyllama);
	}

	@Test
	void forModelType_falls_back_to_chatml_for_unknown() {
		ChatTemplate t = ChatTemplate.forModelType("some-unknown-model");
		assertThat(t.modelType()).isEqualTo("chatml");
	}

	@Test
	void forModelType_is_case_insensitive() {
		assertThat(ChatTemplate.forModelType("LLaMA3").modelType()).isEqualTo("llama3");
		assertThat(ChatTemplate.forModelType("MISTRAL").modelType()).isEqualTo("mistral");
		assertThat(ChatTemplate.forModelType("TinyLlama").modelType()).isEqualTo("tinyllama");
		assertThat(ChatTemplate.forModelType("ZEPHYR").modelType()).isEqualTo("tinyllama");
	}

	@Test
	void qwen_uses_chatml_template() {
		ChatTemplate t = ChatTemplate.forModelType("qwen");

		assertThat(t.modelType()).isEqualTo("chatml");
	}
}