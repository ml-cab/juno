package cab.ml.juno.tokenizer;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;

import org.junit.jupiter.api.Test;

import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.ChatTemplateFormatter;

class ChatTemplateFormatterTest {

	@Test
	void formats_messages_using_resolved_template() {
		ChatTemplateFormatter formatter = ChatTemplateFormatter.forModelType("llama3");
		List<ChatMessage> messages = List.of(ChatMessage.user("Ping"));

		String prompt = formatter.format(messages);

		assertThat(prompt).contains("Ping");
		assertThat(prompt).contains("<|begin_of_text|>");
		assertThat(formatter.modelType()).isEqualTo("llama3");
	}

	@Test
	void rejects_null_messages() {
		ChatTemplateFormatter formatter = ChatTemplateFormatter.forModelType("chatml");
		assertThatThrownBy(() -> formatter.format(null)).isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	void rejects_empty_message_list() {
		ChatTemplateFormatter formatter = ChatTemplateFormatter.forModelType("chatml");
		assertThatThrownBy(() -> formatter.format(List.of())).isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	void unknown_model_type_still_produces_output() {
		ChatTemplateFormatter formatter = ChatTemplateFormatter.forModelType("whatever");
		String prompt = formatter.format(List.of(ChatMessage.user("hello")));
		assertThat(prompt).isNotBlank();
		assertThat(formatter.modelType()).isEqualTo("chatml"); // fallback
	}
}
