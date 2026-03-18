package cab.ml.juno.tokenizer;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

class ChatMessageTest {

	@Test
	void factory_methods_set_correct_roles() {
		assertThat(ChatMessage.system("s").role()).isEqualTo("system");
		assertThat(ChatMessage.user("u").role()).isEqualTo("user");
		assertThat(ChatMessage.assistant("a").role()).isEqualTo("assistant");
	}

	@Test
	void role_is_normalised_to_lowercase() {
		ChatMessage msg = new ChatMessage("USER", "hello");
		assertThat(msg.role()).isEqualTo("user");
	}

	@Test
	void role_predicates_are_exclusive() {
		ChatMessage u = ChatMessage.user("hi");
		assertThat(u.isUser()).isTrue();
		assertThat(u.isSystem()).isFalse();
		assertThat(u.isAssistant()).isFalse();
	}

	@Test
	void rejects_blank_role() {
		assertThatThrownBy(() -> new ChatMessage("  ", "content")).isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	void rejects_null_content() {
		assertThatThrownBy(() -> new ChatMessage("user", null)).isInstanceOf(IllegalArgumentException.class);
	}
}
