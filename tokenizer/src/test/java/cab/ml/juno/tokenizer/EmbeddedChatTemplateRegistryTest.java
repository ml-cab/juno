package cab.ml.juno.tokenizer;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

class EmbeddedChatTemplateRegistryTest {

	@AfterEach
	void clearRegistry() {
		EmbeddedChatTemplateRegistry.clear();
	}

	@Test
	void lookup_returns_null_when_nothing_registered() {
		assertThat(EmbeddedChatTemplateRegistry.lookup("some-key")).isNull();
	}

	@Test
	void register_then_lookup_returns_the_same_template() {
		ChatTemplate marker = ChatTemplate.gemma();
		EmbeddedChatTemplateRegistry.register("my-model", marker);

		assertThat(EmbeddedChatTemplateRegistry.lookup("my-model")).isSameAs(marker);
	}

	@Test
	void clear_removes_all_registrations() {
		EmbeddedChatTemplateRegistry.register("a", ChatTemplate.llama3());
		EmbeddedChatTemplateRegistry.clear();

		assertThat(EmbeddedChatTemplateRegistry.lookup("a")).isNull();
	}

	@Test
	void register_ignores_null_key_or_template() {
		EmbeddedChatTemplateRegistry.register(null, ChatTemplate.chatml());
		EmbeddedChatTemplateRegistry.register("k", null);

		assertThat(EmbeddedChatTemplateRegistry.lookup("k")).isNull();
	}

	@Test
	void chatTemplateFormatter_forModelType_prefers_registry_override() {
		ChatTemplate marker = new ChatTemplate() {
			@Override
			public String format(List<ChatMessage> messages) {
				return "MARKER-OUTPUT";
			}

			@Override
			public String modelType() {
				return "tinyllama";
			}
		};
		EmbeddedChatTemplateRegistry.register("tinyllama", marker);

		ChatTemplateFormatter formatter = ChatTemplateFormatter.forModelType("tinyllama");
		assertThat(formatter.format(List.of(ChatMessage.user("hi")))).isEqualTo("MARKER-OUTPUT");
	}

	@Test
	void chatTemplateFormatter_forModelType_falls_back_to_named_when_not_registered() {
		ChatTemplateFormatter formatter = ChatTemplateFormatter.forModelType("tinyllama");
		assertThat(formatter.format(List.of(ChatMessage.user("hi")))).contains("<|user|>");
	}
}
