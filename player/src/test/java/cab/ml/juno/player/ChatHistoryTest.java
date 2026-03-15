package cab.ml.juno.player;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.player.ChatHistory;
import cab.ml.juno.tokenizer.ChatMessage;

/**
 * Unit tests for ChatHistory — conversation history for the REPL.
 */
class ChatHistoryTest {

	@Test
	@DisplayName("new history is empty")
	void emptyInitially() {
		ChatHistory history = new ChatHistory();
		assertThat(history.getMessages()).isEmpty();
	}

	@Test
	@DisplayName("addUser and addAssistant accumulate in order")
	void accumulatesTurns() {
		ChatHistory history = new ChatHistory();
		history.addUser("Hello");
		history.addAssistant("Hi there!");
		history.addUser("What is Java?");
		List<ChatMessage> messages = history.getMessages();
		assertThat(messages).hasSize(3);
		assertThat(messages.get(0)).isEqualTo(ChatMessage.user("Hello"));
		assertThat(messages.get(1)).isEqualTo(ChatMessage.assistant("Hi there!"));
		assertThat(messages.get(2)).isEqualTo(ChatMessage.user("What is Java?"));
	}

	@Test
	@DisplayName("getMessages returns a copy")
	void getMessagesIsCopy() {
		ChatHistory history = new ChatHistory();
		history.addUser("a");
		List<ChatMessage> first = history.getMessages();
		List<ChatMessage> second = history.getMessages();
		assertThat(first).isNotSameAs(second);
		assertThat(first).isEqualTo(second);
	}
}
