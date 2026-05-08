package cab.ml.juno.player;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class ChatTrainingFormatsTest {

	@Test
	void tinyllama_qa_includes_assistant_segment() {
		String s = ChatTrainingFormats.qaTurn("What?", "Because.", "tinyllama");
		assertThat(s).contains("<|assistant|>");
		assertThat(s).contains("Because.");
	}
}
