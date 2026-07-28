package cab.ml.juno.player;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class ChatTrainingFormatsTest {

	@Test
	void tinyllama_qa_includes_assistant_segment() {
		String s = ChatTrainingFormats.qaTurn("What?", "Because.", "tinyllama");
		assertThat(s).contains("<|assistant|>");
		assertThat(s).contains("Because.");
		assertThat(s).isEqualTo(ChatTrainingFormats.qaPrefix("What?", "tinyllama")
				+ ChatTrainingFormats.qaCompletion("Because.", "tinyllama"));
	}

	@Test
	void qwen3_qa_includes_empty_think_block() {
		String prefix = ChatTrainingFormats.qaPrefix("Hello?", "qwen3");
		assertThat(prefix).contains("<|im_start|>user");
		assertThat(prefix).contains("<think>");
		assertThat(prefix).contains("</think>");
		assertThat(prefix).endsWith("<think>\n\n</think>\n\n");
		String turn = ChatTrainingFormats.qaTurn("Hello?", "World.", "qwen3");
		assertThat(turn).isEqualTo(prefix + ChatTrainingFormats.qaCompletion("World.", "qwen3"));
	}
}
