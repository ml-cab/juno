package cab.ml.juno.player;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for ChatModelType — model type derivation from GGUF path.
 */
class ChatModelTypeTest {

	@Test
	@DisplayName("null path returns chatml")
	void nullPath() {
		assertThat(ChatModelType.fromPath(null)).isEqualTo("chatml");
	}

	@Test
	@DisplayName("path containing tinyllama returns tinyllama")
	void tinyllama() {
		assertThat(ChatModelType.fromPath("/models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf")).isEqualTo("tinyllama");
		assertThat(ChatModelType.fromPath("C:\\models\\tinyllama.gguf")).isEqualTo("tinyllama");
	}

	@Test
	@DisplayName("path containing zephyr returns tinyllama")
	void zephyr() {
		assertThat(ChatModelType.fromPath("/data/zephyr-7b.gguf")).isEqualTo("tinyllama");
	}

	@Test
	@DisplayName("path containing llama3 returns llama3")
	void llama3() {
		assertThat(ChatModelType.fromPath("/models/llama-3-8b.gguf")).isEqualTo("llama3");
		assertThat(ChatModelType.fromPath("/models/Llama3-8B.gguf")).isEqualTo("llama3");
	}

	@Test
	@DisplayName("path containing mistral or gemma returns respective type")
	void mistralAndGemma() {
		assertThat(ChatModelType.fromPath("/models/mistral-7b.gguf")).isEqualTo("mistral");
		assertThat(ChatModelType.fromPath("/models/gemma-2b.gguf")).isEqualTo("gemma");
	}

	@Test
	@DisplayName("path containing phi-3 or phi3 returns phi3")
	void phi3() {
		assertThat(ChatModelType.fromPath("../models/phi-3.5-mini-instruct.Q4_K_M.gguf")).isEqualTo("phi3");
		assertThat(ChatModelType.fromPath("/models/Phi3-mini.gguf")).isEqualTo("phi3");
		assertThat(ChatModelType.fromPath("/models/phi_3-medium.gguf")).isEqualTo("phi3");
	}

	@Test
	@DisplayName("unknown path returns chatml")
	void unknownPath() {
		assertThat(ChatModelType.fromPath("/models/foo.gguf")).isEqualTo("chatml");
	}
}