package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Hold-back / strip behaviour for chat turn-end markers across supported
 * LoRA templates (TinyLlama/Zephyr, Mistral, Phi-3, LLaMA-3, Gemma, ChatML/Qwen).
 */
@DisplayName("EosOutputFilter")
class EosOutputFilterTest {

	@ParameterizedTest
	@ValueSource(strings = { "</s>", "<|end|>", "<|eot_id|>", "<end_of_turn>", "<|im_end|>", "<|endoftext|>" })
	@DisplayName("exact marker piece: nothing streamed, empty text, stop")
	void exact_marker_stops_without_emit(String marker) {
		EosOutputFilter filter = new EosOutputFilter();
		EosOutputFilter.Outcome o = filter.accept(marker);
		assertThat(o.stop()).isTrue();
		assertThat(o.emit()).isEmpty();
		assertThat(filter.text()).isEmpty();
	}

	@ParameterizedTest
	@ValueSource(strings = { "</s>", "<|end|>", "<|eot_id|>", "<end_of_turn>", "<|im_end|>", "<|endoftext|>" })
	@DisplayName("answer glued to marker in one piece: answer emitted, marker stripped")
	void answer_plus_marker_in_one_piece(String marker) {
		EosOutputFilter filter = new EosOutputFilter();
		EosOutputFilter.Outcome o = filter.accept("Johnatan" + marker);
		assertThat(o.stop()).isTrue();
		assertThat(o.emit()).isEqualTo("Johnatan");
		assertThat(filter.text()).isEqualTo("Johnatan");
	}

	@ParameterizedTest
	@ValueSource(strings = { "</s>", "<|end|>", "<|eot_id|>", "<end_of_turn>", "<|im_end|>", "<|endoftext|>" })
	@DisplayName("marker with trailing newline: still stops and strips")
	void marker_with_trailing_newline(String marker) {
		EosOutputFilter filter = new EosOutputFilter();
		assertThat(filter.accept("Dima").emit()).isEqualTo("Dima");
		EosOutputFilter.Outcome o = filter.accept(marker + "\n");
		assertThat(o.stop()).isTrue();
		assertThat(o.emit()).isEmpty();
		assertThat(filter.text()).isEqualTo("Dima");
	}

	@Test
	@DisplayName("multi-token TinyLlama </s> never reaches emit")
	void multi_token_slash_s_never_emitted() {
		EosOutputFilter filter = new EosOutputFilter();
		assertThat(filter.accept("Johnatan").emit()).isEqualTo("Johnatan");
		assertThat(filter.accept("</").emit()).as("prefix held back").isEmpty();
		assertThat(filter.accept("s").emit()).isEmpty();
		EosOutputFilter.Outcome done = filter.accept(">");
		assertThat(done.stop()).isTrue();
		assertThat(done.emit()).isEmpty();
		assertThat(filter.text()).isEqualTo("Johnatan");
	}

	@Test
	@DisplayName("multi-token ChatML im_end never reaches emit")
	void multi_token_im_end_never_emitted() {
		EosOutputFilter filter = new EosOutputFilter();
		filter.accept("ok");
		assertThat(filter.accept("<|").emit()).isEmpty();
		assertThat(filter.accept("im_end").emit()).isEmpty();
		EosOutputFilter.Outcome done = filter.accept("|>");
		assertThat(done.stop()).isTrue();
		assertThat(done.emit()).isEmpty();
		assertThat(filter.text()).isEqualTo("ok");
	}

	@Test
	@DisplayName("angle-bracket math that is not an EOS prefix is emitted")
	void non_eos_angle_brackets_pass() {
		EosOutputFilter filter = new EosOutputFilter();
		EosOutputFilter.Outcome o = filter.accept("3<x<7");
		assertThat(o.stop()).isFalse();
		assertThat(o.emit()).isEqualTo("3<x<7");
		assertThat(filter.finish("").emit()).isEmpty();
	}

	@Test
	@DisplayName("held '<' released when continuation is not an EOS marker")
	void held_prefix_released_when_not_marker() {
		EosOutputFilter filter = new EosOutputFilter();
		assertThat(filter.accept("a ").emit()).isEqualTo("a ");
		assertThat(filter.accept("<").emit()).isEmpty();
		EosOutputFilter.Outcome o = filter.accept(" b");
		assertThat(o.stop()).isFalse();
		assertThat(o.emit()).isEqualTo("< b");
		assertThat(filter.text()).isEqualTo("a < b");
	}

	@Test
	@DisplayName("finish emits held-back text when generation hits max tokens")
	void finish_emits_held_prefix_without_eos() {
		EosOutputFilter filter = new EosOutputFilter();
		filter.accept("hi");
		filter.accept("<");
		EosOutputFilter.Outcome o = filter.finish("");
		assertThat(o.stop()).isFalse();
		assertThat(o.emit()).isEqualTo("<");
		assertThat(filter.text()).isEqualTo("hi<");
	}

	@Test
	@DisplayName("discardHeld drops unfinished marker prefix before real EOS token id")
	void discard_held_after_partial_marker() {
		EosOutputFilter filter = new EosOutputFilter();
		assertThat(filter.accept("hi").emit()).isEqualTo("hi");
		assertThat(filter.accept("</").emit()).isEmpty();
		filter.discardHeld();
		EosOutputFilter.Outcome o = filter.finish("");
		assertThat(o.emit()).isEmpty();
		assertThat(filter.text()).isEqualTo("hi");
	}
}
