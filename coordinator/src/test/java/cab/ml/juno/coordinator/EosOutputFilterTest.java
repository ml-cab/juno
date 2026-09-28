package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Hold-back / strip behaviour for chat turn markers — both the turn-end markers
 * and the turn-opening role headers — across supported
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

	@ParameterizedTest
	@ValueSource(strings = { "<|user|>", "<|assistant|>", "<|system|>", "<|im_start|>", "<|start_header_id|>",
			"<start_of_turn>" })
	@DisplayName("role header after the answer: answer emitted, header and everything after it dropped")
	void role_header_ends_the_turn(String header) {
		EosOutputFilter filter = new EosOutputFilter();
		EosOutputFilter.Outcome o = filter.accept("Paris." + header);
		assertThat(o.stop()).isTrue();
		assertThat(o.emit()).isEqualTo("Paris.");
		assertThat(filter.text()).isEqualTo("Paris.");
	}

	@Test
	@DisplayName("TinyLlama role-play leak: the invented user turn is never streamed")
	void tinyllama_role_play_leak_is_truncated() {
		EosOutputFilter filter = new EosOutputFilter();
		assertThat(filter.accept("Hey! How are you doing today?").emit()).isEqualTo("Hey! How are you doing today?");
		EosOutputFilter.Outcome o = filter.accept("\n<|user|>\nI'm good, thanks.");
		assertThat(o.stop()).isTrue();
		assertThat(o.emit()).isEqualTo("\n");
		assertThat(filter.text()).isEqualTo("Hey! How are you doing today?\n");
	}

	@Test
	@DisplayName("multi-token <|user|> never reaches emit")
	void multi_token_user_header_never_emitted() {
		EosOutputFilter filter = new EosOutputFilter();
		assertThat(filter.accept("done").emit()).isEqualTo("done");
		assertThat(filter.accept("<|").emit()).isEmpty();
		assertThat(filter.accept("user").emit()).isEmpty();
		EosOutputFilter.Outcome done = filter.accept("|>");
		assertThat(done.stop()).isTrue();
		assertThat(done.emit()).isEmpty();
		assertThat(filter.text()).isEqualTo("done");
	}

	@Test
	@DisplayName("marker split across pieces is still caught far into a long answer")
	void marker_found_after_long_answer() {
		EosOutputFilter filter = new EosOutputFilter();
		for (int i = 0; i < 200; i++)
			filter.accept("word ");
		assertThat(filter.accept("<|start_header").emit()).isEmpty();
		EosOutputFilter.Outcome o = filter.accept("_id|>user");
		assertThat(o.stop()).isTrue();
		assertThat(o.emit()).isEmpty();
		assertThat(filter.text()).hasSize(1000);
	}

	// ── Below a request's minimum the model's own markers must not end it ──────

	@ParameterizedTest
	@ValueSource(strings = { "</s>", "<|end|>", "<|user|>", "<|im_start|>", "<|endoftext|>" })
	@DisplayName("below the minimum a complete marker is emitted as text and does not stop")
	void marker_below_minimum_is_emitted_not_stopped(String marker) {
		EosOutputFilter filter = new EosOutputFilter();
		EosOutputFilter.Outcome o = filter.accept("Paris." + marker, false);
		assertThat(o.stop()).isFalse();
		assertThat(o.emit()).isEqualTo("Paris." + marker);
		assertThat(filter.text()).isEqualTo("Paris." + marker);
	}

	@Test
	@DisplayName("below the minimum a marker split across pieces is still emitted, not dropped")
	void split_marker_below_minimum_is_released() {
		EosOutputFilter filter = new EosOutputFilter();
		assertThat(filter.accept("done", false).emit()).isEqualTo("done");
		assertThat(filter.accept("<|", false).emit()).as("still a prefix: held").isEmpty();
		assertThat(filter.accept("user", false).emit()).isEmpty();
		EosOutputFilter.Outcome o = filter.accept("|>", false);
		assertThat(o.stop()).isFalse();
		assertThat(o.emit()).isEqualTo("<|user|>");
		assertThat(filter.text()).isEqualTo("done<|user|>");
	}

	@Test
	@DisplayName("a marker passed over below the minimum is not rediscovered once the minimum is met")
	void passed_over_marker_is_not_rediscovered() {
		EosOutputFilter filter = new EosOutputFilter();
		assertThat(filter.accept("a</s>", false).emit()).isEqualTo("a</s>");
		EosOutputFilter.Outcome o = filter.accept("b");
		assertThat(o.stop()).isFalse();
		assertThat(o.emit()).isEqualTo("b");
		assertThat(filter.text()).isEqualTo("a</s>b");
	}

	@Test
	@DisplayName("a marker completed once the minimum is met still stops and is stripped")
	void marker_after_minimum_still_stops() {
		EosOutputFilter filter = new EosOutputFilter();
		assertThat(filter.accept("a<|user|>", false).stop()).isFalse();
		assertThat(filter.accept("b<|", false).emit()).isEqualTo("b");
		EosOutputFilter.Outcome o = filter.accept("end|>");
		assertThat(o.stop()).isTrue();
		assertThat(o.emit()).isEmpty();
		assertThat(filter.text()).isEqualTo("a<|user|>b");
	}
}
