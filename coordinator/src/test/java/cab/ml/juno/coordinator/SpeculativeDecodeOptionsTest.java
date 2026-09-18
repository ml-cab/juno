package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

import cab.ml.juno.coordinator.SpeculativeDecodeOptions.SpecType;

class SpeculativeDecodeOptionsTest {

	@Test
	void parseType_accepts_draft_simple() {
		assertThat(SpeculativeDecodeOptions.parseType("draft-simple")).isEqualTo(SpecType.DRAFT_SIMPLE);
	}

	@Test
	void resolve_with_draft_simple_and_model_draft_path_succeeds() {
		SpeculativeDecodeOptions opts = SpeculativeDecodeOptions.resolve("draft-simple", null, null,
				"/path/to/draft.gguf");

		assertThat(opts.specType()).isEqualTo(SpecType.DRAFT_SIMPLE);
		assertThat(opts.modelDraftPath()).isEqualTo("/path/to/draft.gguf");
		assertThat(opts.enabled()).isTrue();
	}

	@Test
	void resolve_with_draft_simple_and_no_model_draft_fails_closed() {
		assertThatThrownBy(() -> SpeculativeDecodeOptions.resolve("draft-simple", null, null, null))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("--model-draft");
	}

	@Test
	void three_arg_resolve_overload_never_carries_a_model_draft_path() {
		SpeculativeDecodeOptions opts = SpeculativeDecodeOptions.resolve("ngram-simple", null, null);

		assertThat(opts.specType()).isEqualTo(SpecType.NGRAM_SIMPLE);
		assertThat(opts.modelDraftPath()).isNull();
	}

	@Test
	void of_leaves_model_draft_path_null_by_default() {
		SpeculativeDecodeOptions opts = SpeculativeDecodeOptions.of(SpecType.NONE, 3, 4);
		assertThat(opts.modelDraftPath()).isNull();
	}
}
