package cab.ml.juno.apiserver;

import static org.assertj.core.api.Assertions.assertThat;

import cab.ml.juno.coordinator.GenerationResult;

import org.junit.jupiter.api.Test;

class OpenAiAdapterTest {

	@Test
	void frequency_penalty_zero_is_no_extra_penalty() {
		assertThat(OpenAiAdapter.repetitionPenaltyFromFrequencyPenalty(0f)).isEqualTo(1.0f);
	}

	@Test
	void frequency_penalty_two_maps_to_two() {
		assertThat(OpenAiAdapter.repetitionPenaltyFromFrequencyPenalty(2f)).isEqualTo(2.0f);
	}

	@Test
	void frequency_penalty_negative_clamps_to_one() {
		assertThat(OpenAiAdapter.repetitionPenaltyFromFrequencyPenalty(-2f)).isEqualTo(1.0f);
	}

	@Test
	void frequency_penalty_one_maps_to_one_point_five() {
		assertThat(OpenAiAdapter.repetitionPenaltyFromFrequencyPenalty(1f)).isEqualTo(1.5f);
	}

	@Test
	void n_null_or_one_allowed() {
		assertThat(OpenAiAdapter.validateCompletionsN(null)).isNull();
		assertThat(OpenAiAdapter.validateCompletionsN(1)).isNull();
	}

	@Test
	void n_greater_than_one_rejected() {
		assertThat(OpenAiAdapter.validateCompletionsN(2)).contains("n must be 1");
	}

	@Test
	void finish_reason_maps_stop_length_error() {
		assertThat(OpenAiAdapter.toOpenAiFinishReason(GenerationResult.StopReason.EOS_TOKEN)).isEqualTo("stop");
		assertThat(OpenAiAdapter.toOpenAiFinishReason(GenerationResult.StopReason.STOP_TOKEN)).isEqualTo("stop");
		assertThat(OpenAiAdapter.toOpenAiFinishReason(GenerationResult.StopReason.MAX_TOKENS)).isEqualTo("length");
		assertThat(OpenAiAdapter.toOpenAiFinishReason(GenerationResult.StopReason.ERROR)).isEqualTo("error");
	}

	@Test
	void chat_completion_id_strips_hyphens_from_uuid() {
		assertThat(OpenAiAdapter.chatCompletionId("550e8400-e29b-41d4-a716-446655440000"))
				.isEqualTo("chatcmpl-550e8400e29b41d4a716446655440000");
	}
}
