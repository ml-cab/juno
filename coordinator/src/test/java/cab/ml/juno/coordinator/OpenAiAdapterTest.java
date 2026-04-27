package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class OpenAiAdapterTest {

	@Test
	void frequency_penalty_mapping() {
		assertThat(OpenAiAdapter.repetitionPenaltyFromFrequencyPenalty(0f)).isEqualTo(1.0f);
		assertThat(OpenAiAdapter.repetitionPenaltyFromFrequencyPenalty(2f)).isEqualTo(2.0f);
		assertThat(OpenAiAdapter.repetitionPenaltyFromFrequencyPenalty(1f)).isEqualTo(1.5f);
		assertThat(OpenAiAdapter.repetitionPenaltyFromFrequencyPenalty(-2f)).isEqualTo(1.0f);
	}

	@Test
	void n_validation_rejects_more_than_one() {
		assertThat(OpenAiAdapter.validateCompletionsN(null)).isNull();
		assertThat(OpenAiAdapter.validateCompletionsN(1)).isNull();
		assertThat(OpenAiAdapter.validateCompletionsN(2)).contains("n must be 1");
	}

	@Test
	void finish_reason_and_id_mapping() {
		assertThat(OpenAiAdapter.toOpenAiFinishReason(GenerationResult.StopReason.EOS_TOKEN)).isEqualTo("stop");
		assertThat(OpenAiAdapter.toOpenAiFinishReason(GenerationResult.StopReason.STOP_TOKEN)).isEqualTo("stop");
		assertThat(OpenAiAdapter.toOpenAiFinishReason(GenerationResult.StopReason.MAX_TOKENS)).isEqualTo("length");
		assertThat(OpenAiAdapter.toOpenAiFinishReason(GenerationResult.StopReason.ERROR)).isEqualTo("error");
		assertThat(OpenAiAdapter.chatCompletionId("550e8400-e29b-41d4-a716-446655440000"))
				.isEqualTo("chatcmpl-550e8400e29b41d4a716446655440000");
	}
}
