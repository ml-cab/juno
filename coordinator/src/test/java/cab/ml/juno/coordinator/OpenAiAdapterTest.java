package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

class OpenAiAdapterTest {

	private static final ObjectMapper JSON = new ObjectMapper();

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

	@Test
	void parse_stop_string_and_array() {
		assertThat(OpenAiAdapter.parseStop(null)).isEmpty();
		assertThat(OpenAiAdapter.parseStop(JSON.getNodeFactory().textNode("END"))).containsExactly("END");
		ArrayNode arr = JSON.createArrayNode().add("a").add("b");
		assertThat(OpenAiAdapter.parseStop(arr)).containsExactly("a", "b");
	}

	@Test
	void parse_stop_rejects_too_many() {
		ArrayNode arr = JSON.createArrayNode().add("1").add("2").add("3").add("4").add("5");
		assertThatThrownBy(() -> OpenAiAdapter.parseStop(arr)).isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	void response_format_text_ok_json_rejected() {
		assertThat(OpenAiAdapter.validateResponseFormat(null)).isNull();
		ObjectNode text = JSON.createObjectNode().put("type", "text");
		assertThat(OpenAiAdapter.validateResponseFormat(text)).isNull();
		ObjectNode json = JSON.createObjectNode().put("type", "json_object");
		assertThat(OpenAiAdapter.validateResponseFormat(json)).contains("not supported");
	}
}
