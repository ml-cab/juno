package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;

import org.junit.jupiter.api.Test;

import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;

class InferenceRequestTest {

	private InferenceRequest valid() {
		return InferenceRequest.of("llama3-8b", List.of(ChatMessage.user("Hello")), SamplingParams.defaults(),
				RequestPriority.NORMAL);
	}

	@Test
	void factory_generates_unique_request_ids() {
		String id1 = valid().requestId();
		String id2 = valid().requestId();
		assertThat(id1).isNotEqualTo(id2);
	}

	@Test
	void high_priority_sorts_before_normal() {
		InferenceRequest high = InferenceRequest.of("m", List.of(ChatMessage.user("hi")), SamplingParams.defaults(),
				RequestPriority.HIGH);
		InferenceRequest normal = InferenceRequest.of("m", List.of(ChatMessage.user("hi")), SamplingParams.defaults(),
				RequestPriority.NORMAL);

		assertThat(high.compareTo(normal)).isLessThan(0);
	}

	@Test
	void same_priority_uses_fifo_ordering() throws InterruptedException {
		InferenceRequest first = InferenceRequest.of("m", List.of(ChatMessage.user("a")), SamplingParams.defaults(),
				RequestPriority.NORMAL);
		Thread.sleep(5); // ensure different timestamps
		InferenceRequest second = InferenceRequest.of("m", List.of(ChatMessage.user("b")), SamplingParams.defaults(),
				RequestPriority.NORMAL);

		assertThat(first.compareTo(second)).isLessThan(0); // first in = first out
	}

	@Test
	void messages_list_is_immutable() {
		InferenceRequest req = valid();
		assertThatThrownBy(() -> req.messages().add(ChatMessage.user("x")))
				.isInstanceOf(UnsupportedOperationException.class);
	}

	@Test
	void rejects_blank_model_id() {
		assertThatThrownBy(() -> InferenceRequest.of("  ", List.of(ChatMessage.user("hi")), SamplingParams.defaults(),
				RequestPriority.NORMAL)).isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	void rejects_empty_messages() {
		assertThatThrownBy(
				() -> InferenceRequest.of("model", List.of(), SamplingParams.defaults(), RequestPriority.NORMAL))
				.isInstanceOf(IllegalArgumentException.class);
	}
}
