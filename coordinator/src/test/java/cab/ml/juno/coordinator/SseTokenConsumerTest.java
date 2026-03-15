package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

import cab.ml.juno.coordinator.SseTokenConsumer;

/**
 * Unit tests for SseTokenConsumer.
 *
 * Uses a List<String> as the SseEmitter — no Javalin needed. Verifies the JSON
 * structure of each emitted SSE data payload.
 */
class SseTokenConsumerTest {

	private final List<String> emitted = new ArrayList<>();
	private final SseTokenConsumer consumer = new SseTokenConsumer("req-123", emitted::add);

	@Test
	void each_token_emits_one_sse_event() {
		consumer.onToken("Hello", 9906, 0);
		consumer.onToken(" world", 1917, 1);

		assertThat(emitted).hasSize(2);
	}

	@Test
	void event_contains_token_text() {
		consumer.onToken("Hello", 9906, 0);
		assertThat(emitted.get(0)).contains("Hello");
	}

	@Test
	void event_contains_token_id() {
		consumer.onToken("Hello", 9906, 0);
		assertThat(emitted.get(0)).contains("9906");
	}

	@Test
	void event_contains_request_id() {
		consumer.onToken("x", 1, 0);
		assertThat(emitted.get(0)).contains("req-123");
	}

	@Test
	void mid_stream_events_have_is_complete_false() {
		consumer.onToken("Hello", 9906, 0);
		assertThat(emitted.get(0)).contains("\"isComplete\":false");
	}

	@Test
	void send_complete_emits_final_event_with_is_complete_true() {
		consumer.onToken("Hello", 9906, 0);
		consumer.sendComplete("stop");

		String finalEvent = emitted.get(1);
		assertThat(finalEvent).contains("\"isComplete\":true");
		assertThat(finalEvent).contains("\"finishReason\":\"stop\"");
	}

	@Test
	void send_complete_without_prior_tokens_emits_single_final_event() {
		consumer.sendComplete("length");

		assertThat(emitted).hasSize(1);
		assertThat(emitted.get(0)).contains("\"isComplete\":true");
		assertThat(emitted.get(0)).contains("\"finishReason\":\"length\"");
	}

	@Test
	void empty_token_piece_is_still_emitted() {
		// Special tokens decode to empty string — still need the event for position
		// tracking
		consumer.onToken("", 2, 0);
		assertThat(emitted).hasSize(1);
		assertThat(emitted.get(0)).contains("\"token\":\"\"");
	}

	@Test
	void token_text_with_quotes_is_escaped() {
		consumer.onToken("say \"hello\"", 42, 0);
		// JSON must be valid — quotes escaped
		assertThat(emitted.get(0)).contains("\\\"hello\\\"");
	}

	@Test
	void events_are_valid_json() {
		consumer.onToken("test", 1, 0);
		consumer.sendComplete("stop");

		for (String event : emitted) {
			assertThat(event).startsWith("{").endsWith("}");
			// Must have all required fields
			assertThat(event).contains("requestId");
			assertThat(event).contains("token");
			assertThat(event).contains("tokenId");
			assertThat(event).contains("isComplete");
		}
	}
}