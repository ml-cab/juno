package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.ChatTemplateFormatter;
import cab.ml.juno.tokenizer.SimpleTokenizer;

/**
 * The prefix a shift keeps must contain the system prompt on every chat
 * template, including those that fold the system message into the first user
 * turn (Mistral, Phi-3), where the system message formatted alone renders
 * nothing of it.
 */
class ContextWindowKeepTest {

	private final SimpleTokenizer tokenizer = new SimpleTokenizer();

	@ParameterizedTest(name = "{0}")
	@ValueSource(strings = { "llama3", "mistral", "phi3", "tinyllama", "qwen3", "chatml" })
	void keptPrefixHoldsTheSystemPrompt(String model) {
		List<ChatMessage> chat = List.of(ChatMessage.system("zebra quartz system"), ChatMessage.user("hello there"),
				ChatMessage.assistant("fine"), ChatMessage.user("again"));
		InferenceRequest req = InferenceRequest.of(model, chat, SamplingParams.defaults(), RequestPriority.NORMAL);
		int[] prompt = tokenizer.encode(ChatTemplateFormatter.forModelType(model).format(chat));
		int keep = ContextWindow.keepTokens(tokenizer, req, prompt);

		// a word inside the system text: its last word is glued to the end-of-turn marker on
		// some templates, and this whitespace tokenizer would not find it on its own
		int marker = tokenizer.encode("quartz")[0];
		int at = -1;
		for (int i = 0; i < prompt.length && at < 0; i++)
			if (prompt[i] == marker)
				at = i;
		assertThat(at).as("the system prompt is in the prompt").isGreaterThanOrEqualTo(0);
		assertThat(keep).as("kept prefix %s of %s", keep, Arrays.toString(prompt)).isGreaterThan(at);
		int[] user = tokenizer.encode("again");
		assertThat(Arrays.copyOf(prompt, keep)).as("the latest user turn is not kept").doesNotContain(user[0]);
	}
}
