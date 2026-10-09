package cab.ml.juno.master;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import cab.ml.juno.coordinator.GenerationLoop;
import cab.ml.juno.coordinator.GenerationResult;
import cab.ml.juno.coordinator.InferenceRequest;
import cab.ml.juno.coordinator.PrefillMode;
import cab.ml.juno.coordinator.RequestPriority;
import cab.ml.juno.coordinator.TokenConsumer;
import cab.ml.juno.node.CudaAvailability;
import cab.ml.juno.node.ForwardPassHandler;
import cab.ml.juno.node.GgufReader;
import cab.ml.juno.node.GpuContext;
import cab.ml.juno.node.LlamaConfig;
import cab.ml.juno.node.LocalInferencePipeline;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.GgufTokenizer;

/**
 * Phi-3.5-mini at its real limit of 4096 positions (the long RoPE factors are held
 * back): a request held open past it continues with context shifting on, and the
 * same request without the opt-in still fails there with the position error. CUDA
 * only, where a 4000-token prefill takes seconds rather than most of an hour.
 */
@Tag("gpu")
@DisplayName("Phi-3.5-mini - context shift at its 4096-token limit")
class Phi3ContextShiftAtLimitTest {

	private static final Path MODEL = Path.of(System.getProperty("user.dir")).endsWith("juno-master")
			? Path.of("..", "models", "Phi-3.5-mini-instruct-Q4_K_M.gguf")
			: Path.of("models", "Phi-3.5-mini-instruct-Q4_K_M.gguf");
	private static final int PROMPT_TOKENS = 3950;
	private static final int GENERATED = 200;

	@Test
	void shiftsAtTheLimitAndFailsThereWithoutTheOptIn() throws Exception {
		assumeTrue(MODEL.toFile().exists(), "Phi-3.5-mini not present");
		assumeTrue(CudaAvailability.isAvailable(), "CUDA not available");
		LlamaConfig cfg;
		GgufTokenizer tokenizer;
		try (GgufReader r = GgufReader.open(MODEL)) {
			cfg = LlamaConfig.from(r);
			tokenizer = GgufTokenizer.load(r);
		}
		List<ChatMessage> chat = List.of(ChatMessage.system("You are a careful assistant."),
				ChatMessage.user(ModelLiveChecks.longPrompt(tokenizer, PROMPT_TOKENS)));
		SamplingParams params = SamplingParams.deterministic().withMaxTokens(GENERATED).withMinTokens(GENERATED);

		try (GpuContext gpu = GpuContext.init(0)) {
			ForwardPassHandler h = ModelLiveChecks.loadSingleShard(MODEL.toString(), cfg, gpu.createMatVec());
			try {
				LocalInferencePipeline pipeline = ModelLiveChecks.singleShardPipeline(cfg, h);
				assertThat(pipeline.contextLimit()).isEqualTo(4096);
				GenerationLoop loop = new GenerationLoop(tokenizer, Sampler.create(), pipeline,
						ModelLiveChecks.newKvCache(4096), PrefillMode.BATCHED, 4096);

				InferenceRequest base = InferenceRequest.of("phi3", chat, params, RequestPriority.NORMAL);
				GenerationResult shifted = loop.generate(base.withContextShift(true), TokenConsumer.discard());
				System.out.printf("Phi-3.5 at 4096: prompt_tokens=%d generated=%d%n", shifted.promptTokens(),
						shifted.generatedTokens());
				assertThat(shifted.promptTokens() + shifted.generatedTokens()).as("crossed the limit")
						.isGreaterThan(4096);
				assertThat(shifted.generatedTokens()).isEqualTo(GENERATED);
				assertThat(ModelLiveChecks.cleanText(shifted.text())).isNotEmpty();

				assertThatThrownBy(() -> loop.generate(base.withContextShift(false), TokenConsumer.discard()))
						.hasMessageContaining("original context length");
			} finally {
				h.releaseGpuResources();
			}
		}
	}
}
