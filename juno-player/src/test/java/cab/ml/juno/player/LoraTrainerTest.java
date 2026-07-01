package cab.ml.juno.player;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;

/**
 * Integration tests for loss-target LoRA training via {@link LoraTrainer}.
 *
 * <p>
 * Skipped when no model path is provided:
 *
 * <pre>
 * mvn test -pl juno-player -Dtest=LoraTrainerTest -Djuno.test.model=/path/to/tinyllama.Q4_K_M.gguf
 * </pre>
 */
class LoraTrainerTest {

	private static final float QA_LOSS_TARGET = 1.2f;
	private static final float TEXT_LOSS_TARGET = 1.8f;
	private static final int MAX_TRAIN_ITERS = 50;

	private static Path modelPath() {
		String raw = System.getProperty("juno.test.model");
		return raw != null && !raw.isBlank() ? Path.of(raw) : null;
	}

	@Test
	@DisplayName("trainQaPairUntil reaches target and loraPlay recalls answer")
	void trainQaPairUntilThenPlay(@TempDir Path tmpDir) throws Exception {
		Path model = modelPath();
		assumeTrue(model != null && Files.exists(model), "set -Djuno.test.model=/path/to/model.gguf to run");

		Path adapterPath = tmpDir.resolve("qa.lora");
		String modelKey = ChatModelType.fromPath(model.toString());

		try (LoraTrainer trainer = LoraTrainer.open(model, adapterPath, 8, 16f, 1e-4)) {
			LoraTrainer.TrainUntilResult result = trainer.trainQaPairUntil(
					"What is the name of the AI assistant?", "Orion", modelKey, QA_LOSS_TARGET, MAX_TRAIN_ITERS);
			assertThat(result.iterations()).isBetween(1, MAX_TRAIN_ITERS);
			assertThat(result.finalLoss()).isLessThanOrEqualTo(QA_LOSS_TARGET);
			assertThat(result.targetReached()).isTrue();
			trainer.save();
		}

		assertThat(adapterPath).exists();

		SamplingParams params = SamplingParams.defaults().withMaxTokens(32).withTemperature(0.1f);
		try (JunoPlayer player = JunoPlayer.builder(model).nodeCount(1).useGpu(false).loraPlayPath(adapterPath)
				.samplingParams(params).build()) {
			String reply = player.chat(List.of(ChatMessage.user("What is the name of the AI assistant?"))).text();
			assertThat(reply.toLowerCase()).contains("orion");
		}
	}

	@Test
	@DisplayName("trainRawTextUntil reaches target and loraPlay uses passage vocabulary")
	void trainRawTextUntilThenPlay(@TempDir Path tmpDir) throws Exception {
		Path model = modelPath();
		assumeTrue(model != null && Files.exists(model), "set -Djuno.test.model=/path/to/model.gguf to run");

		Path adapterPath = tmpDir.resolve("text.lora");
		String passage = "Helixa is a distributed inference engine for low-latency language model serving. "
				+ "Helixa supports tensor parallelism and dynamic batching. "
				+ "Helixa was created to make fast LLM inference accessible without specialized hardware.";

		try (LoraTrainer trainer = LoraTrainer.open(model, adapterPath, 8, 16f, 1e-4)) {
			LoraTrainer.TrainUntilResult result = trainer.trainRawTextUntil(passage, TEXT_LOSS_TARGET, MAX_TRAIN_ITERS,
					128);
			assertThat(result.iterations()).isBetween(1, MAX_TRAIN_ITERS);
			assertThat(result.finalLoss()).isLessThanOrEqualTo(TEXT_LOSS_TARGET);
			assertThat(result.targetReached()).isTrue();
			trainer.save();
		}

		assertThat(adapterPath).exists();

		SamplingParams params = SamplingParams.defaults().withMaxTokens(40).withTemperature(0.3f);
		try (JunoPlayer player = JunoPlayer.builder(model).nodeCount(1).useGpu(false).loraPlayPath(adapterPath)
				.samplingParams(params).build()) {
			String reply = player.chat(List.of(ChatMessage.user("Tell me about Helixa. What is it and what does it support?")))
					.text();
			assertThat(reply).isNotBlank();
			String replyLower = reply.toLowerCase();
			assertThat(replyLower.contains("helixa") || replyLower.contains("inference")
					|| replyLower.contains("latency") || replyLower.contains("language")
					|| replyLower.contains("parallelism")).isTrue();
		}
	}
}
