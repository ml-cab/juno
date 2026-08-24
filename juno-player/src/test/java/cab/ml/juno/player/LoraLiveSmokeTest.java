package cab.ml.juno.player;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import org.junit.jupiter.api.io.TempDir;

import cab.ml.juno.lora.LoraAdapterConfig;
import cab.ml.juno.node.LoraProjection;
import cab.ml.juno.node.LoraTrainDevice;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;

/**
 * Gated live LoRA smokes (Tier 10). Each test enables only when its GGUF fixture
 * exists under {@code models/}. Missing fixtures skip — they must not fail CI.
 */
@DisplayName("LoRA live smoke (gated fixtures)")
class LoraLiveSmokeTest {

	private static final int RANK = 8;
	private static final float ALPHA = 16f;
	private static final double LR = 1e-3;
	private static final int MAX_ITERS = 3;
	private static final float LOSS_TARGET = 0.5f;

	private static Path modelsRoot() {
		Path cwd = Path.of(System.getProperty("user.dir"));
		if (cwd.endsWith("juno-player") || cwd.endsWith("node"))
			return cwd.getParent().resolve("models");
		return cwd.resolve("models");
	}

	private static Path model(String fileName) {
		return modelsRoot().resolve(fileName);
	}

	private static boolean tinyLlamaPresent() {
		return Files.isRegularFile(model("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"));
	}

	private static boolean qwen25Present() {
		return Files.isRegularFile(model("qwen2.5-3b-instruct-q4_k_m.gguf"));
	}

	private static boolean phi35Present() {
		return Files.isRegularFile(model("Phi-3.5-mini-instruct-Q4_K_M.gguf"));
	}

	private static boolean denseQwen3Present() {
		Path root = modelsRoot();
		if (!Files.isDirectory(root))
			return false;
		try (var stream = Files.list(root)) {
			return stream.anyMatch(p -> {
				String n = p.getFileName().toString().toLowerCase();
				return n.endsWith(".gguf") && n.contains("qwen3") && !n.contains("qwen3.5") && !n.contains("moe")
						&& !n.contains("a3b");
			});
		} catch (Exception e) {
			return false;
		}
	}

	private static Path denseQwen3Path() {
		Path root = modelsRoot();
		try (var stream = Files.list(root)) {
			return stream.filter(p -> {
				String n = p.getFileName().toString().toLowerCase();
				return n.endsWith(".gguf") && n.contains("qwen3") && !n.contains("qwen3.5") && !n.contains("moe")
						&& !n.contains("a3b");
			}).findFirst().orElseThrow();
		} catch (Exception e) {
			throw new IllegalStateException(e);
		}
	}

	@Test
	@EnabledIf("tinyLlamaPresent")
	@DisplayName("TinyLlama: train-qa few updates, save, playback recalls answer")
	void tinyLlama_smoke(@TempDir Path tmp) throws Exception {
		smoke(model("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"), tmp);
	}

	@Test
	@EnabledIf("qwen25Present")
	@DisplayName("Qwen2.5: train-qa few updates, save, playback recalls answer")
	void qwen25_smoke(@TempDir Path tmp) throws Exception {
		smoke(model("qwen2.5-3b-instruct-q4_k_m.gguf"), tmp);
	}

	@Test
	@EnabledIf("phi35Present")
	@DisplayName("Phi-3.5-mini: train-qa few updates, save, playback recalls answer")
	void phi35_smoke(@TempDir Path tmp) throws Exception {
		smoke(model("Phi-3.5-mini-instruct-Q4_K_M.gguf"), tmp);
	}

	@Test
	@EnabledIf("denseQwen3Present")
	@DisplayName("Dense Qwen3: train-qa few updates, save, playback recalls answer")
	void denseQwen3_smoke(@TempDir Path tmp) throws Exception {
		smoke(denseQwen3Path(), tmp);
	}

	private static void smoke(Path gguf, Path tmp) throws Exception {
		Path adapterPath = tmp.resolve("smoke.lora");
		String modelKey = ChatModelType.fromPath(gguf.toString());
		String question = "What is the name of the AI assistant?";
		String answer = "Orion";

		LoraTrainingConfig cfg = LoraTrainingConfig.builder()
				.adapterConfig(LoraAdapterConfig.legacy(RANK, ALPHA))
				.targets(LoraProjection.qv())
				.learningRate(LR)
				.trainDevice(LoraTrainDevice.CPU)
				.seed(42L)
				.build();

		float endLoss;
		try (LoraTrainer trainer = LoraTrainer.open(gguf, adapterPath, cfg)) {
			LoraTrainer.TrainUntilResult result = trainer.trainQaPairUntil(question, answer, modelKey, LOSS_TARGET,
					MAX_ITERS);
			assertThat(result.iterations()).isBetween(1, MAX_ITERS);
			endLoss = result.finalLoss();
			assertThat(endLoss).isFinite().isLessThan(20f);
			trainer.save();
		}
		assertThat(adapterPath).exists();

		SamplingParams params = SamplingParams.defaults().withMaxTokens(24).withTemperature(0f);
		try (JunoPlayer player = JunoPlayer.builder(gguf).nodeCount(1).useGpu(false).loraPlayPath(adapterPath)
				.samplingParams(params).build()) {
			String reply = player.chat(List.of(ChatMessage.user(question))).text();
			assertThat(reply).isNotBlank();
			assertThat(reply)
					.as("turn-end markers must not leak into playback text")
					.doesNotContain("</s>", "<|end|>", "<|eot_id|>", "<end_of_turn>", "<|im_end|>", "<|endoftext|>");
			String lower = reply.toLowerCase();
			boolean recalled = lower.contains("orion");
			boolean notCollapsed = reply.trim().length() >= 2 && !reply.chars().allMatch(c -> c == reply.charAt(0));
			assertThat(recalled || notCollapsed)
					.as("reply should recall Orion or at least avoid mode collapse: %s", reply)
					.isTrue();
		}
	}
}
