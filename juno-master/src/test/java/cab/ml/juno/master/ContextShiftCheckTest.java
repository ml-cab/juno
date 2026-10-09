package cab.ml.juno.master;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Path;
import java.util.EnumSet;
import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import cab.ml.juno.master.ModelLiveChecks.LiveCheck;
import cab.ml.juno.master.ModelLiveChecks.Suite;

/**
 * Check 10 on its own against TinyLlama, so the context-shift path through a real
 * model runs without the forked-cluster suites: shifts happen, no position reaches
 * the limit, and both schedules finish every requested token.
 */
@DisplayName("ModelLiveChecks - check 10, context shift (TinyLlama)")
class ContextShiftCheckTest {

	private static final Path MODEL = Path.of(System.getProperty("user.dir")).endsWith("juno-master")
			? Path.of("..", "models", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
			: Path.of("models", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");

	static boolean modelPresent() {
		return MODEL.toFile().exists();
	}

	@Test
	@EnabledIf("modelPresent")
	void contextShiftPastTheLimit() throws Exception {
		List<LiveCheck> results = ModelLiveChecks.run(MODEL.toString(), EnumSet.of(Suite.CONTEXT_SHIFT), System.out);
		assertThat(results).hasSize(1);
		LiveCheck c = results.get(0);
		assertThat(c.number()).isEqualTo(10);
		assertThat(c.passed()).as(c.detail()).isTrue();
	}
}
