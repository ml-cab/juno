package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.nio.file.Path;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * The loader must not send an architecture it has not verified to the dense
 * Llama-family handler: those tensor layouts differ (sliding-window attention,
 * logit softcapping, recurrent layers, routed experts), and a handler that finds
 * enough of the tensors it expects produces fluent-looking wrong output instead
 * of an error. The four architectures below are the ones that exist as real files
 * in the project's model directory; each was observed to reach the Llama handler.
 */
@DisplayName("ForwardPassHandlerLoader architecture guard")
class ForwardPassHandlerLoaderArchitectureTest {

	@TempDir
	Path dir;

	@ParameterizedTest(name = "{0} is rejected with a clear error")
	@ValueSource(strings = { "qwen35", "gemma4", "mistral3", "minimax-m2", "gemma", "deepseek2" })
	@DisplayName("unverified architectures fail closed before any tensor is read")
	void unverified_architecture_is_rejected(String architecture) throws IOException {
		Path gguf = MetadataOnlyGguf.write(dir, architecture);
		ShardContext context = new ShardContext("n0", 0, 1, true, true, 8, 8, 2);

		assertThatThrownBy(() -> ForwardPassHandlerLoader.load(gguf, context, CpuMatVec.INSTANCE))
				.isInstanceOf(IOException.class).hasMessageContaining("Unsupported model architecture")
				.hasMessageContaining("'" + architecture + "'").hasMessageContaining("llama");
	}

	@ParameterizedTest(name = "{0} passes the guard")
	@ValueSource(strings = { "llama", "mistral", "tinyllama", "qwen2", "qwen2.5" })
	@DisplayName("verified Llama-family architectures pass")
	void verified_architecture_passes(String architecture) {
		assertThat(LlamaFamilyArchitectures.isVerified(architecture)).isTrue();
		assertThatCode(() -> LlamaFamilyArchitectures.requireVerified(architecture, Path.of("model.gguf")))
				.doesNotThrowAnyException();
	}

	@ParameterizedTest(name = "{0} is not verified")
	@ValueSource(strings = { "qwen35", "gemma4", "mistral3", "minimax-m2", "phi3", "qwen3moe", "" })
	@DisplayName("everything outside the verified set is not verified")
	void unverified_architecture_is_not_verified(String architecture) {
		assertThat(LlamaFamilyArchitectures.isVerified(architecture)).isFalse();
	}

	@ParameterizedTest(name = "{0} is supported")
	@ValueSource(strings = { "phi2", "phi3", "qwen3", "qwen3moe", "llama", "mistral", "tinyllama", "qwen2", "qwen2.5" })
	@DisplayName("isSupportedArchitecture covers every dispatched architecture")
	void supported_architecture_predicate(String architecture) {
		assertThat(ForwardPassHandlerLoader.isSupportedArchitecture(architecture)).isTrue();
	}

	@ParameterizedTest(name = "{0} is not supported")
	@ValueSource(strings = { "qwen35", "gemma4", "mistral3", "minimax-m2" })
	@DisplayName("isSupportedArchitecture is false for the rejected real-file architectures")
	void unsupported_architecture_predicate(String architecture) {
		assertThat(ForwardPassHandlerLoader.isSupportedArchitecture(architecture)).isFalse();
	}
}
