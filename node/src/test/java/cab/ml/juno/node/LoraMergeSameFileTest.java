package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import cab.ml.juno.lora.LoraAdapterSet;

/**
 * A merge writes a new file. Its output opens with truncation while the model is still being read, so an output
 * that is the model file (or the checkpoint) destroys it. The refusal compares files, not path strings: on Windows
 * {@code Model.GGUF} and {@code model.gguf} are one file, and on every system a link or a {@code ./} spelling is.
 */
@DisplayName("LoraMerge — output is an input file")
class LoraMergeSameFileTest {

	@TempDir
	Path dir;

	@Test
	@DisplayName("an output naming the model file is refused and the model is left as it was")
	void output_is_the_model_file() throws Exception {
		Path model = MetadataOnlyGguf.write(dir, "llama");
		byte[] before = Files.readAllBytes(model);
		Path lora = emptyCheckpoint();

		assertThatThrownBy(() -> LoraMerge.merge(model, lora, model)).isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("same file as the model").message().doesNotContain("\n");
		assertThat(Files.readAllBytes(model)).isEqualTo(before);
	}

	@Test
	@DisplayName("another spelling of the model file, and a hard link to it, are the same file")
	void other_spellings_of_the_model_file() throws Exception {
		Path model = MetadataOnlyGguf.write(dir, "llama");
		byte[] before = Files.readAllBytes(model);
		Path lora = emptyCheckpoint();
		Path dotted = dir.resolve(".").resolve(model.getFileName());
		Path link = Files.createLink(dir.resolve("link.gguf"), model);

		assertThatThrownBy(() -> LoraMerge.merge(model, lora, dotted)).isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("same file as the model");
		assertThatThrownBy(() -> LoraMerge.merge(model, lora, link)).isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("same file as the model");
		assertThat(Files.readAllBytes(model)).isEqualTo(before);
	}

	@Test
	@DisplayName("an output naming the checkpoint is refused and the checkpoint is left as it was")
	void output_is_the_checkpoint() throws Exception {
		Path model = MetadataOnlyGguf.write(dir, "llama");
		Path lora = emptyCheckpoint();
		byte[] before = Files.readAllBytes(lora);

		assertThatThrownBy(() -> LoraMerge.merge(model, lora, lora)).isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("same file as the LoRA checkpoint");
		assertThat(Files.readAllBytes(lora)).isEqualTo(before);
	}

	private Path emptyCheckpoint() throws Exception {
		Path lora = dir.resolve("adapter.lora");
		new LoraAdapterSet().save(lora);
		return lora;
	}
}
