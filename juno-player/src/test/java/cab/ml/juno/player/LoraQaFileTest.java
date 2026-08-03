package cab.ml.juno.player;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@DisplayName("LoraQaFile")
class LoraQaFileTest {

	@TempDir
	Path tmp;

	@Test
	@DisplayName("loads JSON array of Q/A pairs")
	void happy_path() throws Exception {
		Path file = write("facts.json", """
				[
				  {"Q": "What is my name?", "A": "Dima"},
				  {"Q": "Where do I live?", "A": "Kyiv"}
				]
				""");
		List<LoraQaFile.Pair> pairs = LoraQaFile.load(file);
		assertThat(pairs).containsExactly(
				new LoraQaFile.Pair("What is my name?", "Dima"),
				new LoraQaFile.Pair("Where do I live?", "Kyiv"));
	}

	@Test
	@DisplayName("rejects missing Q or A keys")
	void missing_keys() throws Exception {
		Path missingQ = write("missing-q.json", "[{\"A\": \"Dima\"}]");
		assertThatThrownBy(() -> LoraQaFile.load(missingQ))
				.isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("Q");

		Path missingA = write("missing-a.json", "[{\"Q\": \"What is my name?\"}]");
		assertThatThrownBy(() -> LoraQaFile.load(missingA))
				.isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("A");
	}

	@Test
	@DisplayName("rejects empty Q or A strings")
	void empty_strings() throws Exception {
		Path emptyQ = write("empty-q.json", "[{\"Q\": \"  \", \"A\": \"Dima\"}]");
		assertThatThrownBy(() -> LoraQaFile.load(emptyQ))
				.isInstanceOf(IllegalArgumentException.class);

		Path emptyA = write("empty-a.json", "[{\"Q\": \"What?\", \"A\": \"\"}]");
		assertThatThrownBy(() -> LoraQaFile.load(emptyA))
				.isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	@DisplayName("rejects non-array root")
	void non_array_root() throws Exception {
		Path file = write("object.json", "{\"Q\": \"What?\", \"A\": \"X\"}");
		assertThatThrownBy(() -> LoraQaFile.load(file))
				.isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("array");
	}

	@Test
	@DisplayName("rejects empty array")
	void empty_array() throws Exception {
		Path file = write("empty.json", "[]");
		assertThatThrownBy(() -> LoraQaFile.load(file))
				.isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("empty");
	}

	@Test
	@DisplayName("rejects non-.json extension")
	void wrong_extension() throws Exception {
		Path file = write("facts.txt", "[{\"Q\": \"What?\", \"A\": \"X\"}]");
		assertThatThrownBy(() -> LoraQaFile.load(file))
				.isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining(".json");
	}

	@Test
	@DisplayName("rejects missing file")
	void missing_file() {
		Path missing = tmp.resolve("nope.json");
		assertThatThrownBy(() -> LoraQaFile.load(missing))
				.isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("not found");
	}

	private Path write(String name, String contents) throws Exception {
		Path file = tmp.resolve(name);
		Files.writeString(file, contents);
		return file;
	}
}
